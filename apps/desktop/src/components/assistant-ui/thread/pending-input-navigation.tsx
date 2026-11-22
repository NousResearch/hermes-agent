import { useStore } from '@nanostores/react'
import {
  createContext,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useEffectEvent,
  useLayoutEffect,
  useMemo,
  useState
} from 'react'

import { useStoreSelector } from '@/lib/use-session-slice'
import { sessionClarifyRequest } from '@/store/clarify'
import { $connectionRequests } from '@/store/connection-request'
import { type ApprovalRequest, sessionApprovalRequest } from '@/store/prompts'
import { $setupBeatRequests } from '@/store/setup-beat'
import { $themePicks } from '@/store/theme-pick'
import { onThreadPageScrollRequest } from '@/store/thread-scroll'

type PendingKind = 'approval' | 'clarify' | 'connection' | 'setup' | 'theme'

interface PendingTarget {
  kind: PendingKind
  requestId: string
  identity: string
}

interface PendingVisibility {
  offscreen: boolean | null
  above: boolean
}

const unmeasured: PendingVisibility = { offscreen: null, above: false }

type NodeRef = RefObject<HTMLElement | null>

interface PaneNavigation {
  nodes: Map<string, Set<NodeRef>>
  registerPendingTarget: (kind: PendingKind, requestId: string, nodeRef: NodeRef) => () => void
  changed: (() => void) | null
  requestJump: (() => boolean) | null
  cancelReveal: (() => void) | null
  /** Spend the one automatic reveal if the card mounted after the request. */
  retryReveal: (() => void) | null
  requested: string | null
  seen: Set<string>
  publishVisibility: (visibility: PendingVisibility) => void
  firstTarget: string | null | undefined
  readingUntil: number
  readingDuringLoad: boolean
  transcriptKey: string | null
}

const PaneNavigationContext = createContext<PaneNavigation | null>(null)

const PendingTargetContext = createContext<{ active: boolean; target: PendingTarget | null }>({
  active: false,
  target: null
})

const VisibilityContext = createContext(unmeasured)

const legacyApprovalKeys = new WeakMap<ApprovalRequest, string>()
let nextLegacyApprovalKey = 0

export function approvalTargetKey(request: ApprovalRequest): string {
  if (request.requestId) {
    return request.requestId
  }

  if (request.serverRequestId) {
    return JSON.stringify(['server', request.serverRequestId])
  }

  // Legacy requests without wire IDs still share one request object across the pane and card.
  let key = legacyApprovalKeys.get(request)

  if (!key) {
    key = JSON.stringify(['legacy', request.command, ++nextLegacyApprovalKey])
    legacyApprovalKeys.set(request, key)
  }

  return key
}

const cardKey = (kind: PendingKind, requestId: string) => JSON.stringify([kind, requestId])

function createPaneNavigation(publishVisibility: PaneNavigation['publishVisibility']): PaneNavigation {
  const pane: PaneNavigation = {
    nodes: new Map(),
    registerPendingTarget(kind, requestId, nodeRef) {
      const key = cardKey(kind, requestId)
      const refs = pane.nodes.get(key) ?? new Set<NodeRef>()
      refs.add(nodeRef)
      pane.nodes.set(key, refs)
      pane.changed?.()
      pane.retryReveal?.()

      return () => {
        refs.delete(nodeRef)

        if (!refs.size) {
          pane.nodes.delete(key)
        }

        pane.changed?.()
      }
    },
    changed: null,
    requestJump: null,
    cancelReveal: null,
    retryReveal: null,
    requested: null,
    seen: new Set(),
    publishVisibility,
    firstTarget: undefined,
    readingUntil: 0,
    readingDuringLoad: false,
    transcriptKey: null
  }

  return pane
}

export function PendingInputNavigationProvider({
  active = true,
  children,
  sessionId
}: {
  active?: boolean
  children: ReactNode
  sessionId: string | null
}) {
  const [visibility, setVisibility] = useState(unmeasured)

  const [pane] = useState(() =>
    createPaneNavigation(next =>
      setVisibility(current =>
        current.offscreen === next.offscreen && current.above === next.above ? current : next
      )
    )
  )

  const approval = useStore(useMemo(() => sessionApprovalRequest(sessionId), [sessionId]))
  const clarify = useStore(useMemo(() => sessionClarifyRequest(sessionId), [sessionId]))
  const operation = useStoreSelector($connectionRequests, requests => requests[sessionId ?? ''])
  const setup = useStoreSelector($setupBeatRequests, requests => requests[sessionId ?? ''])
  const theme = useStoreSelector($themePicks, requests => requests[sessionId ?? ''])

  const kind: PendingKind = approval
    ? 'approval'
    : clarify?.questions.length
      ? 'clarify'
      : operation && !operation.settled
        ? 'connection'
        : setup
          ? 'setup'
          : 'theme'

  const requestId = {
    approval: approval ? approvalTargetKey(approval) : undefined,
    clarify: clarify?.requestId,
    connection: operation?.opId,
    setup: setup?.requestId,
    theme: theme?.requestId
  }[kind]

  const selection = useMemo(
    () => ({
      active,
      target:
        sessionId && requestId ? { kind, requestId, identity: JSON.stringify([sessionId, kind, requestId]) } : null
    }),
    [active, kind, requestId, sessionId]
  )

  return (
    <PaneNavigationContext.Provider value={pane}>
      <PendingTargetContext.Provider value={selection}>
        <VisibilityContext.Provider value={visibility}>{children}</VisibilityContext.Provider>
      </PendingTargetContext.Provider>
    </PaneNavigationContext.Provider>
  )
}

export function usePendingTargetRegistration(kind: PendingKind, requestId: string | undefined, nodeRef: NodeRef) {
  const pane = useContext(PaneNavigationContext)

  useLayoutEffect(
    () => (requestId ? pane?.registerPendingTarget(kind, requestId, nodeRef) : undefined),
    [kind, nodeRef, pane, requestId]
  )
}

export function usePendingTargetState() {
  const { target } = useContext(PendingTargetContext)
  const pane = useContext(PaneNavigationContext)
  const visibility = useContext(VisibilityContext)
  const requestPendingJump = useCallback(() => pane?.requestJump?.() ?? false, [pane])

  return { target, ...visibility, jump: requestPendingJump }
}

interface PendingScrollOwner {
  viewport: RefObject<HTMLElement | null>
  content: RefObject<HTMLElement | null>
  visible: boolean
  following: boolean
  transcriptKey: string | null
  ready: boolean
  historical: boolean
  returnToLatest: () => void
  blocked: (viewport: HTMLElement) => boolean
  measure: (node: HTMLElement, viewport: HTMLElement) => { visible: boolean; delta: number } | null
  move: (viewport: HTMLElement, delta: number) => void
}

const SETTLE_MS = 240
const SETTLE_FRAMES = 6

/** Reconcile a pane's request with its transcript. Only the scroll owner can move it. */
export function usePendingInputReveal({
  viewport,
  content,
  visible,
  following,
  transcriptKey,
  ready,
  historical,
  returnToLatest,
  blocked,
  measure,
  move
}: PendingScrollOwner) {
  const pane = useContext(PaneNavigationContext)
  const { active, target } = useContext(PendingTargetContext)

  const isFollowing = useEffectEvent(() => following)

  const recordReadingIntent = useEffectEvent(() => {
    if (!pane) {
      return
    }

    pane.readingDuringLoad ||= !ready
    pane.requested = null
    pane.readingUntil = performance.now() + SETTLE_MS
    pane.cancelReveal?.()
  })

  useLayoutEffect(() => {
    const el = viewport.current

    if (!pane || !el || !visible) {
      return
    }

    if (pane.transcriptKey !== transcriptKey) {
      pane.transcriptKey = transcriptKey
      pane.firstTarget = undefined
      pane.readingUntil = 0
      pane.readingDuringLoad = false
    }

    // Listen before the authoritative request or its projected row arrives.
    const record = () => recordReadingIntent()

    const wheel = (event: WheelEvent) => {
      if (event.deltaY) {
        record()
      }
    }

    const key = (event: KeyboardEvent) => {
      if (['ArrowUp', 'PageUp', 'Home'].includes(event.key) || (event.shiftKey && event.key === ' ')) {
        record()
      }
    }

    const selection = () => {
      if (blocked(el)) {
        record()
      }
    }

    const touchStart = () => {
      record()
      pane.readingUntil = Infinity
    }

    // The desktop PageUp shortcut can page the transcript from composer focus.
    const stopPageScroll = onThreadPageScrollRequest(direction => {
      if (direction < 0) {
        record()
      }
    }, transcriptKey)

    document.addEventListener('selectionchange', selection)
    el.addEventListener('wheel', wheel, { passive: true, capture: true })
    el.addEventListener('pointerdown', record, { passive: true, capture: true })
    el.addEventListener('keydown', key, true)
    el.addEventListener('touchstart', touchStart, { passive: true })
    el.addEventListener('touchend', record, { passive: true })
    el.addEventListener('touchcancel', record, { passive: true })

    return () => {
      stopPageScroll()
      document.removeEventListener('selectionchange', selection)
      el.removeEventListener('wheel', wheel, true)
      el.removeEventListener('pointerdown', record, true)
      el.removeEventListener('keydown', key, true)
      el.removeEventListener('touchstart', touchStart)
      el.removeEventListener('touchend', record)
      el.removeEventListener('touchcancel', record)

      if (pane.readingUntil === Infinity) {
        pane.readingUntil = 0
      }
    }
  }, [blocked, pane, transcriptKey, viewport, visible])

  useLayoutEffect(() => {
    const el = viewport.current
    const body = content.current

    if (!pane) {
      return
    }

    // Let a restored request reconcile once the transcript and its pane are ready.
    if (pane.firstTarget === undefined && el && body && visible && ready && active) {
      pane.firstTarget = target?.identity ?? null
    }

    if (!target || !el || !body || !visible) {
      if (!target) {
        pane.requested = null
      }

      pane.publishVisibility(unmeasured)

      return
    }

    const { identity, kind, requestId } = target
    const key = cardKey(kind, requestId)

    const find = () => {
      for (const ref of pane.nodes.get(key) ?? []) {
        if (ref.current && el.contains(ref.current)) {
          return ref.current
        }
      }

      return null
    }

    const readerBlocked = () => blocked(el) || pane.readingDuringLoad || pane.readingUntil > performance.now()

    const consume = () => {
      pane.readingDuringLoad = false
      pane.seen.add(identity)
    }

    const read = () => {
      const node = historical ? null : find()
      const bounds = node ? measure(node, el) : null
      pane.publishVisibility({ offscreen: node ? !bounds?.visible : null, above: (bounds?.delta ?? 0) < 0 })

      return bounds
    }

    let observing = true
    let frame = 0
    let correctResize: (() => void) | null = null

    let stop = () => {}

    const schedule = () => {
      if (observing && !frame) {
        frame = requestAnimationFrame(() => {
          frame = 0
          read()
        })
      }
    }

    pane.changed = schedule
    read()

    const sizes = new ResizeObserver(() => {
      correctResize?.()
      schedule()
    })

    sizes.observe(body)
    sizes.observe(el)
    el.addEventListener('scroll', schedule, { passive: true })

    const start = (explicit: boolean) => {
      stop()
      let raf = 0
      let frames = 0
      let done = false
      let aligning = false
      const deadline = performance.now() + SETTLE_MS
      let alignedHeight: number | null = null

      const timer = window.setTimeout(() => {
        consume()
        stop()
      }, SETTLE_MS)

      stop = () => {
        done = true
        cancelAnimationFrame(raf)
        clearTimeout(timer)
        correctResize = null
        pane.cancelReveal = null
      }

      pane.cancelReveal = () => {
        consume()
        stop()
      }

      const pass = () => {
        if (done) {
          return
        }

        if (performance.now() >= deadline || (!explicit && readerBlocked())) {
          pane.cancelReveal?.()

          return
        }

        const bounds = read()

        if (bounds) {
          consume()

          if (!explicit && !aligning && bounds.visible) {
            stop()

            return
          }

          aligning = true
          alignedHeight = el.scrollHeight

          move(el, bounds.delta)
          schedule()
        }
      }

      const tick = () => {
        pass()

        if (!done && ++frames < SETTLE_FRAMES) {
          raf = requestAnimationFrame(tick)
        } else if (!done && !aligning) {
          consume()
          stop()
        }
      }

      // The observer also delivers an initial notification. Spend the one
      // correction only when content height actually changes after alignment.
      correctResize = () => {
        if (alignedHeight !== null && el.scrollHeight !== alignedHeight) {
          correctResize = null
          pass()
        }
      }

      raf = requestAnimationFrame(tick)
    }

    pane.requestJump = () => {
      if (historical) {
        pane.requested = identity
        returnToLatest()

        return true
      }

      if (!find()) {
        return false
      }

      if (ready) {
        start(true)
      } else {
        pane.requested = identity
      }

      return true
    }

    if (pane.requested && pane.requested !== identity) {
      pane.requested = null
    }

    if (ready && !historical && pane.requested === identity) {
      pane.requested = null
      start(true)
    } else if (historical || readerBlocked()) {
      consume()
    } else if (ready && active && !pane.seen.has(identity)) {
      if (isFollowing() || pane.firstTarget === identity) {
        // The store can lead the DOM: a projected or hydrated card may mount a
        // beat after the request. Arm the one reveal instead of spending it on
        // an empty tree, and let the registration spend it.
        if (find()) {
          start(false)
        } else {
          pane.retryReveal = () => {
            if (!pane.seen.has(identity) && find()) {
              pane.retryReveal = null
              start(false)
            }
          }
        }
      } else {
        consume()
      }
    }

    return () => {
      observing = false
      pane.changed = null
      pane.requestJump = null
      pane.retryReveal = null
      stop()
      cancelAnimationFrame(frame)
      sizes.disconnect()
      el.removeEventListener('scroll', schedule)
    }
  }, [
    active,
    blocked,
    content,
    historical,
    measure,
    move,
    pane,
    ready,
    returnToLatest,
    target,
    transcriptKey,
    viewport,
    visible
  ])
}
