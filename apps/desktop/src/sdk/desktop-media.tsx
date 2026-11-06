import { type ComponentType, type CSSProperties, useCallback, useEffect, useRef, useState } from 'react'

import { isElementInHiddenPane, usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { usePlayerAgentBridge } from '@/lib/player-agent-bridge'
import type { PlayerAgentOperation, PlayerAgentResponse } from '@/lib/player-context-contract'
import type { PlayerCommand, PlayerMediaJobCapabilities, PlayerSnapshot } from '@/lib/player-contract'
import { isPlayerSource } from '@/lib/player-services'

type SafeCommand = Exclude<PlayerCommand, { type: 'bounds' | 'release' }>
export interface MediaPlayerController {
  available: boolean
  snapshot: PlayerSnapshot | null
  canLink: boolean
  linked: boolean
  command(command: SafeCommand): Promise<PlayerSnapshot>
  link(): Promise<unknown>
  unlink(): Promise<unknown>
  request(operation: PlayerAgentOperation): Promise<PlayerAgentResponse>
  seedPrompt(text: string): Promise<boolean>
  setupMediaHelpers(): Promise<PlayerMediaJobCapabilities>
  revealMediaJob(jobId: string): Promise<boolean>
  clearLocalData(): Promise<PlayerSnapshot>
}
export interface DesktopMediaSurfaceProps {
  className?: string
  style?: CSSProperties
  hidden?: boolean
}
export interface DesktopMedia {
  usePlayer(): MediaPlayerController
  Surface: ComponentType<DesktopMediaSurfaceProps>
}

let activeSurface: { owner: symbol; lease: symbol } | null = null
// A floating guest survives Surface unmount. Retain its adapter's ownership
// until unload so teardown still reaches it when its pane is hidden.
let playerOwner: symbol | null = null
let boundsRevision = Date.now() * 1000

const OVERLAYS =
  '[data-slot="dialog-overlay"],[data-slot="dialog-content"],[data-slot="sheet-overlay"],[data-slot="sheet-content"],[data-slot="popover-content"],[data-slot="dropdown-menu-content"],[data-slot="select-content"],[data-slot="context-menu-content"]'

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

function validCommand(value: unknown): value is SafeCommand {
  if (!isRecord(value) || typeof value.type !== 'string' || value.type === 'bounds') {return false}

  const allowed: Record<string, string[]> = {
    open: ['type', 'source'],
    status: ['type'],
    dock: ['type'],
    popout: ['type'],
    home: ['type'],
    back: ['type'],
    forward: ['type'],
    'media-control': ['type', 'action', 'positionSeconds'],
    pin: ['type', 'pinned'],
    fullscreen: ['type', 'fullscreen'],
    close: ['type']
  }

  const keys = allowed[value.type]

  if (!keys || !Object.keys(value).every(k => keys.includes(k))) {return false}

  switch (value.type) {
    case 'open':
      return Object.keys(value).length === 2 && isPlayerSource(value.source)

    case 'media-control':
      return (
        typeof value.action === 'string' &&
        ['play', 'pause', 'mute', 'unmute', 'seek'].includes(value.action) &&
        (value.action === 'seek'
          ? typeof value.positionSeconds === 'number' &&
            Number.isFinite(value.positionSeconds) &&
            value.positionSeconds >= 0 &&
            value.positionSeconds <= 86_400
          : value.positionSeconds === undefined)
      )

    case 'pin':

    case 'fullscreen':
      return typeof value[value.type === 'pin' ? 'pinned' : 'fullscreen'] === 'boolean'

    default:
      return Object.keys(value).length === 1
  }
}

function measure(slot: HTMLElement, paneVisible: boolean, hidden: boolean) {
  const rect = slot.getBoundingClientRect()
  const top = document.elementFromPoint?.(rect.left + rect.width / 2, rect.top + rect.height / 2)

  const occluded =
    Boolean(top && top !== slot && !slot.contains(top)) ||
    [...document.querySelectorAll<HTMLElement>(OVERLAYS)].some(el => {
      const r = el.getBoundingClientRect(),
        s = getComputedStyle(el)

      return (
        s.visibility !== 'hidden' &&
        s.display !== 'none' &&
        r.width > 0 &&
        r.height > 0 &&
        r.left < rect.right &&
        r.right > rect.left &&
        r.top < rect.bottom &&
        r.bottom > rect.top
      )
    })

  return {
    bounds: { x: rect.left, y: rect.top, width: Math.max(0, rect.width), height: Math.max(0, rect.height) },
    visible:
      paneVisible &&
      !hidden &&
      document.visibilityState === 'visible' &&
      !isElementInHiddenPane(slot) &&
      rect.width > 0 &&
      rect.height > 0 &&
      !occluded
  }
}

/** Private host adapter. The plugin receives safe verbs and a slot, never the native bridge. */
export function createDesktopMedia(pluginId: string, track: (dispose: () => void) => void): DesktopMedia {
  const owner = Symbol(pluginId)
  let disposed = false
  const releaseCallbacks = new Set<() => void>()
  let cached: PlayerSnapshot | null = null
  let stateVersion = 0
  const subscribers = new Set<(s: PlayerSnapshot | null) => void>()

  const publish = (next: PlayerSnapshot | null) => {
    if (disposed) {return}
    stateVersion++
    cached = next
    subscribers.forEach(fn => fn(next))
  }

  const dispose = () => {
    if (disposed) {return}
    disposed = true
    releaseCallbacks.forEach(fn => fn())
    releaseCallbacks.clear()

    if (activeSurface?.owner === owner) {activeSurface = null}
    subscribers.clear()

    if (playerOwner === owner) {
      playerOwner = null
      void window.hermesDesktop?.player?.request({ type: 'release' }).catch(() => undefined)
    }
  }

  track(dispose)

  function usePlayer(): MediaPlayerController {
    const agent = usePlayerAgentBridge()
    const [snapshot, setSnapshot] = useState<PlayerSnapshot | null>(cached)
    useEffect(() => {
      subscribers.add(setSnapshot)
      setSnapshot(cached)

      return () => {
        subscribers.delete(setSnapshot)
      }
    }, [])
    const bridge = typeof window === 'undefined' ? undefined : window.hermesDesktop?.player

    const command = useCallback(
      async (value: SafeCommand) => {
        if (!validCommand(value)) {throw new TypeError('Unsupported player command')}

        if (disposed) {throw new Error('Plugin media capability is disposed')}

        if (!bridge) {throw new Error('HermesTV player is unavailable')}

        if (value.type !== 'status') {
          if (playerOwner && playerOwner !== owner) {throw new Error('Player belongs to another plugin')}
          playerOwner = owner
        }

        const version = stateVersion
        const next = await bridge.request(value)

        if (disposed) {throw new Error('Plugin media capability is disposed')}

        if (version === stateVersion) {publish(next)}

        return next
      },
      [bridge]
    )

    const active = () => {
      if (disposed) {throw new Error('Plugin media capability is disposed')}
    }

    const runActive = async <T,>(run: () => Promise<T>) => {
      active()
      const value = await run()
      active()

      return value
    }

    return {
      available: Boolean(bridge),
      snapshot,
      canLink: Boolean(agent.scope && bridge),
      linked: agent.linked,
      command,
      link: () => runActive(agent.link),
      unlink: () => runActive(agent.unlink),
      request: operation => runActive(() => agent.request(operation)),
      seedPrompt: async text => {
        if (disposed) {return false}

        const inserted = await agent.seedPrompt(text)

        return !disposed && inserted
      },
      setupMediaHelpers: () => runActive(agent.setupMediaHelpers),
      revealMediaJob: id => (disposed ? Promise.resolve(false) : runActive(() => agent.revealMediaJob(id))),
      clearLocalData: () =>
        runActive(async () => {
          const s = await agent.clearLocalData()
          publish(s)

          return s
        })
    }
  }

  function Surface({ className, style, hidden = false }: DesktopMediaSurfaceProps) {
    const paneVisible = usePaneVisible()
    const slotRef = useRef<HTMLDivElement>(null)
    const lease = useRef(Symbol(pluginId))
    const frame = useRef<number | null>(null)
    const lastBounds = useRef('')
    const [current, setCurrent] = useState<PlayerSnapshot | null>(cached)
    const scheduleRef = useRef<() => void>(() => undefined)
    const [owned, setOwned] = useState(false)
    useEffect(() => {
      subscribers.add(setCurrent)
      setCurrent(cached)

      return () => {
        subscribers.delete(setCurrent)
      }
    }, [])

    const schedule = useCallback(() => {
      if (frame.current !== null) {cancelAnimationFrame(frame.current)}
      frame.current = requestAnimationFrame(() => {
        frame.current = null

        const slot = slotRef.current,
          bridge = window.hermesDesktop?.player

        if (!slot || !bridge || !owned || disposed || activeSurface?.lease !== lease.current) {return}
        const s = current

        const m =
          s && s.status !== 'closed' && s.guestId !== null
            ? measure(slot, paneVisible, hidden || s.presentation === 'floating')
            : { bounds: { x: 0, y: 0, width: 0, height: 0 }, visible: false }

        if (!s || s.status === 'closed' || s.guestId === null) {
          if (lastBounds.current) {
            lastBounds.current = ''
            void bridge.request({ type: 'bounds', ...m, revision: ++boundsRevision }).catch(() => undefined)
          }

          return
        }

        const key = JSON.stringify({ guestId: s.guestId, source: s.source, ...m })

        if (key === lastBounds.current) {return}
        lastBounds.current = key
        void bridge.request({ type: 'bounds', ...m, revision: ++boundsRevision }).catch(() => undefined)
      })
    }, [current, hidden, owned, paneVisible])

    scheduleRef.current = schedule
    useResizeObserver(schedule, slotRef)
    useEffect(() => {
      const bridge = window.hermesDesktop?.player

      if (!bridge) {return}
      const unsub = bridge.onState(publish)
      const version = stateVersion
      void bridge
        .request({ type: 'status' })
        .then(next => {
          if (version === stateVersion) {publish(next)}
        })
        .catch(() => undefined)
      releaseCallbacks.add(unsub)

      return () => {
        unsub()
        releaseCallbacks.delete(unsub)
      }
    }, [])
    // Lifecycle cleanup clears an imperative animation-frame handle, not app state.
    // eslint-disable-next-line no-restricted-syntax
    useEffect(() => {
      if (disposed || (activeSurface && activeSurface.lease !== lease.current)) {
        setOwned(false)

        return
      }

      activeSurface = { owner, lease: lease.current }
      setOwned(true)
      scheduleRef.current()

      const release = () => {
        if (frame.current !== null) {cancelAnimationFrame(frame.current)}
        frame.current = null

        if (activeSurface?.lease !== lease.current) {return}
        activeSurface = null

        const bridge = window.hermesDesktop?.player,
          s = cached

        if (s && s.status !== 'closed' && s.guestId !== null && bridge)
          {void bridge
            .request({
              type: 'bounds',
              bounds: { x: 0, y: 0, width: 0, height: 0 },
              visible: false,
              revision: ++boundsRevision
            })
            .catch(() => undefined)}
      }

      releaseCallbacks.add(release)

      return () => {
        if (frame.current !== null) {cancelAnimationFrame(frame.current)}
        releaseCallbacks.delete(release)
        release()
      }
    }, [])
    useEffect(() => {
      schedule()
    }, [schedule, current?.guestId, current?.status, current?.presentation, hidden, paneVisible])
    useEffect(() => {
      const handler = () => schedule()

      let pointer: number | null = null,
        dragFrame: number | null = null

      const drag = () => {
        dragFrame = null

        if (pointer !== null) {
          schedule()
          dragFrame = requestAnimationFrame(drag)
        }
      }

      const begin = (e: PointerEvent) => {
        if (e.button !== 0 || !(e.target instanceof Element) || !e.target.closest('[role="separator"]')) {return}
        pointer = e.pointerId
        schedule()

        if (dragFrame === null) {dragFrame = requestAnimationFrame(drag)}
      }

      const finish = (e?: PointerEvent) => {
        if (pointer === null || (e && pointer !== e.pointerId)) {return}
        pointer = null

        if (dragFrame !== null) {cancelAnimationFrame(dragFrame)}
        dragFrame = null
        schedule()
      }

      const blur = () => finish()

      const observeRoots = (root: ParentNode) => {
        const roots = [
          ...(root instanceof Element && root.matches(OVERLAYS) ? [root] : []),
          ...root.querySelectorAll<HTMLElement>(OVERLAYS)
        ]

        roots.forEach(el => {
          observer.observe(el, { attributes: true, attributeFilter: ['class', 'data-state', 'style'] })

          if (el.parentElement) {observer.observe(el.parentElement, { childList: true })}
        })
      }

      const observer = new MutationObserver(records => {
        let relevant = records.some(r => r.type === 'attributes')

        for (const record of records)
          {if (record.type === 'childList') {
            const added = [...record.addedNodes].filter(
              n => n instanceof Element && (n.matches(OVERLAYS) || Boolean(n.querySelector(OVERLAYS)))
            )

            relevant ||=
              added.length > 0 ||
              [...record.removedNodes].some(
                n => n instanceof Element && (n.matches(OVERLAYS) || Boolean(n.querySelector(OVERLAYS)))
              )
            added.forEach(n => {
              if (n instanceof Element) {observeRoots(n)}
            })
          }}

        if (relevant) {schedule()}
      })

      observer.observe(document.body, { childList: true })
      observeRoots(document.body)
      window.addEventListener('resize', handler)
      window.addEventListener('scroll', handler, true)
      window.addEventListener('pointerdown', begin, true)
      window.addEventListener('pointerup', finish, true)
      window.addEventListener('pointercancel', finish, true)
      window.addEventListener('blur', blur)
      document.addEventListener('visibilitychange', handler)

      const cleanup = () => {
        finish()
        observer.disconnect()
        window.removeEventListener('resize', handler)
        window.removeEventListener('scroll', handler, true)
        window.removeEventListener('pointerdown', begin, true)
        window.removeEventListener('pointerup', finish, true)
        window.removeEventListener('pointercancel', finish, true)
        window.removeEventListener('blur', blur)
        document.removeEventListener('visibilitychange', handler)
      }

      releaseCallbacks.add(cleanup)

      return () => {
        cleanup()
        releaseCallbacks.delete(cleanup)
      }
    }, [schedule])

    return <div className={className} data-hermes-media-slot="true" hidden={hidden} ref={slotRef} style={style} />
  }

  return { usePlayer, Surface }
}
