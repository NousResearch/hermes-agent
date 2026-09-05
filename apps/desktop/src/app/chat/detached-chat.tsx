/**
 * A real chat, mounted inside a surface that isn't a pane.
 *
 * Give it a stored session id and it renders that conversation with the whole
 * chat stack — same thread, same tool cards, same streaming indicators, same
 * composer. It resumes the session onto a live runtime, claims it so the
 * transcript isn't evicted while it's on screen, and re-points itself when a
 * reaped runtime is recovered mid-turn.
 *
 * The lifecycle is deliberately thinner than a tile's. A tile is restored from
 * disk at boot into a tab the user expects to find, so it latches errors,
 * retries per reconnect and must never delete itself on an inconclusive
 * lookup. A detached chat is owned by whatever mounted it: that surface knows
 * how to make a new session if this one is gone, so the failure mode here is
 * "say so, offer retry" rather than a recovery protocol.
 *
 * Presentation is not here. The tree renders exactly as the workspace pane's
 * does, and a host that wants a different chat layout styles it from its own
 * container — the way HUD mode restyles this same tree through
 * `[data-hud-shell]` instead of asking for variant props.
 */

import { useStore } from '@nanostores/react'
import { atom } from 'nanostores'
import { type RefObject, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { CenteredThreadSpinner } from '@/components/assistant-ui/thread/status'
import { Button } from '@/components/ui/button'
import { bindDetachedSession, releaseDetachedSession } from '@/store/detached-sessions'
import { sessionAwaitingInput } from '@/store/prompts'
import { $gatewayState } from '@/store/session'
import { $sessionTileDelegateRevision, sessionTileDelegate } from '@/store/session-states'

import { $detachedNewSession } from './detached-new'
import { SessionChat } from './session-chat'
import { buildSessionView } from './session-view'

export interface DetachedSessionChatProps {
  /** The conversation to show. The host creates it and remembers this id. */
  storedSessionId: string
  /** The composer's model pill. Defaults on — same catalog as the main pane. */
  modelMenu?: boolean
  /** The stored id is gone (never persisted, reaped on refresh). The host
   *  owns minting a replacement — do not latch the "couldn't open" overlay. */
  onMissing?: () => void
  /** `/new` in this composer. When set, the slash stays on this surface
   *  instead of starting a workspace draft. */
  onNew?: () => void
}

/** After a turn the user was watching ends, linger so the last line doesn't
 *  vanish the instant busy flips. Opening the sheet is never this timer's job. */
const HOLD_MS = 1100

/**
 * Whether the host should keep the transcript up.
 *
 * The user opens it (focus). Graph tool calls, streaming, and busy must not
 * summon it — watching Hermes edit a node is the canvas, not a chat sheet
 * stealing the graph. Once they HAVE opened it, a live turn pins it for the
 * whole run (HUD lets an unfocused band fade after one hold window). A
 * blocking prompt is the one exception that may raise it: otherwise the ask
 * is neither readable nor clickable.
 */
function useHeldBand(
  wrap: RefObject<HTMLDivElement | null>,
  view: ReturnType<typeof buildSessionView>,
  runtimeId: null | string
): boolean {
  const busy = useStore(view.$busy)
  const awaiting = useStore(view.$awaitingResponse)
  const awaitingInput = useStore(useMemo(() => sessionAwaitingInput(runtimeId), [runtimeId]))
  const working = busy || awaiting

  const [userOpened, setUserOpened] = useState(false)
  const [grace, setGrace] = useState(false)
  const workingRef = useRef(working)
  const necessaryRef = useRef(awaitingInput)
  workingRef.current = working
  necessaryRef.current = awaitingInput

  useEffect(() => {
    // The wrap is `display:contents`, so listen on the document and ask the
    // node about its descendants rather than attaching to a box that isn't.
    const inside = (node: EventTarget | null) => Boolean(node && wrap.current?.contains(node as Node))

    const onIn = (e: FocusEvent) => {
      if (inside(e.target)) {
        setUserOpened(true)
      }
    }

    const onOut = (e: FocusEvent) => {
      queueMicrotask(() => {
        if (inside(document.activeElement) || inside(e.relatedTarget)) {
          return
        }

        if (workingRef.current || necessaryRef.current) {
          return
        }

        setUserOpened(false)
      })
    }

    document.addEventListener('focusin', onIn)
    document.addEventListener('focusout', onOut)

    return () => {
      document.removeEventListener('focusin', onIn)
      document.removeEventListener('focusout', onOut)
    }
  }, [wrap, runtimeId])

  useEffect(() => {
    if (working || awaitingInput) {
      setGrace(true)

      return
    }

    const timer = window.setTimeout(() => setGrace(false), HOLD_MS)

    return () => window.clearTimeout(timer)
  }, [working, awaitingInput])

  useEffect(() => {
    if (working || awaitingInput || grace) {
      return
    }

    if (!wrap.current?.contains(document.activeElement)) {
      setUserOpened(false)
    }
  }, [awaitingInput, grace, working, wrap])

  return awaitingInput || (userOpened && (working || grace))
}

function sessionIsGone(message: string): boolean {
  const text = message.toLowerCase()

  return text.includes('session not found') || text.includes('resume returned no session id')
}

export function DetachedSessionChat({ modelMenu, onMissing, onNew, storedSessionId }: DetachedSessionChatProps) {
  const [runtimeId, setRuntimeId] = useState<null | string>(null)
  const [error, setError] = useState<string | undefined>(undefined)

  const gatewayOpen = useStore($gatewayState) === 'open'
  const delegateRevision = useStore($sessionTileDelegateRevision)
  const resumingRef = useRef(false)

  // The view reads the runtime through this atom rather than closing over
  // state, so recovering onto a new runtime re-points the same view instead of
  // rebuilding it — which would remount the thread and throw away scroll.
  const $runtimeId = useRef(atom<null | string>(null)).current

  const view = useMemo(() => buildSessionView('detached', $runtimeId, storedSessionId), [$runtimeId, storedSessionId])

  const wrapRef = useRef<HTMLDivElement>(null)
  const held = useHeldBand(wrapRef, view, runtimeId)

  const claim = useCallback(
    (id: string) => {
      $runtimeId.set(id)
      setRuntimeId(id)
      bindDetachedSession(storedSessionId, id)
    },
    [$runtimeId, storedSessionId]
  )

  // Same gating as the tile and the primary's route resume: never fire
  // session.resume before the gateway is OPEN. A surface can easily mount
  // while it is still connecting, and an ungated resume rejects there.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!gatewayOpen || runtimeId || error || resumingRef.current) {
      return
    }

    const delegate = sessionTileDelegate()

    if (!delegate) {
      return
    }

    resumingRef.current = true

    delegate
      .resumeTile(storedSessionId)
      .then(claim)
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err)

        if (onMissing && sessionIsGone(message)) {
          onMissing()

          return
        }

        setError(message)
      })
      .finally(() => {
        resumingRef.current = false
      })
  }, [claim, delegateRevision, error, gatewayOpen, onMissing, runtimeId, storedSessionId])

  // The gateway (re)opening invalidates a latched error — it most likely came
  // from the previous connection. Clearing it retriggers the resume: one
  // bounded auto-retry per reconnect, as the tile does.
  useEffect(() => {
    if (gatewayOpen) {
      setError(undefined)
    }
  }, [gatewayOpen])

  // Releasing on unmount is what makes the transcript collectable again —
  // leaving the claim behind pins every session the surface ever showed.
  useEffect(() => () => releaseDetachedSession(storedSessionId), [storedSessionId])

  useEffect(() => {
    if (!onNew) {
      return
    }

    $detachedNewSession.set(onNew)

    return () => {
      if ($detachedNewSession.get() === onNew) {
        $detachedNewSession.set(null)
      }
    }
  }, [onNew])

  if (error) {
    return (
      <div className="grid h-full place-items-center p-4">
        <div className="max-w-[24rem] space-y-2 text-center font-mono text-[11px]">
          <div className="text-(--ui-danger,#f87171)">Couldn't open this conversation</div>
          <div className="break-words text-(--ui-text-quaternary)">{error}</div>
          <Button onClick={() => setError(undefined)} size="sm" variant="outline">
            Retry
          </Button>
        </div>
      </div>
    )
  }

  if (!runtimeId) {
    // The SAME session loader the primary thread shows (Thread's
    // loading === 'session' branch) — one loading language everywhere.
    return (
      <div className="relative h-full">
        <CenteredThreadSpinner />
      </div>
    )
  }

  return (
    // display:contents — a styling handle, not a box. It stays out of the
    // host's layout while giving its CSS an ancestor to key visibility off,
    // and the focus probe an element to ask about.
    <div className="contents" data-chat-held={held ? '' : undefined} ref={wrapRef}>
      <SessionChat
        modelMenu={modelMenu}
        onRuntimeBound={claim}
        runtimeId={runtimeId}
        storedSessionId={storedSessionId}
        view={view}
      />
    </div>
  )
}
