import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { GrokVoiceSession } from '@/lib/voice-live-grok'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'

import type { ConversationStatus } from './use-voice-conversation'

/** How long an accepted delegation may sit before the gateway shows the turn running. */
const SUBMIT_SETTLE_GRACE_MS = 15_000

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceLiveGrokConversationOptions {
  /** The open chat's real Hermes session id — the voice session must ride it
   * (backend routes events + delegation by this id; see GrokVoiceSession.useSessionId). */
  chatSessionId: () => null | string
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  /** Interrupt the in-flight Hermes turn (Stop-button seam). Fired when a new
   *  delegation supersedes one still running. */
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  /** Submit a Hermes turn: `text` is the user's last words (the bubble and the
   *  persisted row), `voiceContext` the recent spoken exchange for the model. */
  onSubmit: (text: string, voiceContext: string) => Promise<void> | void
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  /** Names of tools currently running in the turn (quiet progress for the voice). */
  activeToolLabel?: () => null | string
  beforeMicOpen?: () => Promise<void> | void
}

/**
 * Grok-Live conversation engine — same public shape as `useVoiceLiveConversation`
 * (GPT-Live) so the composer can mount either from `voice.voice_chat_mode`.
 *
 * Unlike GPT-Live, the backend owns the xAI websocket and does its own
 * transcript accumulation + utterance-settle judgement (SPEC §5) — a
 * `voice.grok.delegation` event already carries the assembled `context`, so
 * there is no client-side transcript buffer or stop-word settle timer here;
 * the backend's per-utterance flush already ran before the event arrived.
 * The one client-side stop-word check left is on the DELEGATION prompt
 * itself (mirrors GPT-Live's belt-and-suspenders check in `onDelegation`).
 */
export function useVoiceLiveGrokConversation({
  busy,
  chatSessionId,
  enabled,
  onFatalError,
  onInterrupt,
  onStopWord,
  onSubmit,
  pendingResponse,
  consumePendingResponse,
  activeToolLabel,
  beforeMicOpen
}: VoiceLiveGrokConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  const [activeDelegation, setActiveDelegation] = useState<null | string>(null)
  const sessionRef = useRef<null | GrokVoiceSession>(null)
  const startEpochRef = useRef(0)
  const startingRef = useRef(false)
  const turnObservedRef = useRef(false)
  const submittedAtRef = useRef(0)
  const busyRef = useRef(busy)
  const delegationRef = useRef<null | string>(null)
  const spokenLengthRef = useRef(0)
  const spokenResponseIdRef = useRef<null | string>(null)
  const lastToolLabelRef = useRef<null | string>(null)
  const wasEnabledRef = useRef(enabled)

  const latest = useRef({
    activeToolLabel,
    beforeMicOpen,
    chatSessionId,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse
  })

  latest.current = {
    activeToolLabel,
    beforeMicOpen,
    chatSessionId,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse
  }

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const setDelegation = useCallback((id: null | string) => {
    delegationRef.current = id
    setActiveDelegation(id)
  }, [])

  const refreshStatus = useCallback((state?: string) => {
    if (!sessionRef.current) {
      setStatus('idle')

      return
    }

    if (state === 'speaking') {
      setStatus('speaking')
    } else if (delegationRef.current) {
      setStatus('thinking')
    } else {
      setStatus('listening')
    }
  }, [])

  const end = useCallback(async () => {
    startEpochRef.current += 1
    startingRef.current = false

    const session = sessionRef.current
    sessionRef.current = null
    setDelegation(null)
    spokenResponseIdRef.current = null
    spokenLengthRef.current = 0
    session?.close()
    setMuted(false)
    setLevel(0)
    setStatus('idle')
  }, [setDelegation])

  const start = useCallback(async () => {
    if (sessionRef.current || startingRef.current) {
      return
    }

    startingRef.current = true
    const epoch = ++startEpochRef.current

    try {
      await latest.current.beforeMicOpen?.()
    } catch {
      // A wake-pause failure must not block an explicit start.
    }

    if (startEpochRef.current !== epoch) {
      startingRef.current = false

      return
    }

    const session = new GrokVoiceSession({
      onClosed: reason => {
        if (sessionRef.current !== session) {
          return
        }

        sessionRef.current = null
        setDelegation(null)
        setStatus('idle')

        if (reason !== 'close_requested') {
          notify({ kind: 'warning', message: reason, title: voiceCopy.liveEnded })
          latest.current.onFatalError?.()
        }
      },
      onDelegation: (delegationId, context) => {
        if (sessionRef.current !== session) {
          return
        }

        if (context && isVoiceStopCommand(context)) {
          void end()
          latest.current.onStopWord?.()

          return
        }

        if (busyRef.current) {
          void latest.current.onInterrupt?.()
        }

        setDelegation(delegationId)
        spokenResponseIdRef.current = null
        spokenLengthRef.current = 0
        lastToolLabelRef.current = null
        turnObservedRef.current = false
        submittedAtRef.current = Date.now()
        latest.current.consumePendingResponse()
        refreshStatus('thinking')
        void Promise.resolve(latest.current.onSubmit(context, context)).catch(error => {
          notifyError(error, voiceCopy.liveDelegationFailed)
          setDelegation(null)
          refreshStatus()
        })
      },
      onState: (state, reason) => {
        if (sessionRef.current !== session) {
          return
        }

        if (state === 'error' || state === 'degraded') {
          notify({ kind: state === 'error' ? 'error' : 'warning', message: reason ?? state, title: voiceCopy.liveError })
        }

        setLevel(state === 'speaking' ? 0.6 : 0)
        refreshStatus(state)
      }
    })

    sessionRef.current = session
    startingRef.current = false
    setMuted(false)
    setStatus('thinking')

    try {
      // The voice session must ride the open chat's real Hermes session id —
      // the backend routes voice.grok.* events and the delegation seam by it.
      // A wake-triggered fresh draft has NO id yet at this moment, so poll
      // briefly for it (the composer's sessionId prop arrives a beat after
      // startFreshSessionDraft); falling back to the synthetic id would route
      // every event to stdio and no-op the delegation — the wake path's
      // original silent-failure mode.
      const deadline = Date.now() + 10_000
      let sid = latest.current.chatSessionId?.()
      while (!sid && Date.now() < deadline && startEpochRef.current === epoch) {
        await new Promise(resolve => setTimeout(resolve, 150))
        sid = latest.current.chatSessionId?.()
      }
      session.useSessionId(sid)
      await session.start()

      if (sessionRef.current !== session || startEpochRef.current !== epoch) {
        session.close()

        return
      }

      refreshStatus('listening')
    } catch (error) {
      if (sessionRef.current === session) {
        sessionRef.current = null
      }

      session.close()

      if (startEpochRef.current !== epoch) {
        return
      }

      notifyError(error, voiceCopy.couldNotStartSession)
      setStatus('idle')
      latest.current.onFatalError?.()
    }
  }, [
    end,
    refreshStatus,
    setDelegation,
    voiceCopy.couldNotStartSession,
    voiceCopy.liveDelegationFailed,
    voiceCopy.liveEnded,
    voiceCopy.liveError
  ])

  // Drive the reply back into Hermes' progress note; the backend speaks the
  // final reply itself once submitted (SPEC §5's "speak arbitrary text" leg
  // lives server-side) — this effect only tracks turn settlement so the UI's
  // `thinking` state clears at the right time, same polling shape as GPT-Live.
  // eslint-disable-next-line no-restricted-syntax -- turn-coordination refs (delegation id / spoken cursor), not atom mirrors
  useEffect(() => {
    const session = sessionRef.current
    const delegationId = delegationRef.current

    if (!session || !delegationId) {
      return undefined
    }

    const tick = () => {
      if (sessionRef.current !== session || delegationRef.current !== delegationId) {
        return
      }

      if (busyRef.current) {
        turnObservedRef.current = true
      }

      const tool = latest.current.activeToolLabel?.() ?? null

      lastToolLabelRef.current = tool

      const response = latest.current.pendingResponse()

      if (response) {
        turnObservedRef.current = true

        if (!response.pending) {
          spokenResponseIdRef.current = response.id
          spokenLengthRef.current = response.text.length
          latest.current.consumePendingResponse()
          setDelegation(null)
          refreshStatus()
        }

        return
      }

      if (
        !busyRef.current &&
        (turnObservedRef.current || Date.now() - submittedAtRef.current > SUBMIT_SETTLE_GRACE_MS)
      ) {
        setDelegation(null)
        refreshStatus()
      }
    }

    const timer = window.setInterval(tick, 200)
    tick()

    return () => window.clearInterval(timer)
  }, [activeDelegation, busy, refreshStatus, setDelegation, status])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value
      sessionRef.current?.setMuted(next)

      return next
    })
  }, [])

  /** No explicit turn boundary in full duplex; grok-live has no client-side
   *  nudge primitive (the backend owns the ws) so this is a no-op today. */
  const stopTurn = useCallback(() => {
    // Intentionally empty: mirrors GPT-Live's public shape; grok-live's
    // backend judges utterance completion itself (SPEC §5).
  }, [])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      void start()
    }

    if (!enabled && wasEnabledRef.current) {
      void end()
    }

    wasEnabledRef.current = enabled
  }, [enabled, end, start])

  useEffect(() => () => void end(), [end])

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
