import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { GrokVoiceSession } from '@/lib/voice-live-grok'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { sanitizeTextForSpeech } from '@/lib/speech-text'
import { notify, notifyError } from '@/store/notifications'

import type { ConversationStatus } from './use-voice-conversation'

/** How long an accepted delegation may sit before the gateway shows the turn running.
 *  Voice turns with tools routinely exceed 15s — a short grace here aborted
 *  speak-back before Hermes finished (live: 25s turn, Eve silent). */
const SUBMIT_SETTLE_GRACE_MS = 180_000
/** After busy falls, wait this long for the assistant bubble to land before giving up. */
const POST_BUSY_SETTLE_MS = 3_000

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceLiveGrokConversationOptions {
  /** The open chat's real Hermes session id — adopted when it exists. A FRESH
   *  DRAFT has none yet: the voice session starts under its synthetic id
   *  (events route via the caller transport the backend captured at start) and
   *  re-keys onto the real id once the delegation's submit mints the session. */
  chatSessionId: () => null | string
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  /** Interrupt the in-flight Hermes turn (Stop-button seam). Fired when a new
   *  delegation supersedes one still running. */
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  /** Submit a Hermes turn: `text` is the user's last words (the bubble and the
   *  persisted row), `voiceContext` the recent spoken exchange for the model.
   *  This is the SINGLE prompt.submit caller for grok-live delegations — the
   *  backend never submits (single-submitter invariant, gpt-live parity). */
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
 * `voice.grok.delegation` event already carries the assembled `prompt` +
 * `context`, so there is no client-side transcript buffer or stop-word settle
 * timer here; the backend's per-utterance flush already ran before the event
 * arrived. The one client-side stop-word check left is on the DELEGATION prompt
 * itself (mirrors GPT-Live's belt-and-suspenders check in `onDelegation`).
 *
 * Renderer-submits design: THIS hook's onSubmit is the only prompt.submit
 * caller (the backend's old delegation sink was removed — it double-submitted
 * on an existing chat and no-oped on a fresh draft's synthetic id). When the
 * turn settles, the finished reply is spoken through `voice.grok.speak`.
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
  const idleSinceRef = useRef(0)
  const busyRef = useRef(busy)
  const delegationRef = useRef<null | string>(null)
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
      onDelegation: (delegationId, prompt, context) => {
        if (sessionRef.current !== session) {
          return
        }

        // A spoken stop command ends the conversation instead of becoming a turn.
        if (prompt && isVoiceStopCommand(prompt)) {
          void end()
          latest.current.onStopWord?.()

          return
        }

        // A newer request supersedes an in-flight turn: stop it so the answer
        // the voice speaks is for what the user asked last.
        if (busyRef.current) {
          void latest.current.onInterrupt?.()
        }

        setDelegation(delegationId)
        lastToolLabelRef.current = null
        turnObservedRef.current = false
        idleSinceRef.current = 0
        submittedAtRef.current = Date.now()
        latest.current.consumePendingResponse()
        refreshStatus('thinking')
        // THE single submit (renderer-submits design): exactly one Hermes turn
        // per delegation, submitted like a typed message — which also lazily
        // creates the chat session when this conversation started on a fresh
        // draft. After it resolves, adopt the (possibly newly minted) chat id
        // so the backend re-keys the voice bridge onto the real session.
        void Promise.resolve(latest.current.onSubmit(prompt, context))
          .then(() => {
            if (sessionRef.current === session) {
              session.useSessionId(latest.current.chatSessionId?.())
            }
          })
          .catch(error => {
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
      // Adopt the open chat's real session id when one exists (no polling: a
      // FRESH DRAFT has no id until its first submit, so waiting can never
      // produce one — the synthetic id routes via the caller transport the
      // backend captured at start, and the delegation submit re-keys later).
      session.useSessionId(latest.current.chatSessionId?.())
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

  // Drive the reply back into the voice: when the submitted turn settles, speak
  // the final reply through `voice.grok.speak` (xAI force_message — verbatim,
  // no model involvement). This effect only tracks turn settlement so the UI's
  // `thinking` state clears at the right time, same polling shape as GPT-Live.
  // eslint-disable-next-line no-restricted-syntax -- turn-coordination refs (delegation id), not atom mirrors
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

      // A tool-using Hermes turn routinely runs longer than SUBMIT_SETTLE_GRACE_MS.
      // Aborting speak-back on that timer is what made Eve stay silent after a
      // successful voice turn (live: 25s turn, zero voice.grok.speak).
      if (busyRef.current) {
        turnObservedRef.current = true
        idleSinceRef.current = 0

        return
      }

      const tool = latest.current.activeToolLabel?.() ?? null

      lastToolLabelRef.current = tool

      const response = latest.current.pendingResponse()

      if (response) {
        turnObservedRef.current = true
        idleSinceRef.current = 0

        if (!response.pending) {
          const spoken = sanitizeTextForSpeech(response.text).trim()

          if (spoken) {
            session.speak(spoken)
          }

          latest.current.consumePendingResponse()
          setDelegation(null)
          refreshStatus()
        }

        return
      }

      if (turnObservedRef.current) {
        if (!idleSinceRef.current) {
          idleSinceRef.current = Date.now()
        }

        if (Date.now() - idleSinceRef.current > POST_BUSY_SETTLE_MS) {
          setDelegation(null)
          refreshStatus()
        }

        return
      }

      if (Date.now() - submittedAtRef.current > SUBMIT_SETTLE_GRACE_MS) {
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
   * nudge primitive (the backend owns the ws) so this is a no-op today. */
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
