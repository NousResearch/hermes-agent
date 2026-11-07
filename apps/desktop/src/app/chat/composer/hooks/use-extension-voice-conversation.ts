import { useCallback, useEffect, useRef, useState } from 'react'

import {
  findLiveVoiceEngine,
  type LiveVoiceEngineHistoryMessage,
  type LiveVoiceEngineSession,
  liveVoiceEngineStatus
} from '@/contrib/voice-engines'
import { useI18n } from '@/i18n'
import { sanitizeTextForSpeech } from '@/lib/speech-text'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'

import { micError } from './use-mic-recorder'
import type { ConversationStatus } from './use-voice-conversation'
import { liveEndedMessage } from './use-voice-live-conversation'

/** How long an accepted delegation may sit before the gateway shows the turn running. */
const SUBMIT_SETTLE_GRACE_MS = 15_000
const SETTLE_TICK_MS = 200
/** Quiet after the last user caption before the utterance is judged as a whole. */
const UTTERANCE_SETTLE_MS = 1_500

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface ExtensionVoiceConversationOptions {
  /** The `<plugin>:<engine>` id latched when the conversation started. */
  engineId: null | string
  busy: boolean
  enabled: boolean
  sessionId: null | string
  onFatalError?: () => void
  /** The engine's `status()` said no at start: the composer falls back to chained. */
  onUnavailable?: () => void
  /** Interrupt the in-flight Hermes turn (Stop-button seam). */
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  /** Submit a Hermes turn (`prompt.submit`, surface `voice-live`). */
  onSubmit: (text: string, voiceContext: string) => Promise<void> | void
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  seedHistory: () => LiveVoiceEngineHistoryMessage[]
  beforeMicOpen?: () => Promise<void> | void
}

interface DelegationWaiter {
  reject: (error: Error) => void
  resolve: (text: string) => void
  submittedAt: number
  observed: boolean
  timer: null | number
}

// One extension call at a time across every composer: a second start while
// another window or tile holds the microphone is refused.
let activeCall: null | symbol = null

/**
 * Conversation engine for a live voice engine an extension registered on
 * `VOICE_ENGINE_AREA` — same public shape as `useVoiceConversation` and
 * `useVoiceLiveConversation`, so the composer mounts it from
 * `voice.voice_chat_mode`.
 *
 * The engine owns its transport and audio; this hook owns the call's life
 * (start, one call at a time, teardown on end/unmount/stop phrase) and the
 * delegation: `io.delegate` runs a normal Hermes turn on the chat and resolves
 * with the turn's final reply text (prepared for speech) once it settles.
 */
export function useExtensionVoiceConversation({
  engineId,
  busy,
  enabled,
  sessionId,
  onFatalError,
  onUnavailable,
  onInterrupt,
  onStopWord,
  onSubmit,
  pendingResponse,
  consumePendingResponse,
  seedHistory,
  beforeMicOpen
}: ExtensionVoiceConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  const sessionRef = useRef<LiveVoiceEngineSession | null>(null)
  const callRef = useRef<null | symbol>(null)
  const startEpochRef = useRef(0)
  const startingRef = useRef(false)
  const speakingRef = useRef(false)
  const waiterRef = useRef<DelegationWaiter | null>(null)
  const utteranceRef = useRef('')
  const utteranceTimerRef = useRef<null | number>(null)
  const enabledRef = useRef(enabled)
  const busyRef = useRef(busy)
  const wasEnabledRef = useRef(enabled)

  const latest = useRef({
    beforeMicOpen,
    consumePendingResponse,
    engineId,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    onUnavailable,
    pendingResponse,
    seedHistory,
    sessionId
  })

  latest.current = {
    beforeMicOpen,
    consumePendingResponse,
    engineId,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    onUnavailable,
    pendingResponse,
    seedHistory,
    sessionId
  }

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const refreshStatus = useCallback(() => {
    if (!sessionRef.current) {
      setStatus('idle')
    } else if (speakingRef.current) {
      setStatus('speaking')
    } else if (waiterRef.current) {
      setStatus('thinking')
    } else {
      setStatus('listening')
    }
  }, [])

  const settleWaiter = useCallback(
    (outcome: { error: Error } | { text: string }) => {
      const waiter = waiterRef.current

      if (!waiter) {
        return
      }

      waiterRef.current = null

      if (waiter.timer !== null) {
        window.clearInterval(waiter.timer)
      }

      if ('error' in outcome) {
        waiter.reject(outcome.error)
      } else {
        waiter.resolve(outcome.text)
      }

      refreshStatus()
    },
    [refreshStatus]
  )

  const end = useCallback(async () => {
    startEpochRef.current += 1
    startingRef.current = false

    if (utteranceTimerRef.current) {
      window.clearTimeout(utteranceTimerRef.current)
      utteranceTimerRef.current = null
    }

    utteranceRef.current = ''
    // An abandoned request answers empty: the engine is being stopped anyway.
    settleWaiter({ text: '' })
    const session = sessionRef.current
    sessionRef.current = null
    speakingRef.current = false

    if (callRef.current && activeCall === callRef.current) {
      activeCall = null
    }

    callRef.current = null
    setMuted(false)
    setLevel(0)
    setStatus('idle')
    await session?.stop().catch(() => undefined)
  }, [settleWaiter])

  const stopByVoice = useCallback(() => {
    void end()
    latest.current.onStopWord?.()
  }, [end])

  /** Wait for the submitted turn to settle and hand back its reply text. */
  const awaitReply = useCallback(
    (waiter: DelegationWaiter) => {
      const tick = () => {
        if (waiterRef.current !== waiter) {
          return
        }

        if (busyRef.current) {
          waiter.observed = true

          return
        }

        const response = latest.current.pendingResponse()

        if (response && !response.pending) {
          latest.current.consumePendingResponse()
          settleWaiter({ text: sanitizeTextForSpeech(response.text) })

          return
        }

        if (response) {
          waiter.observed = true

          return
        }

        // The submit ack lags: only read "idle and no reply" as settled once
        // the turn was seen running, or after the grace period.
        if (waiter.observed || Date.now() - waiter.submittedAt > SUBMIT_SETTLE_GRACE_MS) {
          settleWaiter({ text: '' })
        }
      }

      waiter.timer = window.setInterval(tick, SETTLE_TICK_MS)
    },
    [settleWaiter]
  )

  const delegate = useCallback(
    async (call: symbol, request: string, voiceContext: string): Promise<string> => {
      if (callRef.current !== call || !sessionRef.current) {
        throw new Error('voice conversation is not active')
      }

      const prompt = request.replace(/\s+/g, ' ').trim()

      if (!prompt) {
        return ''
      }

      if (isVoiceStopCommand(prompt)) {
        stopByVoice()

        return ''
      }

      // A newer request supersedes one still running: the answer the voice
      // speaks is for what the user asked last.
      settleWaiter({ text: '' })

      if (busyRef.current) {
        void latest.current.onInterrupt?.()
      }

      latest.current.consumePendingResponse()

      return new Promise<string>((resolve, reject) => {
        const waiter: DelegationWaiter = { observed: false, reject, resolve, submittedAt: Date.now(), timer: null }
        waiterRef.current = waiter
        refreshStatus()

        Promise.resolve(latest.current.onSubmit(prompt, voiceContext.trim())).then(
          () => awaitReply(waiter),
          error => {
            notifyError(error, voiceCopy.liveDelegationFailed)

            if (waiterRef.current === waiter) {
              settleWaiter({ error: error instanceof Error ? error : new Error(String(error)) })
            }
          }
        )
      })
    },
    [awaitReply, refreshStatus, settleWaiter, stopByVoice, voiceCopy]
  )

  const start = useCallback(async () => {
    if (sessionRef.current || startingRef.current) {
      return
    }

    const engine = findLiveVoiceEngine(latest.current.engineId)

    if (!engine) {
      notify({
        id: 'voice-live-unavailable',
        kind: 'warning',
        message: voiceCopy.extensionMissing(latest.current.engineId ?? '')
      })
      latest.current.onUnavailable?.()

      return
    }

    if (activeCall) {
      notifyError(new Error('another live voice call is already active'), voiceCopy.couldNotStartSession)
      latest.current.onFatalError?.()

      return
    }

    startingRef.current = true
    const epoch = ++startEpochRef.current
    const call = Symbol(engine.id)
    activeCall = call
    callRef.current = call
    setStatus('thinking')

    const abandoned = () => !enabledRef.current || startEpochRef.current !== epoch

    const release = () => {
      if (activeCall === call) {
        activeCall = null
      }

      if (callRef.current === call) {
        callRef.current = null
      }

      startingRef.current = false
    }

    const verdict = await liveVoiceEngineStatus(engine)

    if (abandoned()) {
      release()

      return
    }

    if (!verdict.available) {
      release()
      setStatus('idle')
      notify({
        id: 'voice-live-unavailable',
        kind: 'warning',
        message: voiceCopy.extensionUnavailable(engine.label, verdict.reason ?? t.composer.voiceEngineUnavailable)
      })
      latest.current.onUnavailable?.()

      return
    }

    try {
      await latest.current.beforeMicOpen?.()
    } catch {
      // A wake-pause failure must not block an explicit start.
    }

    if (abandoned()) {
      release()

      return
    }

    let session: LiveVoiceEngineSession

    try {
      session = await engine.engine.start({
        delegate: (request, voiceContext) => delegate(call, request, voiceContext),
        history: latest.current.seedHistory(),
        onClosed: reason => {
          if (callRef.current !== call) {
            return
          }

          void end()

          // The user hung up through the voice ("goodbye"): an ordinary end, no warning.
          if (reason === 'close_requested') {
            latest.current.onStopWord?.()

            return
          }

          notify({ kind: 'warning', message: liveEndedMessage(reason, null, voiceCopy), title: voiceCopy.liveEnded })
          latest.current.onFatalError?.()
        },
        onError: (message, fatal) => {
          if (callRef.current !== call) {
            return
          }

          notify({ kind: fatal ? 'error' : 'warning', message, title: engine.label })

          if (fatal) {
            void end()
            latest.current.onFatalError?.()
          }
        },
        onSpeakingChange: speaking => {
          if (callRef.current !== call) {
            return
          }

          speakingRef.current = speaking
          setLevel(speaking ? 0.6 : 0)
          refreshStatus()
        },
        // A bare "stop" is judged on the user's caption once the utterance
        // settles: the voice model may answer it itself and never delegate it.
        onTranscript: (speaker, text) => {
          if (speaker !== 'user' || callRef.current !== call) {
            return
          }

          utteranceRef.current += text

          if (utteranceTimerRef.current) {
            window.clearTimeout(utteranceTimerRef.current)
          }

          utteranceTimerRef.current = window.setTimeout(() => {
            utteranceTimerRef.current = null
            const utterance = utteranceRef.current
            utteranceRef.current = ''

            if (callRef.current === call && isVoiceStopCommand(utterance)) {
              stopByVoice()
            }
          }, UTTERANCE_SETTLE_MS)
        },
        sessionId: latest.current.sessionId
      })
    } catch (error) {
      release()

      if (startEpochRef.current === epoch) {
        notifyError(error instanceof DOMException ? micError(error, voiceCopy) : error, voiceCopy.couldNotStartSession)
        setStatus('idle')
        latest.current.onFatalError?.()
      }

      return
    }

    // Ended (or toggled off) while the engine was starting: close what it opened.
    if (abandoned() || callRef.current !== call) {
      release()
      await session.stop().catch(() => undefined)

      return
    }

    startingRef.current = false
    sessionRef.current = session
    setMuted(false)
    refreshStatus()
  }, [delegate, end, refreshStatus, stopByVoice, t, voiceCopy])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value
      sessionRef.current?.setMuted?.(next)

      return next
    })
  }, [])

  // Full duplex has no turn boundary to close.
  const stopTurn = useCallback(() => undefined, [])

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
