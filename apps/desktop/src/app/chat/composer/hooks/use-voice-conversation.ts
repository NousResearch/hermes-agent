import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { startThinkingSound, stopThinkingSound } from '@/lib/thinking-sound'
import { monitorSpeechDuringPlayback } from '@/lib/voice-barge-in'
import {
  markVoicePlaybackInterrupted,
  playSpeechText,
  type SpeechStreamSession,
  startSpeechStream,
  stopVoicePlayback
} from '@/lib/voice-playback'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'
import { $voicePlayback } from '@/store/voice-playback'

import { useMicRecorder } from './use-mic-recorder'

export type ConversationStatus = 'idle' | 'listening' | 'transcribing' | 'thinking' | 'speaking'

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceConversationOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  /** Interrupt the in-flight agent turn (the same seam as the Stop button).
   *  Fired when the user speaks while the model is still generating. */
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  onSubmit: (text: string) => Promise<void> | void
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  /** Awaited right before the mic is opened. Used to let the wake-word listener
   *  fully release the capture device first, so the two never contend. */
  beforeMicOpen?: () => Promise<void> | void
}

/** How long a barge-triggered interrupt may take to settle before we submit
 *  the captured utterance anyway. */
const INTERRUPT_SETTLE_TIMEOUT_MS = 5_000

export function useVoiceConversation({
  busy,
  enabled,
  onFatalError,
  onInterrupt,
  onStopWord,
  onSubmit,
  onTranscribeAudio,
  pendingResponse,
  consumePendingResponse,
  beforeMicOpen
}: VoiceConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { handle, level } = useMicRecorder(voiceCopy)
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const turnTimeoutRef = useRef<number | null>(null)
  const pendingStartRef = useRef(false)
  const turnClosingRef = useRef(false)
  const awaitingSpokenResponseRef = useRef(false)
  const responseIdRef = useRef<string | null>(null)
  const spokenSourceLengthRef = useRef(0)
  const speechSessionRef = useRef<null | SpeechStreamSession>(null)
  const stopBargeMonitorRef = useRef<(() => void) | null>(null)
  const bargeCapturePendingRef = useRef(false)
  const bargedRef = useRef(false)
  const speechStartSequenceRef = useRef(0)
  const enabledRef = useRef(enabled)
  const mutedRef = useRef(muted)
  const busyRef = useRef(busy)
  const statusRef = useRef<ConversationStatus>('idle')
  const wasEnabledRef = useRef(enabled)
  const onStopWordRef = useRef(onStopWord)
  const onInterruptRef = useRef(onInterrupt)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    onInterruptRef.current = onInterrupt
  }, [onInterrupt])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    onStopWordRef.current = onStopWord
  }, [onStopWord])

  const beforeMicOpenRef = useRef(beforeMicOpen)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    beforeMicOpenRef.current = beforeMicOpen
  }, [beforeMicOpen])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    mutedRef.current = muted
  }, [muted])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    statusRef.current = status
  }, [status])

  const clearTurnTimeout = () => {
    if (turnTimeoutRef.current) {
      window.clearTimeout(turnTimeoutRef.current)
      turnTimeoutRef.current = null
    }
  }

  const dropSpeechSession = () => {
    stopBargeMonitorRef.current?.()
    stopBargeMonitorRef.current = null
    bargeCapturePendingRef.current = false
    bargedRef.current = false
    speechSessionRef.current = null
    responseIdRef.current = null
    spokenSourceLengthRef.current = 0
  }

  const handleTurn = useCallback(
    async (forceTranscribe = false) => {
      if (turnClosingRef.current) {
        return
      }

      turnClosingRef.current = true
      clearTurnTimeout()
      setStatus('transcribing')

      try {
        const result = await handle.stop()

        if (!result || (!result.heardSpeech && !forceTranscribe) || !onTranscribeAudio) {
          if (enabledRef.current && !mutedRef.current && !busyRef.current && statusRef.current !== 'speaking') {
            pendingStartRef.current = true
          }

          setStatus('idle')

          return
        }

        try {
          const transcript = (await onTranscribeAudio(result.audio)).trim()

          if (!transcript) {
            if (enabledRef.current) {
              pendingStartRef.current = true
            }

            setStatus('idle')

            return
          }

          // A spoken "stop" (or "never mind", "goodbye", …) ends the
          // conversation instead of being submitted as a turn. Only whole-
          // utterance stop commands match, so "stop the container" still goes
          // through as a real request.
          if (isVoiceStopCommand(transcript)) {
            dropSpeechSession()
            setStatus('idle')
            onStopWordRef.current?.()

            return
          }

          awaitingSpokenResponseRef.current = true
          dropSpeechSession()
          await onSubmit(transcript)
          setStatus('thinking')
        } catch (error) {
          notifyError(error, voiceCopy.transcriptionFailed)

          if (enabledRef.current && !mutedRef.current && !busyRef.current) {
            pendingStartRef.current = true
          }

          setStatus('idle')
        }
      } finally {
        turnClosingRef.current = false
      }
    },
    [handle, onSubmit, onTranscribeAudio, voiceCopy.transcriptionFailed]
  )

  const startListening = useCallback(async () => {
    pendingStartRef.current = false

    if (!enabledRef.current || mutedRef.current || busyRef.current) {
      return
    }

    if (bargeCapturePendingRef.current) {
      return // the barge monitor is mid-capture and owns the mic
    }

    if (statusRef.current !== 'idle') {
      return
    }

    // Let the wake-word listener fully release the capture device before we
    // open ours — opening the mic while wake still holds it makes getUserMedia
    // fail (the "clicked voice but it never starts listening" bug).
    try {
      await beforeMicOpenRef.current?.()
    } catch {
      // A pause failure shouldn't block the user's explicit start.
    }

    // enabled/muted/busy or an interleaved turn may have changed while we waited.
    if (!enabledRef.current || mutedRef.current || busyRef.current || statusRef.current !== 'idle') {
      return
    }

    try {
      // VAD tuning mirrors `tools.voice_mode` defaults so the browser loop matches the CLI.
      await handle.start({
        silenceLevel: 0.075,
        silenceMs: 1_250,
        idleSilenceMs: 12_000,
        onError: error => {
          notifyError(error, voiceCopy.microphoneFailed)
          pendingStartRef.current = false
          onFatalError?.()
        },
        onSilence: () => void handleTurn()
      })
      setStatus('listening')
      // Clear any prior turn-timeout before arming a fresh one. Each listen
      // cycle reassigns turnTimeoutRef; without clearing first, a stale 60s
      // timer from an earlier cycle survives and later fires handleTurn() in
      // the middle of a new listen, cutting it short (or, after enough idle
      // re-listens, wedging the loop into a state it doesn't re-arm from).
      clearTurnTimeout()
      turnTimeoutRef.current = window.setTimeout(() => void handleTurn(), 60_000)
    } catch (error) {
      notifyError(error, voiceCopy.couldNotStartSession)
      pendingStartRef.current = false
      setStatus('idle')
      onFatalError?.()
    }
  }, [handle, handleTurn, onFatalError, voiceCopy.couldNotStartSession, voiceCopy.microphoneFailed])

  const settleAfterSpeech = useCallback(
    (barged: boolean) => {
      if (barged || !awaitingSpokenResponseRef.current) {
        awaitingSpokenResponseRef.current = false
        consumePendingResponse()
      }

      if (bargeCapturePendingRef.current) {
        // The barge monitor is still capturing the user's interruption — it
        // owns the next turn. Keep it alive and don't re-open the mic; the
        // utterance callback transcribes and submits when they go quiet.
        speechSessionRef.current = null
        responseIdRef.current = null
        spokenSourceLengthRef.current = 0
        setStatus('listening')

        return
      }

      dropSpeechSession()

      // If stopVoicePlayback() was called externally (Stop button, end), the
      // voice-playback sequence has advanced past what we captured at speech
      // start — don't auto-start the next sentence, the user chose to stop.
      const stoppedByUser =
        speechStartSequenceRef.current > 0 && $voicePlayback.get().sequence > speechStartSequenceRef.current

      speechStartSequenceRef.current = 0

      if (enabledRef.current && !stoppedByUser) {
        pendingStartRef.current = true
      }

      setStatus('idle')
    },
    [consumePendingResponse]
  )

  /**
   * Submit the utterance the barge monitor captured — the user's interruption
   * from its first syllable, no re-listen round trip. Empty/failed captures
   * fall back to normal listening.
   */
  const submitCapturedUtterance = useCallback(
    async (audio: Blob | null) => {
      const resumeListening = () => {
        if (enabledRef.current && !mutedRef.current) {
          pendingStartRef.current = true
        }

        setStatus('idle')
      }

      if (!audio || !onTranscribeAudio) {
        resumeListening()

        return
      }

      setStatus('transcribing')

      try {
        const transcript = (await onTranscribeAudio(audio)).trim()

        if (!transcript) {
          resumeListening()

          return
        }

        // A spoken stop command while barging means "stop everything" — the
        // turn/playback was already cut at trip time; now end the conversation
        // instead of submitting "stop" as a new prompt.
        if (isVoiceStopCommand(transcript)) {
          dropSpeechSession()
          setStatus('idle')
          onStopWordRef.current?.()

          return
        }

        // A generation-phase barge interrupted the in-flight turn; the submit
        // path refuses while `busy`, so wait for the interrupt to settle.
        const deadline = Date.now() + INTERRUPT_SETTLE_TIMEOUT_MS

        while (busyRef.current && Date.now() < deadline) {
          await new Promise(resolve => window.setTimeout(resolve, 100))
        }

        awaitingSpokenResponseRef.current = true
        dropSpeechSession()
        consumePendingResponse()
        await onSubmit(transcript)
        setStatus('thinking')
      } catch (error) {
        notifyError(error, voiceCopy.transcriptionFailed)
        resumeListening()
      }
    },
    [consumePendingResponse, onSubmit, onTranscribeAudio, voiceCopy.transcriptionFailed]
  )

  /**
   * Full-duplex barge-in monitor for the WHOLE agent turn: armed at submit,
   * live through generation (thinking) AND playback (speaking).
   *
   * - generation phase (`busy`): speech interrupts the in-flight turn via
   *   `onInterrupt` — the same seam as the Stop button — and cuts any TTS that
   *   managed to start, so the stale reply never speaks.
   * - playback phase: speech cuts playback and the captured interruption is
   *   transcribed and submitted as the next turn.
   *
   * Idempotent — one monitor owns the mic per turn; re-arming while one is
   * live is a no-op (the live/fallback speech paths and the turn-drive effect
   * all call this).
   */
  const ensureBargeMonitor = useCallback(() => {
    if (stopBargeMonitorRef.current) {
      return
    }

    stopBargeMonitorRef.current = monitorSpeechDuringPlayback({
      isPlaying: () => $voicePlayback.get().status === 'speaking',
      onSpeech: () => {
        bargeCapturePendingRef.current = true
        bargedRef.current = true
        markVoicePlaybackInterrupted()
        stopVoicePlayback()

        if (busyRef.current) {
          // Mid-generation: stop the in-flight turn so the captured utterance
          // becomes the next one instead of queueing behind a stale reply.
          void onInterruptRef.current?.()
        }
      },
      onUtterance: audio => {
        bargeCapturePendingRef.current = false
        stopBargeMonitorRef.current = null
        void submitCapturedUtterance(audio)
      }
    })
  }, [submitCapturedUtterance])

  /** Push any new reply text into the live session; finish when complete. */
  const feedSpeechSession = useCallback(
    (responseId: string) => {
      const session = speechSessionRef.current

      if (!session || responseIdRef.current !== responseId) {
        return
      }

      const response = pendingResponse()

      if (response && response.id === responseId) {
        if (response.text.length > spokenSourceLengthRef.current) {
          session.append(response.text.slice(spokenSourceLengthRef.current))
          spokenSourceLengthRef.current = response.text.length
        }

        if (!response.pending && !busyRef.current) {
          session.finish()
        }
      } else if (!busyRef.current) {
        // Reply consumed/vanished while we were speaking — close out the turn.
        session.finish()
      }
    },
    [pendingResponse]
  )

  /** Whole-text fallback: wait for the reply to complete, then speak it. */
  const awaitFallbackSpeech = useCallback(
    (responseId: string) => {
      const poll = () => {
        if (responseIdRef.current !== responseId) {
          return
        }

        const response = pendingResponse()

        if (!response || response.id !== responseId) {
          settleAfterSpeech(false)

          return
        }

        if (response.pending || busyRef.current) {
          window.setTimeout(poll, 250)

          return
        }

        // The full-duplex monitor is normally already live (armed at submit);
        // this is a safety net for read-aloud-style entries into the loop.
        ensureBargeMonitor()

        speechStartSequenceRef.current = $voicePlayback.get().sequence

        void playSpeechText(response.text, { source: 'voice-conversation' })
          .catch(error => notifyError(error, voiceCopy.playbackFailed))
          .finally(() => {
            if (responseIdRef.current === responseId) {
              awaitingSpokenResponseRef.current = false
              settleAfterSpeech(bargedRef.current)
            }
          })
      }

      poll()
    },
    [ensureBargeMonitor, pendingResponse, settleAfterSpeech, voiceCopy.playbackFailed]
  )

  /**
   * Live-speak the streaming reply: one speech session per response, fed
   * incremental text as the assistant generates it. Audio overlaps generation
   * — no wait for the full reply, no per-sentence gaps.
   */
  const openLiveSpeech = useCallback(
    (responseId: string) => {
      responseIdRef.current = responseId
      spokenSourceLengthRef.current = 0
      speechStartSequenceRef.current = $voicePlayback.get().sequence
      setStatus('speaking')

      // VAD barge-in: the user talking over the reply cuts playback, drops
      // the not-yet-spoken remainder, AND keeps capturing — the interruption
      // is transcribed from its first syllable instead of losing the opening
      // words to a mic re-open. Usually already live (armed at submit).
      ensureBargeMonitor()

      void (async () => {
        const session = await startSpeechStream({ source: 'voice-conversation' })

        // The session may resolve after the loop moved on (barge, disable).
        if (responseIdRef.current !== responseId) {
          if (session) {
            stopVoicePlayback()
          }

          return
        }

        if (!session) {
          // No streaming backend/provider: speak the whole reply once it lands.
          speechSessionRef.current = null
          awaitFallbackSpeech(responseId)

          return
        }

        speechSessionRef.current = session

        // Timer-driven feed: reply text flows into the session at delta rate
        // regardless of React render cadence.
        const feedTimer = window.setInterval(() => feedSpeechSession(responseId), 150)
        feedSpeechSession(responseId)

        const outcome = await session.done
        window.clearInterval(feedTimer)

        if (responseIdRef.current !== responseId) {
          return
        }

        if (outcome === 'fallback') {
          awaitFallbackSpeech(responseId)

          return
        }

        awaitingSpokenResponseRef.current = false
        settleAfterSpeech(bargedRef.current)
      })()
    },
    [awaitFallbackSpeech, ensureBargeMonitor, feedSpeechSession, settleAfterSpeech]
  )

  const start = useCallback(async () => {
    if (!onTranscribeAudio) {
      notify({
        kind: 'warning',
        title: voiceCopy.unavailable,
        message: voiceCopy.configureSpeechToText
      })
      onFatalError?.()

      return
    }

    setMuted(false)
    awaitingSpokenResponseRef.current = false
    dropSpeechSession()
    consumePendingResponse()
    pendingStartRef.current = true
    await startListening()
  }, [
    consumePendingResponse,
    onFatalError,
    onTranscribeAudio,
    startListening,
    voiceCopy.configureSpeechToText,
    voiceCopy.unavailable
  ])

  const end = useCallback(async () => {
    pendingStartRef.current = false
    clearTurnTimeout()
    stopVoicePlayback()
    handle.cancel()
    turnClosingRef.current = false
    awaitingSpokenResponseRef.current = false
    dropSpeechSession()
    consumePendingResponse()
    setMuted(false)
    setStatus('idle')
  }, [consumePendingResponse, handle])

  const stopTurn = useCallback(() => {
    if (statusRef.current === 'listening') {
      void handleTurn(true)
    }
  }, [handleTurn])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value

      if (next) {
        clearTurnTimeout()
        handle.cancel()
        setStatus('idle')
      } else if (enabledRef.current && !busyRef.current && statusRef.current === 'idle') {
        pendingStartRef.current = true
      }

      return next
    })
  }, [handle])

  useEffect(() => {
    if (!enabled) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code !== 'Space' || event.repeat || event.metaKey || event.ctrlKey || event.altKey) {
        return
      }

      if (statusRef.current !== 'listening') {
        return
      }

      event.preventDefault()
      stopTurn()
    }

    window.addEventListener('keydown', onKeyDown, { capture: true })

    return () => window.removeEventListener('keydown', onKeyDown, { capture: true })
  }, [enabled, stopTurn])

  // Ambient "thinking" sound: while the agent works (status 'thinking') no
  // audio flows, which reads as dead air mid-conversation. Calm bubble blips
  // fill the gap; they stop the INSTANT speech starts, the mic re-arms, or the
  // conversation ends. Gated by voice.thinking_sound + the shared sound mute.
  useEffect(() => {
    if (enabled && !muted && status === 'thinking') {
      startThinkingSound()

      return stopThinkingSound
    }

    stopThinkingSound()

    return undefined
  }, [enabled, muted, status])

  // Drive the loop: when a voice-submitted reply appears, open a live speech
  // session (which feeds itself from then on). Otherwise start listening when
  // idle between turns.
  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (!enabled || muted) {
      return
    }

    if (awaitingSpokenResponseRef.current && status !== 'speaking') {
      // Generation phase: the turn is in flight but no reply audio exists
      // yet. Keep the mic live so speech can interrupt the model mid-
      // generation (full-duplex) instead of going deaf until playback.
      if (status === 'thinking' && (busy || bargeCapturePendingRef.current)) {
        ensureBargeMonitor()
      }

      const response = pendingResponse()

      if (response) {
        openLiveSpeech(response.id)

        return
      }

      if (!busy && status === 'thinking' && !bargeCapturePendingRef.current) {
        // Turn finished without any speakable reply (tool-only, error). A
        // live barge capture owns the loop instead — it submits or resumes.
        awaitingSpokenResponseRef.current = false
        dropSpeechSession()
        pendingStartRef.current = true
        setStatus('idle')

        return
      }
    }

    if (busy || status !== 'idle') {
      return
    }

    if (pendingStartRef.current) {
      void startListening()
    }
  }, [busy, enabled, muted, ensureBargeMonitor, openLiveSpeech, pendingResponse, startListening, status])

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

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
