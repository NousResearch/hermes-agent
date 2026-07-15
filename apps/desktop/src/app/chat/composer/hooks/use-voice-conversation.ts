import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { type RealtimeTranscript, RealtimeVoiceSession, type RealtimeVoiceStatus } from '@/lib/realtime-voice-session'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { notify, notifyError } from '@/store/notifications'
import type { VoiceInputMode } from '@/store/voice-prefs'

import { useMicRecorder } from './use-mic-recorder'

export type ConversationStatus =
  | 'idle'
  | 'connecting'
  | 'listening'
  | 'transcribing'
  | 'thinking'
  | 'speaking'
  | 'error'

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface VoiceConversationOptions {
  busy: boolean
  enabled: boolean
  mode: VoiceInputMode
  onFatalError?: () => void
  onSubmit: (text: string) => Promise<void> | void
  onTranscribeAudio?: (audio: Blob) => Promise<string>
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  sessionId?: string | null
}

export function useVoiceConversation({
  busy,
  enabled,
  mode,
  onFatalError,
  onSubmit,
  onTranscribeAudio,
  pendingResponse,
  consumePendingResponse,
  sessionId
}: VoiceConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { handle, level } = useMicRecorder(voiceCopy)
  const micHandleRef = useRef(handle)
  micHandleRef.current = handle
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const realtimeSessionRef = useRef<RealtimeVoiceSession | null>(null)
  const realtimeSubmittingRef = useRef(false)
  const acceptedRealtimeTurnIdsRef = useRef(new Set<string>())
  const pendingRealtimeTurnsRef = useRef<RealtimeTranscript[]>([])
  const generatedBindingIdRef = useRef<string | null>(null)
  generatedBindingIdRef.current ??=
    typeof globalThis.crypto?.randomUUID === 'function'
      ? `desktop-voice-${globalThis.crypto.randomUUID()}`
      : `desktop-voice-${Date.now()}-${Math.random().toString(36).slice(2)}`
  const turnTimeoutRef = useRef<number | null>(null)
  const pendingStartRef = useRef(false)
  const turnClosingRef = useRef(false)
  const awaitingSpokenResponseRef = useRef(false)
  const responseIdRef = useRef<string | null>(null)
  const spokenSourceLengthRef = useRef(0)
  const speechBufferRef = useRef('')
  const enabledRef = useRef(enabled)
  const mutedRef = useRef(muted)
  const busyRef = useRef(busy)
  const statusRef = useRef<ConversationStatus>('idle')
  const wasEnabledRef = useRef(enabled)
  const modeRef = useRef(mode)

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    mutedRef.current = muted
  }, [muted])

  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  useEffect(() => {
    statusRef.current = status
  }, [status])

  useEffect(() => {
    modeRef.current = mode
  }, [mode])

  const clearTurnTimeout = () => {
    if (turnTimeoutRef.current) {
      window.clearTimeout(turnTimeoutRef.current)
      turnTimeoutRef.current = null
    }
  }

  const resetSpeechBuffer = () => {
    responseIdRef.current = null
    spokenSourceLengthRef.current = 0
    speechBufferRef.current = ''
  }

  const appendSpeechText = (text: string) => {
    if (!text) {
      return
    }

    speechBufferRef.current = `${speechBufferRef.current}${text}`
  }

  const takeSpeechChunk = (force = false): string | null => {
    const buffer = speechBufferRef.current.replace(/\s+/g, ' ').trim()

    if (!buffer) {
      speechBufferRef.current = ''

      return null
    }

    const sentence = buffer.match(/^(.+?[.!?。！？])(?:\s+|$)/)

    if (sentence?.[1] && (sentence[1].length >= 8 || force)) {
      const chunk = sentence[1].trim()
      speechBufferRef.current = buffer.slice(sentence[1].length).trim()

      return chunk
    }

    if (!force && buffer.length > 220) {
      const softBoundary = Math.max(
        buffer.lastIndexOf(', ', 180),
        buffer.lastIndexOf('; ', 180),
        buffer.lastIndexOf(': ', 180)
      )

      if (softBoundary > 80) {
        const chunk = buffer.slice(0, softBoundary + 1).trim()
        speechBufferRef.current = buffer.slice(softBoundary + 1).trim()

        return chunk
      }
    }

    if (!force) {
      return null
    }

    speechBufferRef.current = ''

    return buffer
  }

  const drainRealtimeTurns = useCallback(async () => {
    if (realtimeSubmittingRef.current || busyRef.current || !enabledRef.current || modeRef.current !== 'realtime') {
      return
    }

    const turn = pendingRealtimeTurnsRef.current.shift()

    if (!turn) {
      return
    }

    realtimeSubmittingRef.current = true
    awaitingSpokenResponseRef.current = true
    resetSpeechBuffer()
    setStatus('thinking')

    try {
      await onSubmit(turn.text)
    } catch (error) {
      awaitingSpokenResponseRef.current = false
      notifyError(error, voiceCopy.transcriptionFailed)
      setStatus('listening')
    } finally {
      realtimeSubmittingRef.current = false
    }
  }, [onSubmit, voiceCopy.transcriptionFailed])

  const acceptRealtimeTranscript = useCallback(
    (turn: RealtimeTranscript) => {
      if (acceptedRealtimeTurnIdsRef.current.has(turn.id)) {
        return
      }

      acceptedRealtimeTurnIdsRef.current.add(turn.id)
      pendingRealtimeTurnsRef.current.push(turn)
      stopVoicePlayback()
      awaitingSpokenResponseRef.current = false
      consumePendingResponse()
      resetSpeechBuffer()
      setStatus('thinking')
      void drainRealtimeTurns()
    },
    [consumePendingResponse, drainRealtimeTurns]
  )

  const applyRealtimeStatus = useCallback((next: RealtimeVoiceStatus) => {
    if (next === 'idle') {
      setStatus('idle')
    } else if (next === 'connecting') {
      setStatus('connecting')
    } else if (next === 'transcribing') {
      setStatus('transcribing')
    } else if (next === 'error') {
      setStatus('error')
    } else if (statusRef.current !== 'thinking' && statusRef.current !== 'speaking') {
      setStatus('listening')
    }
  }, [])

  const startRealtimeListening = useCallback(async () => {
    realtimeSessionRef.current?.disconnect()

    const realtime = new RealtimeVoiceSession({
      onError: error => {
        notifyError(error, voiceCopy.couldNotStartSession)
        setStatus('error')
      },
      onSpeechStarted: () => {
        // Barge-in owns only playback. Hermes tools keep running under the
        // existing explicit interrupt policy; the next transcript waits until
        // the current turn becomes submit-ready.
        stopVoicePlayback()
        awaitingSpokenResponseRef.current = false
        consumePendingResponse()
        resetSpeechBuffer()
        setStatus('listening')
      },
      onStatus: applyRealtimeStatus,
      onTranscript: acceptRealtimeTranscript
    })

    realtimeSessionRef.current = realtime
    await realtime.connect({ sessionId: sessionId || generatedBindingIdRef.current! })
    realtime.setMuted(mutedRef.current)
  }, [acceptRealtimeTranscript, applyRealtimeStatus, consumePendingResponse, sessionId, voiceCopy.couldNotStartSession])

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

          awaitingSpokenResponseRef.current = true
          resetSpeechBuffer()
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

    if (!enabledRef.current || mutedRef.current || busyRef.current || modeRef.current !== 'legacy') {
      return
    }

    if (statusRef.current !== 'idle') {
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
      turnTimeoutRef.current = window.setTimeout(() => void handleTurn(), 60_000)
    } catch (error) {
      notifyError(error, voiceCopy.couldNotStartSession)
      pendingStartRef.current = false
      setStatus('idle')
      onFatalError?.()
    }
  }, [handle, handleTurn, onFatalError, voiceCopy.couldNotStartSession, voiceCopy.microphoneFailed])

  const speak = useCallback(
    async (text: string) => {
      setStatus('speaking')

      try {
        await playSpeechText(text, { source: 'voice-conversation' })
      } catch (error) {
        notifyError(error, voiceCopy.playbackFailed)
      } finally {
        if (enabledRef.current) {
          if (modeRef.current === 'legacy') {
            pendingStartRef.current = true
            setStatus('idle')
          } else {
            setStatus('listening')
          }
        } else {
          setStatus('idle')
        }
      }
    },
    [voiceCopy.playbackFailed]
  )

  const start = useCallback(async () => {
    mutedRef.current = false
    setMuted(false)
    awaitingSpokenResponseRef.current = false
    acceptedRealtimeTurnIdsRef.current.clear()
    pendingRealtimeTurnsRef.current = []
    resetSpeechBuffer()
    consumePendingResponse()

    if (modeRef.current === 'realtime') {
      pendingStartRef.current = false
      await startRealtimeListening()

      return
    }

    if (!onTranscribeAudio) {
      notify({
        kind: 'warning',
        title: voiceCopy.unavailable,
        message: voiceCopy.configureSpeechToText
      })
      onFatalError?.()

      return
    }

    pendingStartRef.current = true
    await startListening()
  }, [
    consumePendingResponse,
    onFatalError,
    onTranscribeAudio,
    startRealtimeListening,
    startListening,
    voiceCopy.configureSpeechToText,
    voiceCopy.unavailable
  ])

  const end = useCallback(async () => {
    pendingStartRef.current = false
    clearTurnTimeout()
    stopVoicePlayback()
    realtimeSessionRef.current?.disconnect()
    realtimeSessionRef.current = null
    handle.cancel()
    turnClosingRef.current = false
    awaitingSpokenResponseRef.current = false
    resetSpeechBuffer()
    acceptedRealtimeTurnIdsRef.current.clear()
    pendingRealtimeTurnsRef.current = []
    realtimeSubmittingRef.current = false
    consumePendingResponse()
    setMuted(false)
    setStatus('idle')
  }, [consumePendingResponse, handle])

  const stopTurn = useCallback(() => {
    if (modeRef.current === 'realtime') {
      realtimeSessionRef.current?.cancelInput()

      return
    }

    if (statusRef.current === 'listening') {
      void handleTurn(true)
    }
  }, [handleTurn])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value

      if (next) {
        clearTurnTimeout()

        if (modeRef.current === 'realtime') {
          realtimeSessionRef.current?.setMuted(true)
        } else {
          handle.cancel()
        }

        setStatus('idle')
      } else if (enabledRef.current && !busyRef.current && statusRef.current === 'idle') {
        if (modeRef.current === 'realtime') {
          realtimeSessionRef.current?.setMuted(false)
          setStatus('listening')
        } else {
          pendingStartRef.current = true
        }
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

  // Drive the loop: after a voice-submitted turn, speak stable chunks as the
  // assistant stream grows. Otherwise start listening when idle between turns.
  useEffect(() => {
    if (!enabled || muted) {
      return
    }

    if (
      modeRef.current === 'realtime' &&
      !busy &&
      !realtimeSubmittingRef.current &&
      pendingRealtimeTurnsRef.current.length > 0
    ) {
      void drainRealtimeTurns()

      return
    }

    if (awaitingSpokenResponseRef.current && status !== 'speaking') {
      const response = pendingResponse()

      if (response) {
        if (response.id !== responseIdRef.current) {
          resetSpeechBuffer()
          responseIdRef.current = response.id
        }

        if (response.text.length > spokenSourceLengthRef.current) {
          appendSpeechText(response.text.slice(spokenSourceLengthRef.current))
          spokenSourceLengthRef.current = response.text.length
        }

        const chunk = takeSpeechChunk(!response.pending && !busy)

        if (chunk) {
          void speak(chunk)

          return
        }

        if (!response.pending && !busy) {
          awaitingSpokenResponseRef.current = false
          consumePendingResponse()
          resetSpeechBuffer()

          if (modeRef.current === 'legacy') {
            pendingStartRef.current = true
            setStatus('idle')
          } else {
            setStatus('listening')
          }

          return
        }
      }

      if (!busy && status === 'thinking') {
        awaitingSpokenResponseRef.current = false
        resetSpeechBuffer()

        if (modeRef.current === 'legacy') {
          pendingStartRef.current = true
          setStatus('idle')
        } else {
          setStatus('listening')
        }

        return
      }
    }

    if (busy || status !== 'idle') {
      return
    }

    if (pendingStartRef.current) {
      void startListening()
    }
  }, [busy, consumePendingResponse, drainRealtimeTurns, enabled, muted, pendingResponse, speak, startListening, status])

  useEffect(() => {
    if (enabled && !wasEnabledRef.current) {
      void start()
    }

    if (!enabled && wasEnabledRef.current) {
      void end()
    }

    wasEnabledRef.current = enabled
  }, [enabled, end, start])

  useEffect(
    () => () => {
      if (turnTimeoutRef.current) {
        window.clearTimeout(turnTimeoutRef.current)
        turnTimeoutRef.current = null
      }

      realtimeSessionRef.current?.disconnect()
      realtimeSessionRef.current = null
      micHandleRef.current.cancel()
      stopVoicePlayback()
    },
    []
  )

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
