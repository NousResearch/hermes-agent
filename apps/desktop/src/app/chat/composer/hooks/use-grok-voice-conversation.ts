import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { GrokVoiceSession } from '@/lib/grok-voice'
import { sanitizeTextForSpeech } from '@/lib/speech-text'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'

import { useComposerScope } from '../scope'

import { micError } from './use-mic-recorder'
import type { ConversationStatus } from './use-voice-conversation'

/** How long an accepted delegation may sit before the gateway shows the turn running. */
const SUBMIT_SETTLE_GRACE_MS = 15_000
/** Quiet after the last user transcript before the utterance is judged as a whole. */
const UTTERANCE_SETTLE_MS = 1_500

interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

interface GrokVoiceConversationOptions {
  beforeMicOpen?: () => Promise<void> | void
  busy: boolean
  consumePendingResponse: () => void
  enabled: boolean
  onFatalError?: () => void
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  onSubmit: (text: string, voiceContext: string) => Promise<void> | void
  pendingResponse: () => null | PendingVoiceResponse
}

/**
 * Full-duplex Grok Voice. `hermes_task` is the same voice-live submit GPT-Live
 * uses; only the model's final text is spoken back, queued if it is still talking.
 */
export function useGrokVoiceConversation({
  beforeMicOpen,
  busy,
  consumePendingResponse,
  enabled,
  onFatalError,
  onInterrupt,
  onStopWord,
  onSubmit,
  pendingResponse
}: GrokVoiceConversationOptions): {
  end: () => Promise<void>
  level: number
  muted: boolean
  start: () => Promise<void>
  status: ConversationStatus
  stopTurn: () => void
  toggleMute: () => void
} {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice
  const { connectionId: ownerConnectionId, profile: ownerProfile } = useComposerScope()
  const ownerRef = useRef({ connectionId: ownerConnectionId, profile: ownerProfile })
  ownerRef.current = { connectionId: ownerConnectionId, profile: ownerProfile }

  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [level, setLevel] = useState(0)
  const [muted, setMuted] = useState(false)
  const [delegation, setDelegation] = useState<null | string>(null)

  const sessionRef = useRef<GrokVoiceSession | null>(null)
  const enabledRef = useRef(enabled)
  const wasEnabledRef = useRef(false)
  const startingRef = useRef(false)
  const startEpochRef = useRef(0)
  const openedRef = useRef(false)
  const busyRef = useRef(busy)
  const speakingRef = useRef(false)
  const delegationRef = useRef<null | string>(null)
  const submittedAtRef = useRef(0)
  const turnObservedRef = useRef(false)
  const deliveredRef = useRef(false)
  const userUtteranceRef = useRef('')
  const utteranceTimerRef = useRef<null | number>(null)

  const latest = useRef({
    beforeMicOpen,
    consumePendingResponse,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse
  })

  enabledRef.current = enabled
  busyRef.current = busy
  delegationRef.current = delegation
  latest.current = {
    beforeMicOpen,
    consumePendingResponse,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse
  }

  const end = useCallback(async () => {
    startEpochRef.current += 1
    startingRef.current = false
    openedRef.current = false

    if (utteranceTimerRef.current) {
      window.clearTimeout(utteranceTimerRef.current)
      utteranceTimerRef.current = null
    }

    userUtteranceRef.current = ''
    const session = sessionRef.current
    sessionRef.current = null
    setDelegation(null)
    session?.close()
    setMuted(false)
    setLevel(0)
    setStatus('idle')
  }, [])

  const start = useCallback(async () => {
    if (sessionRef.current || startingRef.current) {
      return
    }

    startingRef.current = true
    const epoch = ++startEpochRef.current
    openedRef.current = false

    try {
      await latest.current.beforeMicOpen?.()
    } catch {
      // A wake-pause failure must not block an explicit start.
    }

    if (!enabledRef.current || startEpochRef.current !== epoch) {
      startingRef.current = false

      return
    }

    const session = new GrokVoiceSession(
      {
        onClosed: reason => {
          if (sessionRef.current !== session) {
            return
          }

          sessionRef.current = null
          setDelegation(null)
          setStatus('idle')

          if (!openedRef.current) {
            return
          }

          if (reason === 'end_call') {
            latest.current.onStopWord?.()

            return
          }

          if (reason === 'close_requested') {
            return
          }

          notify({
            kind: 'warning',
            message: reason === 'connection_lost' ? voiceCopy.liveEndedConnectionLost : reason,
            title: voiceCopy.liveEnded
          })
          latest.current.onFatalError?.()
        },
        onDelegate: (request, voiceContext) => {
          if (sessionRef.current !== session) {
            return
          }

          if (request && isVoiceStopCommand(request)) {
            void end()
            latest.current.onStopWord?.()

            return
          }

          if (busyRef.current) {
            void latest.current.onInterrupt?.()
          }

          const id = `${Date.now()}`
          setDelegation(id)
          deliveredRef.current = false
          turnObservedRef.current = false
          submittedAtRef.current = Date.now()
          setStatus('thinking')
          void Promise.resolve(latest.current.onSubmit(request, voiceContext)).catch(error => {
            notifyError(error, voiceCopy.liveDelegationFailed)
            session.deliverResult('Sorry, I could not reach Hermes for that request.')
            setDelegation(null)
            setStatus(speakingRef.current ? 'speaking' : 'listening')
          })
        },
        onError: message => {
          notify({ kind: 'warning', message, title: voiceCopy.liveError })
        },
        onLevel: setLevel,
        onSpeakingChange: speaking => {
          speakingRef.current = speaking
          setStatus(current => {
            if (current === 'idle') {
              return current
            }

            if (speaking) {
              return 'speaking'
            }

            return delegationRef.current ? 'thinking' : 'listening'
          })
        },
        onUserTranscript: text => {
          userUtteranceRef.current += ` ${text}`

          if (utteranceTimerRef.current) {
            window.clearTimeout(utteranceTimerRef.current)
          }

          utteranceTimerRef.current = window.setTimeout(() => {
            utteranceTimerRef.current = null
            const utterance = userUtteranceRef.current
            userUtteranceRef.current = ''

            if (sessionRef.current === session && isVoiceStopCommand(utterance)) {
              void end()
              latest.current.onStopWord?.()
            }
          }, UTTERANCE_SETTLE_MS)
        }
      },
      ownerRef.current
    )

    sessionRef.current = session
    startingRef.current = false
    setMuted(false)
    setStatus('thinking')

    try {
      await session.start()

      if (sessionRef.current !== session || startEpochRef.current !== epoch) {
        session.close()

        return
      }

      openedRef.current = true
      setStatus(speakingRef.current ? 'speaking' : 'listening')
    } catch (error) {
      if (sessionRef.current === session) {
        sessionRef.current = null
      }

      session.close()

      if (startEpochRef.current !== epoch) {
        return
      }

      notifyError(error instanceof DOMException ? micError(error, voiceCopy) : error, voiceCopy.couldNotStartSession)
      setStatus('idle')
      latest.current.onFatalError?.()
    }
  }, [end, voiceCopy])

  // eslint-disable-next-line no-restricted-syntax -- turn-coordination refs (spoken cursor), not atom mirrors
  useEffect(() => {
    const session = sessionRef.current
    const token = delegationRef.current

    if (!session || !token) {
      return undefined
    }

    const tick = () => {
      if (sessionRef.current !== session || delegationRef.current !== token || deliveredRef.current) {
        return
      }

      if (busyRef.current) {
        turnObservedRef.current = true

        return
      }

      const response = latest.current.pendingResponse()

      if (response?.pending) {
        turnObservedRef.current = true

        return
      }

      if (response) {
        deliveredRef.current = true
        session.deliverResult(sanitizeTextForSpeech(response.text))
        latest.current.consumePendingResponse()
        setDelegation(null)
        setStatus(speakingRef.current ? 'speaking' : 'listening')

        return
      }

      if (turnObservedRef.current || Date.now() - submittedAtRef.current > SUBMIT_SETTLE_GRACE_MS) {
        deliveredRef.current = true
        session.deliverResult('')
        setDelegation(null)
        setStatus(speakingRef.current ? 'speaking' : 'listening')
      }
    }

    const timer = window.setInterval(tick, 200)
    tick()

    return () => window.clearInterval(timer)
  }, [busy, delegation])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value
      sessionRef.current?.setMuted(next)

      return next
    })
  }, [])

  const stopTurn = useCallback(() => {
    sessionRef.current?.nudge()
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
