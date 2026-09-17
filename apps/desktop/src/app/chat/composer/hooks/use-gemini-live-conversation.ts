import { useCallback, useEffect, useRef, useState } from 'react'

import { useI18n } from '@/i18n'
import { assistantTextPart, textPart } from '@/lib/chat-messages/parts'
import {
  GeminiLiveSession,
  getStoredGeminiLiveModel,
  getStoredGeminiLiveVoice,
  resolveGeminiLiveApiKey
} from '@/lib/gemini-live'
import { sanitizeTextForSpeech } from '@/lib/speech-text'
import { isVoiceStopCommand } from '@/lib/voice-stop-word'
import { notify, notifyError } from '@/store/notifications'
import { setMessages } from '@/store/session'

import type { ConversationStatus } from './use-voice-conversation'

const SUBMIT_SETTLE_GRACE_MS = 15_000
const UTTERANCE_SETTLE_MS = 1_500

export interface PendingVoiceResponse {
  id: string
  pending: boolean
  text: string
}

export interface GeminiLiveConversationOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  onInterrupt?: () => Promise<void> | void
  onStopWord?: () => void
  onSubmit: (text: string) => Promise<void> | void
  pendingResponse: () => PendingVoiceResponse | null
  consumePendingResponse: () => void
  seedHistory?: () => any[]
  activeToolLabel?: () => null | string
  beforeMicOpen?: () => Promise<void> | void
}

/**
 * Gemini-Live conversation engine — client-side full duplex WebSockets directly
 * from Hermes Desktop to Google GenAI Live (gemini-3.8-live).
 * Seamlessly delegates real tasks to Hermes Agent using onSubmit.
 */
export function useGeminiLiveConversation({
  busy,
  enabled,
  onFatalError,
  onInterrupt,
  onStopWord,
  onSubmit,
  pendingResponse,
  consumePendingResponse,
  seedHistory,
  activeToolLabel,
  beforeMicOpen
}: GeminiLiveConversationOptions) {
  const { t } = useI18n()
  const voiceCopy = t.notifications.voice

  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [muted, setMuted] = useState(false)
  const [level, setLevel] = useState(0)
  const [activeDelegation, setActiveDelegation] = useState<null | string>(null)

  const sessionRef = useRef<null | GeminiLiveSession>(null)
  const startEpochRef = useRef(0)
  const startingRef = useRef(false)
  const turnObservedRef = useRef(false)
  const submittedAtRef = useRef(0)
  const enabledRef = useRef(enabled)
  const busyRef = useRef(busy)
  const speakingRef = useRef(false)
  const userUtteranceRef = useRef('')
  const utteranceTimerRef = useRef<null | number>(null)
  const delegationRef = useRef<null | string>(null)
  const spokenLengthRef = useRef(0)
  const spokenResponseIdRef = useRef<null | string>(null)
  const lastToolLabelRef = useRef<null | string>(null)

  const currentUserMsgIdRef = useRef<string | null>(null)
  const userCommittedTextRef = useRef('')
  const currentAssistantMsgIdRef = useRef<string | null>(null)
  const assistantTextRef = useRef('')

  const latest = useRef({
    activeToolLabel,
    beforeMicOpen,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse,
    seedHistory
  })

  latest.current = {
    activeToolLabel,
    beforeMicOpen,
    onFatalError,
    onInterrupt,
    onStopWord,
    onSubmit,
    pendingResponse,
    consumePendingResponse,
    seedHistory
  }

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const setDelegation = useCallback((id: null | string) => {
    delegationRef.current = id
    setActiveDelegation(id)
  }, [])

  const refreshStatus = useCallback(() => {
    if (!sessionRef.current) {
      setStatus('idle')
      return
    }

    if (speakingRef.current) {
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

    if (utteranceTimerRef.current) {
      window.clearTimeout(utteranceTimerRef.current)
      utteranceTimerRef.current = null
    }

    // Finalize any in-flight live message bubbles
    if (currentUserMsgIdRef.current) {
      const uId = currentUserMsgIdRef.current
      setMessages(messages =>
        messages.map(m => (m.id === uId ? { ...m, pending: false } : m))
      )
      currentUserMsgIdRef.current = null
      userCommittedTextRef.current = ''
    }

    if (currentAssistantMsgIdRef.current) {
      const aId = currentAssistantMsgIdRef.current
      setMessages(messages =>
        messages.map(m => (m.id === aId ? { ...m, pending: false } : m))
      )
      currentAssistantMsgIdRef.current = null
      assistantTextRef.current = ''
    }

    userUtteranceRef.current = ''
    const session = sessionRef.current
    sessionRef.current = null
    setDelegation(null)
    spokenResponseIdRef.current = null
    spokenLengthRef.current = 0
    speakingRef.current = false

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
      // Wake-pause failure should not block start
    }

    if (!enabledRef.current || startEpochRef.current !== epoch) {
      startingRef.current = false
      return
    }

    const apiKey = await resolveGeminiLiveApiKey()
    if (!apiKey) {
      startingRef.current = false
      notify({
        kind: 'error',
        message: 'Google Gemini API key required. Set GEMINI_API_KEY in settings or the voice menu.',
        title: 'Gemini Live'
      })
      setStatus('idle')
      latest.current.onFatalError?.()
      return
    }

    const session = new GeminiLiveSession(
      {
        onTranscript: fragment => {
          if (fragment.speaker === 'user') {
            // User began speaking: finalize any previous assistant message bubble
            if (currentAssistantMsgIdRef.current) {
              const aId = currentAssistantMsgIdRef.current
              setMessages(messages =>
                messages.map(m => (m.id === aId ? { ...m, pending: false } : m))
              )
              currentAssistantMsgIdRef.current = null
              assistantTextRef.current = ''
            }

            if (!currentUserMsgIdRef.current) {
              currentUserMsgIdRef.current = `user_live_${Date.now()}`
              userCommittedTextRef.current = ''
            }

            const targetId = currentUserMsgIdRef.current
            const isInterim = fragment.interim ?? false

            if (!isInterim) {
              userCommittedTextRef.current = userCommittedTextRef.current
                ? `${userCommittedTextRef.current} ${fragment.text.trim()}`
                : fragment.text.trim()
            }

            const displayText = (isInterim
              ? `${userCommittedTextRef.current} ${fragment.text.trim()}`.trim()
              : userCommittedTextRef.current
            ).trim()

            if (displayText) {
              setMessages(messages => {
                const existing = messages.find(m => m.id === targetId)
                if (existing) {
                  return messages.map(m =>
                    m.id === targetId ? { ...m, parts: [textPart(displayText)], pending: isInterim } : m
                  )
                }
                return [
                  ...messages,
                  {
                    id: targetId,
                    role: 'user',
                    parts: [textPart(displayText)],
                    timestamp: Date.now() / 1000,
                    pending: isInterim
                  }
                ]
              })
            }

            userUtteranceRef.current = displayText

            if (utteranceTimerRef.current) {
              window.clearTimeout(utteranceTimerRef.current)
            }

            utteranceTimerRef.current = window.setTimeout(() => {
              utteranceTimerRef.current = null
              const utterance = userUtteranceRef.current
              if (sessionRef.current === session && isVoiceStopCommand(utterance)) {
                void end()
                latest.current.onStopWord?.()
              }
            }, UTTERANCE_SETTLE_MS)
          } else if (fragment.speaker === 'assistant') {
            // Assistant started responding: finalize user message bubble
            if (currentUserMsgIdRef.current) {
              const uId = currentUserMsgIdRef.current
              setMessages(messages =>
                messages.map(m => (m.id === uId ? { ...m, pending: false } : m))
              )
              currentUserMsgIdRef.current = null
              userCommittedTextRef.current = ''
            }

            if (!currentAssistantMsgIdRef.current) {
              currentAssistantMsgIdRef.current = `ast_live_${Date.now()}`
              assistantTextRef.current = ''
            }

            const targetId = currentAssistantMsgIdRef.current
            assistantTextRef.current = assistantTextRef.current
              ? `${assistantTextRef.current} ${fragment.text.trim()}`
              : fragment.text.trim()

            const displayText = assistantTextRef.current.trim()
            if (displayText) {
              setMessages(messages => {
                const existing = messages.find(m => m.id === targetId)
                if (existing) {
                  return messages.map(m =>
                    m.id === targetId ? { ...m, parts: [assistantTextPart(displayText)], pending: true } : m
                  )
                }
                return [
                  ...messages,
                  {
                    id: targetId,
                    role: 'assistant',
                    parts: [assistantTextPart(displayText)],
                    timestamp: Date.now() / 1000,
                    pending: true
                  }
                ]
              })
            }
          }
        },
        onTurnComplete: () => {
          if (currentAssistantMsgIdRef.current) {
            const aId = currentAssistantMsgIdRef.current
            setMessages(messages =>
              messages.map(m => (m.id === aId ? { ...m, pending: false } : m))
            )
            currentAssistantMsgIdRef.current = null
            assistantTextRef.current = ''
          }
          if (currentUserMsgIdRef.current) {
            const uId = currentUserMsgIdRef.current
            setMessages(messages =>
              messages.map(m => (m.id === uId ? { ...m, pending: false } : m))
            )
            currentUserMsgIdRef.current = null
            userCommittedTextRef.current = ''
          }
        },
        onInterrupted: () => {
          if (currentAssistantMsgIdRef.current) {
            const aId = currentAssistantMsgIdRef.current
            setMessages(messages =>
              messages.map(m => (m.id === aId ? { ...m, pending: false } : m))
            )
            currentAssistantMsgIdRef.current = null
            assistantTextRef.current = ''
          }
        },
        onClosed: reason => {
          if (sessionRef.current !== session) {
            return
          }

          sessionRef.current = null
          setDelegation(null)
          setStatus('idle')

          if (reason !== 'close_requested') {
            notify({
              kind: 'warning',
              message: reason,
              title: 'Gemini Live Ended'
            })
            latest.current.onFatalError?.()
          }
        },
        onDelegation: (delegationId, request) => {
          if (sessionRef.current !== session) {
            return
          }

          const prompt = request.trim()
          if (!prompt) {
            return
          }

          if (isVoiceStopCommand(prompt)) {
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
          refreshStatus()

          // Standard chat submit for VPS gateway compatibility
          void Promise.resolve(latest.current.onSubmit(prompt)).catch(error => {
            notifyError(error, voiceCopy.liveDelegationFailed)
            session.speak(delegationId, 'Sorry, I could not reach Hermes for that request.')
            setDelegation(null)
            refreshStatus()
          })
        },
        onError: (message, fatal) => {
          notify({ kind: fatal ? 'error' : 'warning', message, title: 'Gemini Live' })
        },
        onSpeakingChange: speaking => {
          speakingRef.current = speaking
          setLevel(speaking ? 0.6 : 0)
          refreshStatus()
        },
        onLevel: lvl => {
          if (!speakingRef.current) {
            setLevel(lvl)
          }
        }
      },
      {
        apiKey,
        model: getStoredGeminiLiveModel(),
        voice: getStoredGeminiLiveVoice()
      }
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

      refreshStatus()
    } catch (error) {
      if (sessionRef.current === session) {
        sessionRef.current = null
      }

      session.close()

      if (startEpochRef.current !== epoch) {
        return
      }

      notifyError(error, 'Could not start Gemini Live session')
      setStatus('idle')
      latest.current.onFatalError?.()
    }
  }, [end, refreshStatus, setDelegation, voiceCopy.liveDelegationFailed])

  // Stream replies and tool activity back into Gemini Live
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
      if (tool && tool !== lastToolLabelRef.current) {
        lastToolLabelRef.current = tool
        session.think(delegationId, `Hermes is working: ${tool}`)
      }

      const response = latest.current.pendingResponse()
      if (response) {
        turnObservedRef.current = true

        if (spokenResponseIdRef.current !== response.id) {
          spokenResponseIdRef.current = response.id
          spokenLengthRef.current = 0
        }

        const spoken = sanitizeTextForSpeech(response.text)

        if (response.pending || busyRef.current) {
          const boundary = spoken.lastIndexOf('. ', spoken.length - 2)
          const cut = boundary > spokenLengthRef.current ? boundary + 1 : spokenLengthRef.current

          if (cut > spokenLengthRef.current) {
            session.speak(delegationId, spoken.slice(spokenLengthRef.current, cut))
            spokenLengthRef.current = cut
          }
          return
        }

        if (spoken.length > spokenLengthRef.current) {
          session.speak(delegationId, spoken.slice(spokenLengthRef.current))
          spokenLengthRef.current = spoken.length
        }

        latest.current.consumePendingResponse()
        setDelegation(null)
        refreshStatus()
        return
      }

      if (
        !busyRef.current &&
        (turnObservedRef.current || Date.now() - submittedAtRef.current > SUBMIT_SETTLE_GRACE_MS)
      ) {
        if (spokenLengthRef.current === 0) {
          session.think(delegationId, 'Hermes finished the task.')
        }
        setDelegation(null)
        refreshStatus()
      }
    }

    const timer = window.setInterval(tick, 200)
    tick()

    return () => {
      window.clearInterval(timer)
    }
  }, [activeDelegation, refreshStatus, setDelegation])

  useEffect(() => {
    if (enabled && !sessionRef.current && !startingRef.current) {
      void start()
    } else if (!enabled && sessionRef.current) {
      void end()
    }
  }, [enabled, start, end])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value
      sessionRef.current?.setMuted(next)

      return next
    })
  }, [])

  const stopTurn = useCallback(() => {
    sessionRef.current?.stopAudioStream()
  }, [])

  return {
    end,
    interrupt: () => {
      sessionRef.current?.interrupt()
      if (currentAssistantMsgIdRef.current) {
        const aId = currentAssistantMsgIdRef.current
        setMessages(messages =>
          messages.map(m => (m.id === aId ? { ...m, pending: false } : m))
        )
        currentAssistantMsgIdRef.current = null
        assistantTextRef.current = ''
      }
    },
    level,
    muted,
    setMuted,
    start,
    status,
    stopTurn,
    toggleMute
  }
}

