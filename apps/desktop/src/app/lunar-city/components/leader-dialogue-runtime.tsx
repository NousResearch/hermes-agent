import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { useVoiceConversation } from '@/app/chat/composer/hooks/use-voice-conversation'
import { blobToDataUrl } from '@/app/session/hooks/use-prompt-actions/utils'
import type { ClientSessionState } from '@/app/types'
import { transcribeAudio } from '@/hermes'
import { chatMessageText } from '@/lib/chat-messages'
import { transcribeAudioClientDirect, type VoiceClientScope } from '@/lib/voice-client-direct'
import { requestForSessionProfile } from '@/store/session-request-router'
import { $sessionStates } from '@/store/session-states'

import { leaderAnimationForConversation } from '../leader-runtime'
import type { LeaderOwner, LeaderSession } from '../leader-sessions'
import type { LeaderAnimationState, LeaderStateClipMap } from '../model'

import { LeaderDialogue, type LeaderVoiceBridge } from './leader-dialogue'

interface LeaderDialogueRuntimeProps {
  clips?: LeaderStateClipMap
  leaderLabel: string
  onClose(): void
  onLeaderStateChange(state: LeaderAnimationState): void
  onOpenFullChat(storedId: string, owner: LeaderOwner): Promise<void> | void
  onPerfScenarioReady?(
    run: ((text: string) => Promise<{ opened: number; received: number; sent: number }>) | undefined
  ): void
  owner: LeaderOwner
  session: LeaderSession
  voiceAvailable: boolean
}

const EMPTY_SESSION_STATE = {
  awaitingResponse: false,
  busy: false,
  interrupted: false,
  messages: []
} satisfies Pick<ClientSessionState, 'awaitingResponse' | 'busy' | 'interrupted' | 'messages'>

function selectSessionState(runtimeId: string) {
  const state = $sessionStates.get()[runtimeId]

  return state
    ? {
        awaitingResponse: state.awaitingResponse,
        busy: state.busy,
        interrupted: state.interrupted,
        messages: state.messages
      }
    : EMPTY_SESSION_STATE
}

type LeaderSessionState = ReturnType<typeof selectSessionState>

function sameSessionState(left: LeaderSessionState, right: LeaderSessionState): boolean {
  return (
    left.awaitingResponse === right.awaitingResponse &&
    left.busy === right.busy &&
    left.interrupted === right.interrupted &&
    left.messages === right.messages
  )
}

/** Subscribe only to this leader runtime, not every Desktop session update. */
function useLeaderSessionState(runtimeId: string): LeaderSessionState {
  const [state, setState] = useState<LeaderSessionState>(() => selectSessionState(runtimeId))

  useEffect(() => {
    setState(selectSessionState(runtimeId))

    return $sessionStates.listen(() => {
      const next = selectSessionState(runtimeId)

      setState(current => (sameSessionState(current, next) ? current : next))
    })
  }, [runtimeId])

  return state
}

function rejectAmbientRequest<T>(): Promise<T> {
  return Promise.reject(new Error('Leader dialogue requires an exact owner route; ambient routing is forbidden'))
}

function assistantVoiceResponse(messages: LeaderSessionState['messages'], consumed: ReadonlySet<string>) {
  for (const message of [...messages].reverse()) {
    const text = chatMessageText(message).trim()

    if (message.role === 'assistant' && !message.hidden && text && !consumed.has(message.id)) {
      return { id: message.id, pending: false, text }
    }
  }

  return null
}

function staleVoiceRouteError(): Error {
  return new Error('Leader voice route changed during transcription')
}

interface LeaderVoiceRouteToken {
  readonly generation: symbol
  readonly owner: VoiceClientScope
  readonly session: Readonly<Pick<LeaderSession, 'runtimeId' | 'storedId'>>
}

interface LeaderVoiceCaptureToken {
  readonly controller: AbortController
  readonly route: LeaderVoiceRouteToken
}

/**
 * The narrow bridge from a profile-owned leader to the normal Desktop session
 * transport. It owns no transcript, recorder, audio socket, or ambient route:
 * those remain the standard session state and useVoiceConversation hook.
 */
export function LeaderDialogueRuntime({
  clips,
  leaderLabel,
  onClose,
  onLeaderStateChange,
  onOpenFullChat,
  onPerfScenarioReady,
  owner,
  session,
  voiceAvailable
}: LeaderDialogueRuntimeProps) {
  const sessionState = useLeaderSessionState(session.runtimeId)
  const sessionStateRef = useRef(sessionState)
  sessionStateRef.current = sessionState
  const [voiceRequested, setVoiceRequested] = useState(false)
  const [voiceError, setVoiceError] = useState<string | undefined>(undefined)
  const consumedVoiceResponsesRef = useRef<Set<string>>(new Set())
  const pendingVoiceResponseRef = useRef<string | null>(null)
  const activeVoiceRouteRef = useRef<LeaderVoiceRouteToken | undefined>(undefined)
  const activeVoiceCaptureRef = useRef<LeaderVoiceCaptureToken | undefined>(undefined)

  const voiceRoute = useMemo<LeaderVoiceRouteToken>(
    () =>
      Object.freeze({
        generation: Symbol('leader-voice-route'),
        owner: Object.freeze({ connectionId: owner.connectionId.trim(), profile: owner.profile.trim() }),
        session: Object.freeze({ runtimeId: session.runtimeId, storedId: session.storedId })
      }),
    [owner.connectionId, owner.profile, session.runtimeId, session.storedId]
  )

  // A changed route must revoke the prior capture in the synchronous commit
  // phase. Passive effects leave a window where an already-resolved direct
  // request could start the relay against a newly selected leader.
  useLayoutEffect(() => {
    activeVoiceRouteRef.current = voiceRoute

    return () => {
      const capture = activeVoiceCaptureRef.current

      if (capture?.route === voiceRoute) {
        capture.controller.abort()
        activeVoiceCaptureRef.current = undefined
      }

      if (activeVoiceRouteRef.current === voiceRoute) {
        activeVoiceRouteRef.current = undefined
      }
    }
  }, [voiceRoute])

  useEffect(() => {
    if (!voiceAvailable) {
      setVoiceRequested(false)
    }
  }, [voiceAvailable])

  const submit = useCallback(
    async (text: string) => {
      await requestForSessionProfile(owner, rejectAmbientRequest, 'prompt.submit', {
        session_id: session.runtimeId,
        text
      })
    },
    [owner, session.runtimeId]
  )

  const runPerfScenario = useCallback(
    async (text: string) => {
      const baselineAssistantIds = new Set(
        sessionStateRef.current.messages.filter(message => message.role === 'assistant').map(message => message.id)
      )

      await submit(text)
      const deadline = Date.now() + 4_000

      while (Date.now() < deadline) {
        const response = sessionStateRef.current.messages.find(
          message =>
            message.role === 'assistant' &&
            !message.hidden &&
            !baselineAssistantIds.has(message.id) &&
            chatMessageText(message).trim().length > 0
        )

        if (response) {
          return { opened: 1, received: 1, sent: 1 }
        }

        await new Promise(resolve => setTimeout(resolve, 10))
      }

      throw new Error('Timed out waiting for authoritative leader assistant response')
    },
    [submit]
  )

  useEffect(() => {
    onPerfScenarioReady?.(runPerfScenario)

    return () => onPerfScenarioReady?.(undefined)
  }, [onPerfScenarioReady, runPerfScenario])

  const interrupt = useCallback(async () => {
    await requestForSessionProfile(owner, rejectAmbientRequest, 'session.interrupt', { session_id: session.runtimeId })
  }, [owner, session.runtimeId])

  const transcribe = useCallback(
    async (audio: Blob) => {
      if (!voiceAvailable || activeVoiceRouteRef.current !== voiceRoute) {
        throw new Error('Voice is unavailable because this leader is no longer the active exact desktop route')
      }

      activeVoiceCaptureRef.current?.controller.abort()
      const controller = new AbortController()
      const capture: LeaderVoiceCaptureToken = Object.freeze({ controller, route: voiceRoute })
      activeVoiceCaptureRef.current = capture

      const assertCurrentVoiceRoute = (): void => {
        if (
          activeVoiceRouteRef.current !== capture.route ||
          activeVoiceCaptureRef.current !== capture ||
          capture.controller.signal.aborted
        ) {
          throw staleVoiceRouteError()
        }
      }

      // Same client-direct then gateway-relay ladder as the standard composer;
      // this component does not create a recorder, WebSocket, or private audio path.
      const direct = await transcribeAudioClientDirect(audio, capture.route.owner, capture.controller.signal)
      assertCurrentVoiceRoute()

      if (direct !== null) {
        return direct
      }

      const dataUrl = await blobToDataUrl(audio)
      assertCurrentVoiceRoute()
      const result = await transcribeAudio(dataUrl, audio.type, capture.route.owner)
      assertCurrentVoiceRoute()

      return result.transcript
    },
    [voiceAvailable, voiceRoute]
  )

  const pendingResponse = useCallback(() => {
    const response = assistantVoiceResponse(sessionState.messages, consumedVoiceResponsesRef.current)
    pendingVoiceResponseRef.current = response?.id ?? null

    return response
  }, [sessionState.messages])

  const consumePendingResponse = useCallback(() => {
    const id = pendingVoiceResponseRef.current

    if (id) {
      consumedVoiceResponsesRef.current.add(id)
    }

    pendingVoiceResponseRef.current = null
  }, [])

  const startVoice = useCallback(() => {
    // Do not read old assistant messages aloud when an operator starts voice.
    // This is voice-local playback cursor state, not a transcript mirror.
    consumedVoiceResponsesRef.current = new Set(
      sessionState.messages.filter(message => message.role === 'assistant').map(message => message.id)
    )
    pendingVoiceResponseRef.current = null
    setVoiceError(undefined)
    setVoiceRequested(true)
  }, [sessionState.messages])

  const voiceConversation = useVoiceConversation({
    busy: sessionState.busy || sessionState.awaitingResponse,
    consumePendingResponse,
    enabled: voiceRequested && voiceAvailable,
    onFatalError: () => {
      setVoiceError('Voice capture ended. You can continue with text.')
      setVoiceRequested(false)
    },
    onInterrupt: interrupt,
    onStopWord: () => setVoiceRequested(false),
    onSubmit: submit,
    onTranscribeAudio: transcribe,
    pendingResponse
  })

  const voice: LeaderVoiceBridge = {
    active: voiceRequested,
    available: voiceAvailable,
    end: async () => {
      setVoiceRequested(false)
      await voiceConversation.end()
    },
    error: voiceError,
    start: startVoice,
    status: voiceConversation.status,
    stopTurn: voiceConversation.stopTurn
  }

  const animationState = clips
    ? leaderAnimationForConversation({
        clips,
        session: sessionState,
        sessionAvailable: true,
        voice: { active: voice.active, status: voice.status }
      }).state
    : undefined

  useEffect(() => {
    if (animationState) {
      onLeaderStateChange(animationState)
    }
  }, [animationState, onLeaderStateChange])

  useEffect(
    () => () => {
      onLeaderStateChange('idle')
    },
    [onLeaderStateChange]
  )

  return (
    <LeaderDialogue
      leaderLabel={leaderLabel}
      onClose={onClose}
      onInterrupt={() => interrupt()}
      onOpenFullChat={onOpenFullChat}
      onSubmit={text => submit(text)}
      owner={owner}
      session={session}
      sessionAvailable
      sessionState={sessionState}
      voice={voice}
    />
  )
}

export type { LeaderDialogueRuntimeProps }
