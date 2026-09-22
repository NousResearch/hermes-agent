// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { useLayoutEffect } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import type * as HermesModule from '@/hermes'
import { $sessionStates } from '@/store/session-states'

import type { LeaderStateClipMap } from '../model'

import { LeaderDialogueRuntime } from './leader-dialogue-runtime'

const { requestForSessionProfile, transcribeAudio, transcribeAudioClientDirect, voiceConversation, voiceHook } =
  vi.hoisted(() => ({
    requestForSessionProfile: vi.fn(async () => ({ status: 'queued' })),
    transcribeAudio: vi.fn(),
    transcribeAudioClientDirect: vi.fn(),
    voiceConversation: {
      end: vi.fn(),
      start: vi.fn(),
      status: 'idle' as const,
      stopTurn: vi.fn()
    },
    voiceHook: {
      onTranscribeAudio: undefined as undefined | ((audio: Blob) => Promise<string>)
    }
  }))

vi.mock('@/store/session-request-router', () => ({ requestForSessionProfile }))
vi.mock('@/app/chat/composer/hooks/use-voice-conversation', () => ({
  useVoiceConversation: (options: { onTranscribeAudio?: (audio: Blob) => Promise<string> }) => {
    voiceHook.onTranscribeAudio = options.onTranscribeAudio

    return voiceConversation
  }
}))
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  transcribeAudio
}))
vi.mock('@/lib/voice-client-direct', () => ({ transcribeAudioClientDirect }))

const clips: LeaderStateClipMap = {
  acknowledging: 'leader:owl:acknowledging',
  idle: 'leader:owl:idle',
  listening: 'leader:owl:listening',
  talking: 'leader:owl:talking',
  thinking: 'leader:owl:thinking',
  unavailable: 'leader:owl:unavailable'
}

function sessionState(overrides: Partial<ClientSessionState> = {}): ClientSessionState {
  return {
    awaitingResponse: false,
    busy: false,
    cwd: '',
    messages: [],
    ...overrides
  } as ClientSessionState
}

function deferred<T>() {
  let reject!: (error: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((done, fail) => {
    reject = fail
    resolve = done
  })

  return { promise, reject, resolve }
}

interface LeaderDialogueRuntimeHarnessProps {
  onRouteLayout?(): void
  owner: { connectionId: string; profile: string }
  session: { runtimeId: string; storedId: string }
}

/**
 * Resolves a prior capture from the parent layout phase. Child passive effects
 * have not run yet, so this models an owner change racing an audio completion.
 */
function LeaderDialogueRuntimeHarness({ onRouteLayout, owner, session }: LeaderDialogueRuntimeHarnessProps) {
  useLayoutEffect(() => {
    onRouteLayout?.()
  }, [onRouteLayout, owner.connectionId, owner.profile, session.runtimeId, session.storedId])

  return (
    <LeaderDialogueRuntime
      clips={clips}
      leaderLabel={`${owner.profile} leader`}
      onClose={vi.fn()}
      onLeaderStateChange={vi.fn()}
      onOpenFullChat={vi.fn()}
      owner={owner}
      session={session}
      voiceAvailable
    />
  )
}

afterEach(() => {
  cleanup()
  $sessionStates.set({})
  voiceHook.onTranscribeAudio = undefined
  vi.clearAllMocks()
})

describe('LeaderDialogueRuntime', () => {
  it('reports perf dialogue proof only after real submit acceptance and a new authoritative assistant message', async () => {
    let runScenario: ((text: string) => Promise<{ opened: number; received: number; sent: number }>) | undefined
    $sessionStates.set({ 'runtime-owl': sessionState() })

    render(
      <LeaderDialogueRuntime
        clips={clips}
        leaderLabel="owl leader"
        onClose={vi.fn()}
        onLeaderStateChange={vi.fn()}
        onOpenFullChat={vi.fn()}
        onPerfScenarioReady={run => {
          runScenario = run
        }}
        owner={{ connectionId: 'source-a', profile: 'owl' }}
        session={{ runtimeId: 'runtime-owl', storedId: 'stored-owl' }}
        voiceAvailable={false}
      />
    )

    await waitFor(() => expect(runScenario).toBeTypeOf('function'))
    const pending = runScenario!('Acceptance fake voice turn')

    await waitFor(() =>
      expect(requestForSessionProfile).toHaveBeenCalledWith(
        { connectionId: 'source-a', profile: 'owl' },
        expect.any(Function),
        'prompt.submit',
        { session_id: 'runtime-owl', text: 'Acceptance fake voice turn' }
      )
    )
    $sessionStates.set({
      'runtime-owl': sessionState({
        messages: [
          {
            hidden: false,
            id: 'assistant-new',
            parts: [{ text: 'Authoritative fixture reply', type: 'text' }],
            role: 'assistant'
          }
        ] as never
      })
    })

    await expect(pending).resolves.toEqual({ opened: 1, received: 1, sent: 1 })
  })

  it('submits and interrupts only through the exact profile owner while projecting observed session state', async () => {
    const onLeaderStateChange = vi.fn()
    $sessionStates.set({ 'runtime-owl': sessionState() })

    render(
      <LeaderDialogueRuntime
        clips={clips}
        leaderLabel="owl leader"
        onClose={vi.fn()}
        onLeaderStateChange={onLeaderStateChange}
        onOpenFullChat={vi.fn()}
        owner={{ connectionId: 'source-a', profile: 'owl' }}
        session={{ runtimeId: 'runtime-owl', storedId: 'stored-owl' }}
        voiceAvailable={false}
      />
    )

    fireEvent.change(screen.getByRole('textbox', { name: 'Message owl leader' }), { target: { value: 'Keep scope' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() =>
      expect(requestForSessionProfile).toHaveBeenCalledWith(
        { connectionId: 'source-a', profile: 'owl' },
        expect.any(Function),
        'prompt.submit',
        { session_id: 'runtime-owl', text: 'Keep scope' }
      )
    )

    $sessionStates.set({ 'runtime-owl': sessionState({ awaitingResponse: true, busy: true }) })

    await waitFor(() => expect(onLeaderStateChange).toHaveBeenLastCalledWith('thinking'))
    fireEvent.click(screen.getByRole('button', { name: 'Interrupt response' }))

    await waitFor(() =>
      expect(requestForSessionProfile).toHaveBeenCalledWith(
        { connectionId: 'source-a', profile: 'owl' },
        expect.any(Function),
        'session.interrupt',
        { session_id: 'runtime-owl' }
      )
    )
  })

  it('pins a leader capture to its immutable owner and ignores a stale completion after an active connection switch', async () => {
    const direct = deferred<null | string>()
    transcribeAudioClientDirect.mockReturnValue(direct.promise)
    $sessionStates.set({ shared: sessionState() })

    const view = render(
      <LeaderDialogueRuntime
        clips={clips}
        leaderLabel="owl leader"
        onClose={vi.fn()}
        onLeaderStateChange={vi.fn()}
        onOpenFullChat={vi.fn()}
        owner={{ connectionId: 'source-a', profile: 'owl' }}
        session={{ runtimeId: 'shared', storedId: 'stored-owl' }}
        voiceAvailable
      />
    )

    const staleTranscribe = voiceHook.onTranscribeAudio!
    const pending = staleTranscribe(new Blob(['audio'], { type: 'audio/webm' }))

    expect(transcribeAudioClientDirect).toHaveBeenCalledWith(
      expect.any(Blob),
      { connectionId: 'source-a', profile: 'owl' },
      expect.any(AbortSignal)
    )

    // The same runtime id now belongs to a different selected exact route.
    // The old capture must never continue against the new active source.
    view.rerender(
      <LeaderDialogueRuntime
        clips={clips}
        leaderLabel="fox leader"
        onClose={vi.fn()}
        onLeaderStateChange={vi.fn()}
        onOpenFullChat={vi.fn()}
        owner={{ connectionId: 'source-b', profile: 'fox' }}
        session={{ runtimeId: 'shared', storedId: 'stored-fox' }}
        voiceAvailable
      />
    )
    direct.resolve(null)

    await expect(pending).rejects.toThrow(/leader voice route changed/i)
    expect(transcribeAudio).not.toHaveBeenCalled()

    fireEvent.change(screen.getByRole('textbox', { name: 'Message fox leader' }), {
      target: { value: 'Text remains usable' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }))

    await waitFor(() =>
      expect(requestForSessionProfile).toHaveBeenCalledWith(
        { connectionId: 'source-b', profile: 'fox' },
        expect.any(Function),
        'prompt.submit',
        { session_id: 'shared', text: 'Text remains usable' }
      )
    )
  })

  it('ignores a relay transcript that completes after its exact leader route has been replaced', async () => {
    const relay = deferred<{ transcript: string }>()
    transcribeAudioClientDirect.mockResolvedValue(null)
    transcribeAudio.mockReturnValue(relay.promise)
    $sessionStates.set({ shared: sessionState() })

    const view = render(
      <LeaderDialogueRuntime
        clips={clips}
        leaderLabel="owl leader"
        onClose={vi.fn()}
        onLeaderStateChange={vi.fn()}
        onOpenFullChat={vi.fn()}
        owner={{ connectionId: 'source-a', profile: 'owl' }}
        session={{ runtimeId: 'shared', storedId: 'stored-owl' }}
        voiceAvailable
      />
    )

    const pending = voiceHook.onTranscribeAudio!(new Blob(['audio'], { type: 'audio/webm' }))

    await waitFor(() =>
      expect(transcribeAudio).toHaveBeenCalledWith(expect.any(String), 'audio/webm', {
        connectionId: 'source-a',
        profile: 'owl'
      })
    )

    view.rerender(
      <LeaderDialogueRuntime
        clips={clips}
        leaderLabel="fox leader"
        onClose={vi.fn()}
        onLeaderStateChange={vi.fn()}
        onOpenFullChat={vi.fn()}
        owner={{ connectionId: 'source-b', profile: 'fox' }}
        session={{ runtimeId: 'shared', storedId: 'stored-fox' }}
        voiceAvailable
      />
    )
    relay.resolve({ transcript: 'old audio must not reach fox' })

    await expect(pending).rejects.toThrow(/leader voice route changed/i)
  })

  it('rejects a direct capture that completes during an owner-change layout phase before passive effects flush', async () => {
    const direct = deferred<null | string>()
    transcribeAudioClientDirect.mockReturnValue(direct.promise)
    $sessionStates.set({ shared: sessionState() })

    const view = render(
      <LeaderDialogueRuntimeHarness
        owner={{ connectionId: 'source-a', profile: 'owl' }}
        session={{ runtimeId: 'shared', storedId: 'stored-owl' }}
      />
    )

    const pending = voiceHook.onTranscribeAudio!(new Blob(['audio'], { type: 'audio/webm' }))
    const oldSignal = transcribeAudioClientDirect.mock.calls[0]?.[2] as AbortSignal
    const signalAtOwnerCommit = vi.fn()

    view.rerender(
      <LeaderDialogueRuntimeHarness
        onRouteLayout={() => {
          signalAtOwnerCommit(oldSignal.aborted)
          direct.resolve(null)
        }}
        owner={{ connectionId: 'source-b', profile: 'fox' }}
        session={{ runtimeId: 'shared', storedId: 'stored-fox' }}
      />
    )

    expect(signalAtOwnerCommit).toHaveBeenCalledWith(true)
    await expect(pending).rejects.toThrow(/leader voice route changed/i)
    expect(transcribeAudio).not.toHaveBeenCalled()
  })

  it('rejects a relay transcript that completes during an owner-change layout phase before passive effects flush', async () => {
    const relay = deferred<{ transcript: string }>()
    transcribeAudioClientDirect.mockResolvedValue(null)
    transcribeAudio.mockReturnValue(relay.promise)
    $sessionStates.set({ shared: sessionState() })

    const view = render(
      <LeaderDialogueRuntimeHarness
        owner={{ connectionId: 'source-a', profile: 'owl' }}
        session={{ runtimeId: 'shared', storedId: 'stored-owl' }}
      />
    )

    const pending = voiceHook.onTranscribeAudio!(new Blob(['audio'], { type: 'audio/webm' }))

    await waitFor(() => expect(transcribeAudio).toHaveBeenCalledOnce())
    const oldSignal = transcribeAudioClientDirect.mock.calls[0]?.[2] as AbortSignal
    const signalAtOwnerCommit = vi.fn()
    view.rerender(
      <LeaderDialogueRuntimeHarness
        onRouteLayout={() => {
          signalAtOwnerCommit(oldSignal.aborted)
          relay.resolve({ transcript: 'stale audio' })
        }}
        owner={{ connectionId: 'source-b', profile: 'fox' }}
        session={{ runtimeId: 'shared', storedId: 'stored-fox' }}
      />
    )

    expect(signalAtOwnerCommit).toHaveBeenCalledWith(true)
    await expect(pending).rejects.toThrow(/leader voice route changed/i)
  })
})
