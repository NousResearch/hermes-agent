import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import * as voicePlayback from '@/lib/voice-playback'
import { $composerAttachments, clearComposerTerminalSelections, setComposerTerminalSelection } from '@/store/composer'
import * as onboarding from '@/store/onboarding'
import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { useSubmitPrompt } from './submit'
import type { GatewayRequest } from './utils'

const RUNTIME = 'rt-native'
const STORED = 'stored-native'

function setup(requestGateway: GatewayRequest, runtime: null | string = RUNTIME) {
  let state = createClientSessionState(STORED)
  const projection = vi.fn()
  const create = vi.fn(async () => 'unexpected-new-runtime')
  const resume = vi.fn()
  const sync = vi.fn(async (sessionId: string) => ({ sessionId, attachments: [] }))

  const hook = renderHook(() =>
    useSubmitPrompt({
      activeSessionIdRef: { current: runtime },
      busyRef: { current: false },
      copy: en.desktop,
      createBackendSessionForSend: create,
      getRoutedStoredSessionId: () => STORED,
      getRuntimeIdForStoredSession: () => runtime,
      getRouteToken: () => 'native-test-route',
      requestGateway,
      runtimeIdByStoredSessionIdRef: { current: new Map(runtime ? [[STORED, runtime]] : []) },
      resumeStoredSession: resume,
      selectedStoredSessionIdRef: { current: STORED },
      syncAttachmentsForSubmit: sync,
      updateSessionState: (id, updater) => {
        state = updater(state)
        projection(id, state)

        return state
      }
    })
  )

  const send = () =>
    hook.result.current('confirmed plain @terminal:shell:1', {
      attachments: [],
      composerScope: STORED,
      confirmedExternal: true,
      sessionId: runtime,
      storedSessionId: STORED
    })

  return { create, hook, projection, resume, send, sync }
}

beforeEach(() => {
  $sessions.set([{ id: STORED, connection_id: 'local', profile: 'default' } as never])
  $sessionStates.set({ [RUNTIME]: createClientSessionState(STORED) })
  $composerAttachments.set([{ id: 'local', kind: 'file', label: 'unsent.txt', refText: '@file:unsent.txt' }])
  setComposerTerminalSelection('shell:1', 'PRIVATE TERMINAL SELECTION')
})

afterEach(() => {
  cleanup()
  clearComposerTerminalSelections()
  $composerAttachments.set([])
  $sessionStates.set({})
  $sessions.set([])
  vi.restoreAllMocks()
})

describe('confirmed native prompt transport and projection', () => {
  it('does not barge into playback, consume voice interruption or open onboarding', async () => {
    const request = vi.fn(async () => ({ status: 'streaming' })) as unknown as GatewayRequest
    const stop = vi.spyOn(voicePlayback, 'stopVoicePlayback')
    const consumeVoice = vi.spyOn(voicePlayback, 'takeVoicePlaybackInterrupted')
    const consumeWarning = vi.spyOn(onboarding, 'consumePendingCredentialWarning')
    vi.spyOn(voicePlayback, 'isVoicePlaybackActive').mockReturnValue(true)
    const { send } = setup(request)
    await act(async () => {
      expect(await send()).toBe(true)
    })
    expect(stop).not.toHaveBeenCalled()
    expect(consumeVoice).not.toHaveBeenCalled()
    expect(consumeWarning).not.toHaveBeenCalled()
  })

  it('refuses a queued target with a missing binding instead of resuming or using the foreground', async () => {
    const request = vi.fn()
    const { create, resume, hook } = setup(request, null)
    await act(async () => {
      expect(
        await hook.result.current('queued confirmed', {
          attachments: [],
          confirmedExternal: true,
          fromQueue: true,
          sessionId: 'stale-runtime',
          storedSessionId: STORED
        })
      ).toBe(false)
    })
    expect(request).not.toHaveBeenCalled()
    expect(create).not.toHaveBeenCalled()
    expect(resume).not.toHaveBeenCalled()
  })

  it.each(['streaming', 'queued'] as const)(
    'uses real %s acknowledgement, one visible row and no local chips',
    async status => {
      const request = vi.fn(async () => ({ status, user_row_id: 42 })) as unknown as GatewayRequest
      const { hook, projection, sync } = setup(request)
      const acknowledged = vi.fn()
      await act(async () => {
        expect(
          await hook.result.current('confirmed plain @terminal:shell:1', {
            attachments: [],
            composerScope: STORED,
            confirmedExternal: true,
            sessionId: RUNTIME,
            storedSessionId: STORED,
            onExternalAccepted: acknowledged
          })
        ).toBe(true)
      })

      expect(request).toHaveBeenCalledExactlyOnceWith(
        'prompt.submit',
        expect.objectContaining({
          queued: true,
          session_id: RUNTIME,
          text: 'confirmed plain @terminal:shell:1'
        }),
        expect.any(Number)
      )
      expect(acknowledged).toHaveBeenCalledExactlyOnceWith(status === 'queued')
      expect(sync).toHaveBeenCalledWith(RUNTIME, [], expect.any(Object))
      const projected = projection.mock.calls.at(-1)![1]
      expect(projected.messages.filter((message: { role: string }) => message.role === 'user')).toHaveLength(1)
      expect(projected.messages.at(-1).rowId).toBe(42)
      expect($composerAttachments.get()[0]?.label).toBe('unsent.txt')
    }
  )

  it.each(['session not found', 'session is busy', 'request timed out', 'connection lost'])(
    'never replays after %s',
    async message => {
      const request = vi.fn(async () => {
        throw new Error(message)
      })

      const { create, resume, send } = setup(request)
      await act(async () => {
        await expect(send()).rejects.toThrow(message)
      })
      expect(request).toHaveBeenCalledTimes(1)
      expect(create).not.toHaveBeenCalled()
      expect(resume).not.toHaveBeenCalled()
    }
  )

  it.each([undefined, { status: 'accepted' }, { status: 'error' }])(
    'does not invent an acknowledgement for %j',
    async reply => {
      const request = vi.fn(async () => reply) as unknown as GatewayRequest
      const { send } = setup(request)
      await act(async () => {
        await expect(send()).rejects.toThrow('outcome unknown')
      })
      expect(request).toHaveBeenCalledTimes(1)
    }
  )

  it('refuses an unhydrated target without creating, resuming or staging attachments', async () => {
    const request = vi.fn()
    const { create, resume, send, sync } = setup(request, null)
    await act(async () => {
      expect(await send()).toBe(false)
    })
    expect(request).not.toHaveBeenCalled()
    expect(create).not.toHaveBeenCalled()
    expect(resume).not.toHaveBeenCalled()
    expect(sync).not.toHaveBeenCalled()
  })
})
