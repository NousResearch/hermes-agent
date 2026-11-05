import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { createClientSessionState } from '@/lib/chat-runtime'
import * as queue from '@/store/composer-queue'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection, $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { queueKickoffIfSessionBusy } from './queue-if-busy'
import { useSlashCommand } from './slash'
import type { GatewayRequest } from './utils'

beforeEach(() => {
  $sessions.set([])
  $connection.set(null)
  $activeGatewayProfile.set('default')
  $sessionStates.set({
    'runtime-a': { ...createClientSessionState(), busy: true, storedSessionId: 'stored-a' }
  })
  queue.$queuedPromptsBySession.set({})
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  queue.$queuedPromptsBySession.set({})
  $sessionStates.set({})
})

it('leaves canonical busy kickoff admission to the submit pipeline instead of the local queue', () => {
  $connection.set({ mode: 'local', wsUrl: 'ws://localhost/api/ws?native_dial=unminted' } as never)
  expect(queueKickoffIfSessionBusy({ sessionId: 'runtime-a', text: 'expanded' })).toBe('idle')
  expect(queue.getQueuedPrompts('stored-a')).toEqual([])
})

it('forwards kickoff identity to queue admission and leaves idle targets alone', () => {
  const enqueue = vi.spyOn(queue, 'enqueueQueuedPrompt')
  const input = { id: 'caller-kickoff', sessionId: 'runtime-a', text: 'expanded skill', displayText: '/work' }

  expect(queueKickoffIfSessionBusy(input)).toBe('queued')
  expect(enqueue).toHaveBeenCalledWith('stored-a', {
    id: input.id,
    text: input.text,
    displayText: input.displayText,
    attachments: []
  })
  expect(queueKickoffIfSessionBusy({ ...input, sessionId: 'idle-target', foregroundBusy: true })).toBe('idle')
  expect(enqueue).toHaveBeenCalledTimes(1)
})

it('keeps caller-provided and generated slash IDs when a busy target queues the expanded skill', async () => {
  const enqueue = vi.spyOn(queue, 'enqueueQueuedPrompt')
  const generatedId = 'a6c67295-d5e0-4edb-b6f0-8aeb144efb77'
  vi.spyOn(crypto, 'randomUUID').mockReturnValue(generatedId)
  const submitPromptText = vi.fn(async () => true)

  const { result } = renderHook(() =>
    useSlashCommand({
      activeSessionIdRef: { current: 'runtime-a' },
      selectedStoredSessionIdRef: { current: 'stored-a' },
      busyRef: { current: false },
      copy: en.desktop,
      createBackendSessionForSend: async () => null,
      getRoutedStoredSessionId: () => 'stored-a',
      getRuntimeIdForStoredSession: () => 'runtime-a',
      requestGateway: vi.fn(async () => ({ type: 'skill', name: 'work', message: 'expanded skill' })) as GatewayRequest,
      resumeStoredSession: vi.fn(),
      updateSessionState: vi.fn((_sid, updater) => updater(createClientSessionState())),
      submitPromptText,
      appendSessionTextMessage: vi.fn(),
      branchCurrentSession: async () => true,
      handleSkinCommand: () => '',
      handoffSession: async () => ({ ok: true }),
      openMemoryGraph: vi.fn(),
      refreshSessions: async () => undefined,
      startFreshSessionDraft: vi.fn()
    })
  )

  for (const submissionId of ['caller-slash', undefined]) {
    await act(async () => {
      await result.current('/work', { submission_id: submissionId })
    })
    expect(enqueue).toHaveBeenLastCalledWith(
      'stored-a',
      expect.objectContaining({
        id: submissionId ?? generatedId,
        text: 'expanded skill'
      })
    )
  }

  expect(submitPromptText).not.toHaveBeenCalled()
})
