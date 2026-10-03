import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import { en } from '@/i18n/en'
import { $memoryReview } from '@/store/memory-review'
import { $activeGatewayProfile } from '@/store/profile'
import { $connection } from '@/store/session'
import { useSlashCommand } from './slash'

afterEach(() => {
  cleanup()
  $memoryReview.set(null)
})

it('bare memory opens review for focused session while explicit args retain slash execution', async () => {
  const request = vi.fn(async (method: string) =>
    method === 'memory.pending' ? { write_approval: false, batches: [] } : { output: 'Generic result' }
  )
  const append = vi.fn()
  const noop = vi.fn()
  const { result } = renderHook(() =>
    useSlashCommand({
      activeSessionIdRef: { current: 'primary' },
      selectedStoredSessionIdRef: { current: null },
      requestGateway: request as never,
      appendSessionTextMessage: append,
      branchCurrentSession: vi.fn(async () => true),
      busyRef: { current: false },
      copy: en.desktop,
      createBackendSessionForSend: vi.fn(async () => 'new'),
      getRoutedStoredSessionId: () => null,
      getRuntimeIdForStoredSession: () => null,
      handleSkinCommand: () => '',
      handoffSession: vi.fn(async () => ({ ok: true })),
      openMemoryGraph: noop,
      refreshSessions: vi.fn(async () => {}),
      resumeStoredSession: noop,
      startFreshSessionDraft: noop,
      submitPromptText: vi.fn(async () => true),
      updateSessionState: vi.fn() as never
    })
  )
  await act(() => result.current('/memory', { sessionId: 'focused', typed: false }))
  expect($memoryReview.get()?.sessionId).toBe('focused')
  expect(request).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  await act(() => result.current('/memory approve abcdef12', { sessionId: 'focused', typed: false }))
  expect(request).toHaveBeenCalledWith('slash.exec', { session_id: 'focused', command: 'memory approve abcdef12' })
  $memoryReview.set(null)
  request.mockRejectedValueOnce(new Error('Method not found: memory.pending'))
  await act(() => result.current('/memory', { sessionId: 'focused', typed: false }))
  expect($memoryReview.get()).toBeNull()
  expect(request).toHaveBeenCalledWith('slash.exec', { session_id: 'focused', command: 'memory' })
  for (const switchScope of [
    () => $activeGatewayProfile.set('generic-preflight-profile'),
    () =>
      $connection.set({ ...$connection.get(), connectionId: 'generic-preflight-connection' } as NonNullable<
        ReturnType<typeof $connection.get>
      >)
  ]) {
    let finish!: (value: { write_approval: boolean; batches: never[] }) => void
    request.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )
    let command!: Promise<unknown>
    act(() => {
      command = result.current('/memory', { sessionId: 'focused', typed: false })
    })
    await act(async () => {
      await Promise.resolve()
    })
    act(switchScope)
    await act(async () => {
      finish({ write_approval: false, batches: [] })
      await command
    })
    expect($memoryReview.get()).toBeNull()
  }
})
