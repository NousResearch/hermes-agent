// @vitest-environment jsdom
import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'

import type { RealtimeVoiceHandlers } from '@/lib/realtime-voice'
import { $subagentsBySession, type SubagentProgress } from '@/store/subagents'

import type { ReplyMessage } from './agent-reply'
import { useRealtimeConversation } from './use-realtime-conversation'

const mocks = vi.hoisted(() => ({
  handlers: null as RealtimeVoiceHandlers | null,
  notify: vi.fn(() => true),
  request: vi.fn(async () => ({ found: true, status: 'queued' }))
}))

vi.mock('@/lib/live-voice/start', () => ({
  startLiveVoice: async (handlers: RealtimeVoiceHandlers) => {
    mocks.handlers = handlers

    return { notify: mocks.notify, setMuted: vi.fn(), stop: vi.fn() }
  }
}))
vi.mock('@/store/session-states', () => ({ requestForOwnedSession: mocks.request }))
vi.mock('@/store/notifications', () => ({ notifyError: vi.fn() }))
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
  $subagentsBySession.set({})
})

test('adopts a newly created conversation but never voices its report after switching away', async () => {
  vi.useFakeTimers()
  let finish!: () => void

  const submitted = new Promise<void>(resolve => {
    finish = resolve
  })

  const messages: ReplyMessage[] = []

  const { rerender } = renderHook(
    ({ sessionId }: { sessionId: string | null }) =>
      useRealtimeConversation({
        sessionId,
        enabled: true,
        failureLabel: 'error',
        busy: () => false,
        messages: () => messages,
        markSpoken: vi.fn(),
        onFatalError: vi.fn(),
        onSubmit: () => submitted
      }),
    { initialProps: { sessionId: null as string | null } }
  )

  await act(async () => {})
  await act(async () => {
    await mocks.handlers!.onAsk('Przygotuj raport')
  })
  rerender({ sessionId: 'created' })
  messages.push({ id: 'answer', role: 'assistant', parts: [{ type: 'text', text: 'Potwierdzony wynik' }] })
  await act(async () => {
    finish()
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(mocks.notify).toHaveBeenCalledWith('Potwierdzony wynik')
  mocks.notify.mockClear()
  await act(async () => {
    await mocks.handlers!.onAsk('Kolejny raport')
  })
  rerender({ sessionId: 'other' })
  messages.push({ id: 'other-answer', role: 'assistant', parts: [{ type: 'text', text: 'Inna rozmowa' }] })
  await act(async () => {
    await vi.advanceTimersByTimeAsync(3000)
  })
  expect(mocks.notify).not.toHaveBeenCalled()
})

test('acknowledges a task without waiting for its result, then voices only the real reply', async () => {
  vi.useFakeTimers()
  let finish!: () => void

  const submitted = new Promise<void>(resolve => {
    finish = resolve
  })

  const messages: ReplyMessage[] = []
  const onSubmit = vi.fn(() => submitted)
  renderHook(() =>
    useRealtimeConversation({
      sessionId: 's1',
      enabled: true,
      failureLabel: 'error',
      busy: () => false,
      messages: () => messages,
      markSpoken: vi.fn(),
      onFatalError: vi.fn(),
      onSubmit
    })
  )
  await act(async () => {})
  let acknowledgement = ''
  await act(async () => {
    acknowledgement = await mocks.handlers!.onAsk('Zrób raport')
  })
  expect(onSubmit).toHaveBeenCalledOnce()
  expect(acknowledgement).toContain('nie ukończenia')
  expect(mocks.notify).not.toHaveBeenCalled()
  messages.push({ id: 'answer', role: 'assistant', parts: [{ type: 'text', text: 'Raport zapisany w wynik.txt' }] })
  await act(async () => {
    finish()
    await vi.advanceTimersByTimeAsync(1500)
  })
  expect(mocks.notify).toHaveBeenCalledWith('Raport zapisany w wynik.txt')
})

test.each([
  ['zatrzymaj zadanie', 'interrupt'],
  ['zmień polecenie: zapisz CSV', 'steer']
])('spoken control %s targets the session-owned worker', async (request, method) => {
  vi.useFakeTimers()

  const item: SubagentProgress = {
    id: 'worker',
    goal: 'Report',
    parentId: null,
    status: 'running',
    taskCount: 1,
    taskIndex: 0,
    startedAt: 0,
    updatedAt: 0,
    filesRead: [],
    filesWritten: [],
    stream: []
  }

  $subagentsBySession.set({ s1: [item] })
  const onSubmit = vi.fn()
  renderHook(() =>
    useRealtimeConversation({
      sessionId: 's1',
      enabled: true,
      failureLabel: 'error',
      busy: () => true,
      messages: () => [],
      markSpoken: vi.fn(),
      onFatalError: vi.fn(),
      onSubmit
    })
  )
  await act(async () => {})
  const reply = await mocks.handlers!.onAsk(request)
  expect(mocks.request).toHaveBeenCalledWith('s1', expect.any(Function), `subagent.${method}`, {
    session_id: 's1',
    subagent_id: 'worker',
    ...(method === 'steer' ? { text: request } : {})
  })
  expect(reply).toContain(method === 'interrupt' ? 'poczekaj na potwierdzenie' : 'Zmiana została przekazana')
  expect(onSubmit).not.toHaveBeenCalled()
})
