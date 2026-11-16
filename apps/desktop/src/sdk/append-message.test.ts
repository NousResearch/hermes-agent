import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { toChatMessages } from '@/lib/chat-messages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $gateway } from '@/store/gateway'
import { $messages, setActiveSessionId, setMessages, setSessionOwnerHint } from '@/store/session'
import { $sessionStates, clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { appendMessage } from './append-message'

const mocks = vi.hoisted(() => ({ request: vi.fn() }))
vi.mock('@/store/gateway', async importOriginal => ({
  ...((await importOriginal()) as Record<string, unknown>),
  requestGatewayForAgent: mocks.request
}))

beforeEach(() => {
  mocks.request.mockReset()
  setActiveSessionId('speech')
  setSessionOwnerHint('speech', { connectionId: 'speech-owner', profile: 'voice' })
  publishSessionState('speech', createClientSessionState('speech'))
  setMessages([])
})

afterEach(() => {
  setActiveSessionId(null)
  setMessages([])
  clearAllSessionStates()
  vi.restoreAllMocks()
})

it('persists through the target owner before painting, including after a chat switch', async () => {
  let acknowledge!: (receipt: unknown) => void
  mocks.request.mockReturnValue(
    new Promise(resolve => {
      acknowledge = resolve
    })
  )
  expect(appendMessage({ role: 'user', text: 'What I said', clientId: 'speech-1' })).toBe(true)
  expect($messages.get()).toEqual([])
  expect($sessionStates.get().speech.messages).toEqual([])
  expect(mocks.request.mock.calls[0].slice(0, 4)).toEqual([
    'speech-owner',
    'voice',
    'session.append_message',
    { session_id: 'speech', role: 'user', content: 'What I said', client_id: 'speech-1' }
  ])
  setActiveSessionId('other')
  acknowledge({ ok: true, id: 7, message_uid: 'uid-7' })
  await vi.waitFor(() => expect($sessionStates.get().speech.messages).toHaveLength(1))
  expect($messages.get()).toEqual([])
})

it('deduplicates a repeated receipt, including a message already hydrated from storage', async () => {
  mocks.request.mockResolvedValue({ ok: true, id: 7, message_uid: 'uid-7' })
  appendMessage({ text: 'Spoken reply', clientId: 'reply-1' })
  await vi.waitFor(() => expect($messages.get()).toHaveLength(1))
  setMessages(toChatMessages([{ role: 'assistant', content: 'Spoken reply', row_id: 99, message_uid: 'uid-7' }]))
  appendMessage({ text: 'Spoken reply', clientId: 'reply-1' })
  await vi.waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(2))
  expect($messages.get()).toHaveLength(1)
  expect($sessionStates.get().speech.messages).toHaveLength(1)
})

it('does not paint rejected or unavailable writes', async () => {
  const warning = vi.spyOn(console, 'warn').mockImplementation(() => {})
  mocks.request.mockRejectedValue(new Error('storage_unavailable'))
  appendMessage({ text: 'Uncommitted reply' })
  await vi.waitFor(() => expect(warning).toHaveBeenCalled())
  expect($messages.get()).toEqual([])
  expect($sessionStates.get().speech.messages).toEqual([])
  $gateway.set(null)
  setActiveSessionId(null)
  expect(appendMessage({ text: 'No target' })).toBe(false)
  expect(appendMessage({ sessionId: 'speech', role: 'tool', text: 'Invalid' })).toBe(false)
})
