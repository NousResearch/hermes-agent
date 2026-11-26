import { cleanup } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'
import { dispatchNativeNotification } from '@/store/native-notifications'

import { playFrames } from './test-harness'

vi.mock(import('@/store/native-notifications'), async importOriginal => ({
  ...(await importOriginal()),
  dispatchNativeNotification: vi.fn()
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it('settles a silent completion without an assistant bubble or a native notification', async () => {
  const messages = await playFrames('session', [
    ['message.start', {}],
    ['message.complete', { text: '', silent: true, status: 'complete' }]
  ])

  expect(messages).toEqual([])
  expect(dispatchNativeNotification).not.toHaveBeenCalled()
})

it('does not interpret marker text in live or reloaded human-turn replies', async () => {
  const messages = await playFrames('session', [
    ['message.start', {}],
    ['message.delta', { text: '[SILENT]' }],
    ['message.complete', { text: '[SILENT]', status: 'complete' }]
  ])

  expect(messages.map(chatMessageText)).toEqual(['[SILENT]'])
  expect(dispatchNativeNotification).toHaveBeenCalledOnce()

  const row = { role: 'assistant' as const, content: '[SILENT]', timestamp: 1 }
  expect(toChatMessages([row]).map(chatMessageText)).toEqual(['[SILENT]'])
  expect(toChatMessages([{ ...row, display_kind: 'hidden' }])).toEqual([])
})
