// @vitest-environment jsdom
import { expect, test } from 'vitest'

import { newCompletedPair, type TranscriptMessage, transcriptMessages } from './desktop-chat-smoke.ts'
import { MOCK_REPLY } from './mock-server.ts'

interface TranscriptFixture {
  viewport: HTMLElement
  user: HTMLElement
  assistant: HTMLElement
}

function transcript(legacy: boolean): TranscriptFixture {
  const viewport = document.createElement('div')
  const user = document.createElement('div')
  user.dataset.slot = 'aui_user-message-root'
  user.dataset.role = 'user'
  user.textContent = 'new checkpoint nonce'

  // v2026.6.19 StickyHumanMessageContainer drops MessagePrimitive.Root's ID.
  if (!legacy) { user.dataset.messageId = 'new-user' }
  const assistant = document.createElement('div')
  assistant.dataset.role = 'assistant'
  assistant.dataset.messageId = 'new-assistant'
  assistant.textContent = MOCK_REPLY
  viewport.append(user, assistant)

  return { viewport, user, assistant }
}

test('historical and current user bubbles retain ordered fresh-turn proof', (): void => {
  for (const legacy of [true, false]) {
    const { viewport, user, assistant } = transcript(legacy)
    const read = (): TranscriptMessage[] => transcriptMessages([viewport])
    const pair = newCompletedPair(read(), [], 'new checkpoint nonce')
    expect(pair?.user.text).toBe(user.textContent)
    expect(pair?.assistant.id).toBe('new-assistant')

    if (legacy) {
      expect(pair?.user.id).toBe(`legacy-user-text:${user.textContent}`)
      expect(pair?.user.idSource).toBe('legacy-user-text')
    } else {
      expect(pair?.user.id).toBe('new-user')
      expect(pair?.user.idSource).toBeUndefined()
    }

    // Remounting preserved history cannot turn it into a new checkpoint.
    const before = read().map((message: TranscriptMessage): string => message.id)
    const remounted = transcript(legacy)
    expect(newCompletedPair(transcriptMessages([remounted.viewport]), before, 'new checkpoint nonce')).toBeNull()
    user.textContent = 'next checkpoint nonce'

    if (!legacy) { user.dataset.messageId = 'next-user' }
    // A fresh user still cannot reuse the previous assistant's real ID.
    expect(newCompletedPair(read(), before, 'next checkpoint nonce')).toBeNull()
    assistant.dataset.messageId = 'next-assistant'
    expect(newCompletedPair(read(), before, 'next checkpoint nonce')).not.toBeNull()
  }
})

test('legacy extraction never invents an assistant identity or accepts incomplete replies', (): void => {
  const { viewport, user, assistant } = transcript(true)
  const pair = (): ReturnType<typeof newCompletedPair> => newCompletedPair(transcriptMessages([viewport]), [], 'new checkpoint nonce')
  delete assistant.dataset.messageId
  expect(pair()).toBeNull()
  assistant.dataset.messageId = 'new-assistant'
  assistant.dataset.streaming = 'true'
  expect(pair()).toBeNull()
  delete assistant.dataset.streaming
  const marker = document.createElement('span')
  marker.dataset.messageStreaming = 'true'
  assistant.append(marker)
  expect(pair()).toBeNull()
  marker.remove()
  const alert = document.createElement('span')
  alert.setAttribute('role', 'alert')
  assistant.append(alert)
  expect(pair()).toBeNull()
  alert.remove()
  assistant.textContent = 'The full boot chain is working.'
  expect(pair()).toBeNull()
  assistant.textContent = MOCK_REPLY
  const interveningUser = user.cloneNode(true)
  viewport.insertBefore(interveningUser, assistant)
  expect(pair()).toBeNull()
  viewport.removeChild(interveningUser)
  expect(pair()).not.toBeNull()
  delete user.dataset.slot
  expect(pair()).toBeNull()
})
