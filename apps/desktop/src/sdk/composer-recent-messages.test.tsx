import { renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'

import { ComposerScopeProvider, MAIN_COMPOSER_SCOPE } from '@/app/chat/composer/scope'
import { assistantTextPart, type ChatMessage, textPart } from '@/lib/chat-messages'

import { useThisComposer } from './composer'

const user = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  parts: [textPart(text)],
  role: 'user',
  ...extra
})

const assistant = (id: string, text: string, extra: Partial<ChatMessage> = {}): ChatMessage => ({
  id,
  parts: [assistantTextPart(text)],
  role: 'assistant',
  ...extra
})

function readerFor(messages: ChatMessage[]) {
  const $messages = atom(messages)

  const wrapper = ({ children }: { children: ReactNode }) => (
    <ComposerScopeProvider value={{ ...MAIN_COMPOSER_SCOPE, $messages, target: 'tile:b' }}>{children}</ComposerScopeProvider>
  )

  return { $messages, read: renderHook(() => useThisComposer(), { wrapper }).result.current.recentMessages }
}

describe('useThisComposer().recentMessages', () => {
  it('reads that composer\'s visible user and assistant text, newest last, at most `limit`', () => {
    const { read } = readerFor([
      user('1', 'first'),
      assistant('2', 'reply one'),
      { id: 's', parts: [textPart('system note')], role: 'system' } as ChatMessage,
      user('3', 'hidden prompt', { hidden: true }),
      assistant('4', 'still streaming', { pending: true }),
      user('5', 'second')
    ])

    expect(read(10)).toEqual([
      { role: 'user', text: 'first' },
      { role: 'assistant', text: 'reply one' },
      { role: 'user', text: 'second' }
    ])
    expect(read(2)).toEqual([
      { role: 'assistant', text: 'reply one' },
      { role: 'user', text: 'second' }
    ])
  })

  it('reads the transcript at call time, and a zero or negative limit reads nothing', () => {
    const { $messages, read } = readerFor([user('1', 'first')])
    $messages.set([...$messages.get(), assistant('2', 'later')])

    expect(read(5).at(-1)).toEqual({ role: 'assistant', text: 'later' })
    expect(read(0)).toEqual([])
    expect(read(-3)).toEqual([])
  })
})
