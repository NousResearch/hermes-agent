import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { sameCronSignature, sessionMessagesSignature } from './session-signatures'

const session = (id: string, title: string | null): SessionInfo => ({ id, title }) as SessionInfo

describe('sameCronSignature', () => {
  it('is false when the lengths differ', () => {
    expect(sameCronSignature([session('a', 't')], [])).toBe(false)
  })

  it('is true when ids and titles match in order', () => {
    const a = [session('a', 'one'), session('b', 'two')]
    const b = [session('a', 'one'), session('b', 'two')]
    expect(sameCronSignature(a, b)).toBe(true)
  })

  it('is false when a title changed', () => {
    const a = [session('a', 'one')]
    const b = [session('a', 'renamed')]
    expect(sameCronSignature(a, b)).toBe(false)
  })

  it('is false when order differs', () => {
    const a = [session('a', 't'), session('b', 't')]
    const b = [session('b', 't'), session('a', 't')]
    expect(sameCronSignature(a, b)).toBe(false)
  })
})

describe('sessionMessagesSignature', () => {
  const msg = (role: string, content: string) =>
    ({ role, content }) as Parameters<typeof sessionMessagesSignature>[0][number]

  it('is stable for identical transcripts', () => {
    expect(sessionMessagesSignature([msg('user', 'hi')])).toBe(sessionMessagesSignature([msg('user', 'hi')]))
  })

  it('changes when content changes', () => {
    expect(sessionMessagesSignature([msg('user', 'hi')])).not.toBe(sessionMessagesSignature([msg('user', 'yo')]))
  })

  it('changes when a message is appended', () => {
    const one = [msg('user', 'hi')]
    expect(sessionMessagesSignature(one)).not.toBe(sessionMessagesSignature([...one, msg('assistant', 'hey')]))
  })
})
