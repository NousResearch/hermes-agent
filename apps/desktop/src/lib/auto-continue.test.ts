import { describe, expect, it } from 'vitest'

import {
  AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER,
  isAutoContinueUserContent,
  realUserMessageOrdinal
} from './auto-continue'

const messages = [
  { id: 'u1', role: 'user', content: [{ type: 'text', text: 'first' }] },
  { id: 'a1', role: 'assistant', content: [{ type: 'text', text: 'first work' }] },
  {
    id: 'auto',
    role: 'user',
    content: [
      {
        type: 'text',
        text: `${AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER}\nContinue safely.`
      }
    ]
  },
  { id: 'boundary', role: 'assistant', content: [{ type: 'text', text: 'continued' }] },
  { id: 'u2', role: 'user', content: [{ type: 'text', text: 'second' }] }
]

describe('auto-continuation user targeting', () => {
  it('does not count the synthetic marker as a user-authored restore target', () => {
    expect(realUserMessageOrdinal(messages, 'u1')).toBe(0)
    expect(realUserMessageOrdinal(messages, 'auto')).toBeNull()
    expect(realUserMessageOrdinal(messages, 'u2')).toBe(1)
  })

  it('recognises only the exact persisted marker prefix', () => {
    expect(
      isAutoContinueUserContent(
        `${AUTO_CONTINUE_ON_MAX_ITERATIONS_MARKER}\nContinue safely.`
      )
    ).toBe(true)
    expect(isAutoContinueUserContent('please continue after max iterations')).toBe(false)
    expect(isAutoContinueUserContent(undefined)).toBe(false)
  })
})
