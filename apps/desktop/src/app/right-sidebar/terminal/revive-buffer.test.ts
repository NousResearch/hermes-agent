import { describe, expect, it } from 'vitest'

import { cleanReviveSnapshot, isIdlePromptOnly } from './use-terminal-session'

// A default-PowerShell idle prompt: no blank-line separator before it.
const PS_PROMPT = 'PS C:\\Users\\Aleksandr>'

describe('isIdlePromptOnly', () => {
  it('is true for an empty or whitespace-only buffer', () => {
    expect(isIdlePromptOnly('')).toBe(true)
    expect(isIdlePromptOnly('\r\n  \r\n')).toBe(true)
  })

  it('is true for a lone prompt line', () => {
    expect(isIdlePromptOnly(PS_PROMPT)).toBe(true)
  })

  it('is true for a buffer that is only repeated identical prompts (accumulation)', () => {
    expect(isIdlePromptOnly([PS_PROMPT, PS_PROMPT, PS_PROMPT].join('\r\n'))).toBe(true)
  })

  it('ignores blank gaps between repeated prompts (the "gapped" variant)', () => {
    expect(isIdlePromptOnly([PS_PROMPT, '', '', PS_PROMPT].join('\r\n'))).toBe(true)
  })

  it('is false when the buffer holds a real command and output', () => {
    expect(isIdlePromptOnly([PS_PROMPT, 'cd project', 'PS C:\\Users\\Aleksandr\\project>'].join('\r\n'))).toBe(false)
  })

  it('is false when two different prompts are present (cwd actually changed)', () => {
    expect(isIdlePromptOnly([PS_PROMPT, 'PS C:\\Users\\Aleksandr\\project>'].join('\r\n'))).toBe(false)
  })
})

describe('cleanReviveSnapshot', () => {
  it('drops a spaced trailing prompt block after a blank separator (starship)', () => {
    const snapshot = ['echo hi', 'hi', '', PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('echo hi\r\nhi')
  })

  it('drops a multi-line prompt block after a blank separator (powerline)', () => {
    const snapshot = ['work', '', '┌─ user@host ~/project', '└─$'].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('work')
  })

  it('drops a single-line trailing prompt with no preceding blank line (PowerShell)', () => {
    // Default PowerShell prints no blank line before its prompt; the fresh shell
    // reprints it on boot, so the redundant idle prompt must be trimmed here.
    const snapshot = ['echo hi', 'hi', PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('echo hi\r\nhi')
  })

  it('keeps command output and drops only the trailing prompt on a long history', () => {
    const history = ['cmd1', 'out1', 'cmd2', 'out2']
    const snapshot = [...history, PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe(history.join('\r\n'))
  })

  it('reduces a lone prompt to an empty buffer', () => {
    expect(cleanReviveSnapshot(PS_PROMPT)).toBe('')
    expect(cleanReviveSnapshot([PS_PROMPT, '', ''].join('\r\n'))).toBe('')
  })

  it('returns empty for a blank-only buffer without throwing', () => {
    expect(cleanReviveSnapshot('')).toBe('')
    expect(cleanReviveSnapshot('\r\n  \r\n')).toBe('')
  })
})
