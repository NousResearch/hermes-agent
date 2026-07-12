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
  it('drops a short trailing prompt block after a blank separator', () => {
    const snapshot = ['echo hi', 'hi', '', PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe('echo hi\r\nhi')
  })

  it('keeps real output when the tail after the blank line is long', () => {
    const tail = ['line1', 'line2', 'line3', 'line4', 'line5']
    const snapshot = ['start', '', ...tail].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe(['start', '', ...tail].join('\r\n'))
  })

  it('leaves a prompt with no preceding blank line untouched (heuristic limitation)', () => {
    // Default PowerShell prints no blank line before its prompt, so the trailing
    // prompt survives here — the idle-accumulation path (isIdlePromptOnly) is what
    // actually covers that shell, not this trimmer.
    const snapshot = ['echo hi', 'hi', PS_PROMPT].join('\r\n')

    expect(cleanReviveSnapshot(snapshot)).toBe(snapshot)
  })
})
