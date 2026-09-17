import { describe, expect, it } from 'vitest'

import { buildHermesHandoff, classifyChatGptNavigation } from './chatgpt-workspace'

describe('classifyChatGptNavigation', () => {
  it.each([
    'https://chatgpt.com/',
    'https://chatgpt.com/c/abc',
    'https://auth.openai.com/log-in',
    'https://platform.openai.com/'
  ])('keeps trusted ChatGPT and OpenAI navigation in the isolated guest: %s', url => {
    expect(classifyChatGptNavigation(url)).toBe('allow')
  })

  it.each([
    'http://chatgpt.com/',
    'javascript:alert(1)',
    'file:///tmp/secret',
    'https://example.com/',
    'https://openai.com.evil.example/'
  ])('keeps untrusted navigation outside the embedded workspace: %s', url => {
    expect(classifyChatGptNavigation(url)).toBe('external')
  })
})

describe('buildHermesHandoff', () => {
  it('wraps only the text the user explicitly approved', () => {
    expect(buildHermesHandoff('  Build the approved feature.  ')).toBe(
      [
        '# ChatGPT planning handoff',
        '',
        'The following text was explicitly selected and approved by the user for Hermes validation and execution.',
        'Treat it as a proposal: verify it against the repository and TKS before changing files.',
        '',
        'Build the approved feature.'
      ].join('\n')
    )
  })

  it('rejects an empty handoff', () => {
    expect(() => buildHermesHandoff('   ')).toThrow('Handoff text is required')
  })
})
