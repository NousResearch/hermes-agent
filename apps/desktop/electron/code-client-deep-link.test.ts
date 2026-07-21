import { describe, expect, it } from 'vitest'

import { codeClientDeepLink } from './code-client-deep-link'

describe('codeClientDeepLink', () => {
  it('builds an inert Codex composer deep link', () => {
    expect(
      codeClientDeepLink({ client: 'codex', cwd: '/Users/test/repo', prompt: 'Review notes/today.md' })
    ).toBe('codex://new?path=%2FUsers%2Ftest%2Frepo&prompt=Review%20notes%2Ftoday.md')
  })

  it('builds an inert Claude Code composer deep link', () => {
    expect(
      codeClientDeepLink({ client: 'claude-code', cwd: '/Users/test/repo', prompt: 'Review notes/today.md' })
    ).toBe('claude-cli://open?cwd=%2FUsers%2Ftest%2Frepo&q=Review%20notes%2Ftoday.md')
  })

  it('accepts absolute Windows paths without shell interpretation', () => {
    expect(codeClientDeepLink({ client: 'codex', cwd: 'C:\\Users\\test\\repo', prompt: '' }, 'win32')).toBe(
      'codex://new?path=C%3A%5CUsers%5Ctest%5Crepo'
    )
  })

  it.each(['/rooted', '\\rooted'])('rejects drive-relative Windows roots: %s', cwd => {
    expect(() => codeClientDeepLink({ client: 'codex', cwd, prompt: '' }, 'win32')).toThrow(
      'Invalid local working directory'
    )
  })

  it.each([
    ['', 'empty'],
    ['relative/repo', 'relative'],
    ['../repo', 'traversal'],
    ['/Users/test/../repo', 'absolute traversal'],
    ['//server/share', 'network'],
    ['/\\server\\share', 'mixed-separator network'],
    ['\\/server/share', 'mixed-separator network'],
    ['/Users/test/repo\nother', 'control']
  ])('rejects %s paths (%s)', cwd => {
    expect(() => codeClientDeepLink({ client: 'codex', cwd, prompt: '' })).toThrow('Invalid local working directory')
  })

  it('rejects unsupported clients', () => {
    expect(() => codeClientDeepLink({ client: 'terminal' as never, cwd: '/Users/test/repo', prompt: '' })).toThrow(
      'Unsupported code client'
    )
  })

  it.each([null, {}, { client: 'codex', cwd: '/Users/test/repo' }, { client: 'codex', cwd: 42, prompt: '' }])(
    'rejects malformed IPC payloads',
    input => {
      expect(() => codeClientDeepLink(input as never)).toThrow('Invalid code client request')
    }
  )

  it('rejects oversized prompts', () => {
    expect(() => codeClientDeepLink({ client: 'claude-code', cwd: '/Users/test/repo', prompt: 'x'.repeat(5001) })).toThrow(
      'Prompt exceeds 5000 characters'
    )
  })
})
