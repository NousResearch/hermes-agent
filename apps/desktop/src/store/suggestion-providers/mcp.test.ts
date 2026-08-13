import { describe, expect, it } from 'vitest'

import { matchSuggestions } from './mcp'

const INDEX = [
  { keywords: ['linear', 'issue tracker', 'ticket'], server: 'linear' },
  { keywords: ['figma', 'design'], server: 'figma' },
  { keywords: ['unreal', 'ue5'], server: 'unreal-engine' }
]

describe('matchSuggestions', () => {
  it('matches a whole word and reports the keyword that hit', () => {
    expect(matchSuggestions('can you check the linear board', INDEX)).toEqual([{ keyword: 'linear', server: 'linear' }])
  })

  it('does not match inside other words', () => {
    // "linearly" must not suggest Linear — the classic false positive.
    expect(matchSuggestions('this scales linearly with input', INDEX)).toEqual([])
  })

  it('matches multi-word keywords as phrases', () => {
    expect(matchSuggestions('our issue tracker is a mess', INDEX)).toEqual([
      { keyword: 'issue tracker', server: 'linear' }
    ])
  })

  it('is case-insensitive against the draft', () => {
    expect(matchSuggestions('open FIGMA please', INDEX)).toEqual([{ keyword: 'figma', server: 'figma' }])
  })

  it('caps the number of suggestions', () => {
    const matches = matchSuggestions('linear ticket for the figma design in unreal', INDEX)

    expect(matches.length).toBeLessThanOrEqual(2)
  })

  it('one suggestion per server even when several keywords hit', () => {
    expect(matchSuggestions('a linear ticket', INDEX)).toEqual([{ keyword: 'linear', server: 'linear' }])
  })

  it('handles regex metacharacters in keywords safely', () => {
    const index = [{ keywords: ['c++'], server: 'cpp-tools' }]

    expect(matchSuggestions('help with c++ code', index)).toEqual([{ keyword: 'c++', server: 'cpp-tools' }])
  })

  it('matches a pasted vendor URL by host suffix', () => {
    const index = [{ hosts: ['atlassian.net'], keywords: ['jira'], server: 'atlassian' }]

    expect(matchSuggestions('look at https://yourco.atlassian.net/browse/ENG-123 pls', index)).toEqual([
      { keyword: 'atlassian.net', server: 'atlassian' }
    ])
  })

  it('host hit wins over keyword hit as the reported trigger', () => {
    const index = [{ hosts: ['linear.app'], keywords: ['linear'], server: 'linear' }]

    expect(matchSuggestions('linear ticket: https://linear.app/team/issue/ABC-1', index)).toEqual([
      { keyword: 'linear.app', server: 'linear' }
    ])
  })

  it('does not match a host suffix embedded in another domain', () => {
    const index = [{ hosts: ['linear.app'], keywords: [], server: 'linear' }]

    // evil-linear.app.example.com must not fire; nor must notlinear.app.
    expect(matchSuggestions('see https://linear.app.example.com/x', index)).toEqual([])
    expect(matchSuggestions('see https://notlinear.app/x', index)).toEqual([])
  })

  it('strips port and credentials before host comparison', () => {
    const index = [{ hosts: ['sentry.io'], keywords: [], server: 'sentry' }]

    expect(matchSuggestions('logs at https://user@myorg.sentry.io:443/issues', index)).toEqual([
      { keyword: 'sentry.io', server: 'sentry' }
    ])
  })
})
