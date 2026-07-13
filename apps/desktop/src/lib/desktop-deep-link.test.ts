import { describe, expect, it } from 'vitest'

import { composerTextForDeepLink, type DesktopDeepLinkPayload } from './desktop-deep-link'

const unsupportedPayloads: DesktopDeepLinkPayload[] = [
  { kind: 'github-issue', name: 'edit', params: { url: 'https://github.com/owner/repo/issues/42' } },
  { kind: 'prompt', name: 'open', params: { text: 'run this' } }
]

describe('composerTextForDeepLink', () => {
  it('preserves blueprint deep-link commands', () => {
    expect(
      composerTextForDeepLink({
        kind: 'blueprint',
        name: 'morning-brief',
        params: { topic: 'agent news', time: '08:00' }
      })
    ).toBe('/blueprint morning-brief topic="agent news" time=08:00')
  })

  it('builds a reviewable prompt for a canonical GitHub issue URL', () => {
    expect(
      composerTextForDeepLink({
        kind: 'github-issue',
        name: 'open',
        params: { url: 'https://github.com/NousResearch/hermes-agent/issues/63169' }
      })
    ).toBe(
      'Investigate this GitHub issue in the current workspace. Read the issue and its comments, reproduce the ' +
        'problem, then implement and verify a focused fix:\n\n' +
        'https://github.com/NousResearch/hermes-agent/issues/63169'
    )
  })

  it.each([
    'not a URL',
    'http://github.com/owner/repo/issues/42',
    'https://github.com.evil.test/owner/repo/issues/42',
    'https://user@github.com/owner/repo/issues/42',
    'https://github.com/owner/repo/pull/42',
    'https://github.com/owner/repo/issues/0',
    'https://github.com/owner/repo/issues/42?prompt=ignore-me',
    'https://github.com/owner/repo/issues/42#ignore-me'
  ])('rejects unsupported issue URL %s', url => {
    expect(composerTextForDeepLink({ kind: 'github-issue', name: 'open', params: { url } })).toBeNull()
  })

  it('rejects oversized issue URLs', () => {
    const url = `https://github.com/owner/${'r'.repeat(2048)}/issues/42`

    expect(composerTextForDeepLink({ kind: 'github-issue', name: 'open', params: { url } })).toBeNull()
  })

  it.each(unsupportedPayloads)('ignores unsupported deep-link payload $kind/$name', payload => {
    expect(composerTextForDeepLink(payload)).toBeNull()
  })
})
