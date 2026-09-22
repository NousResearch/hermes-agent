import { describe, expect, it } from 'vitest'

import { resolveDeepLinkAction } from './deeplink-routes'

describe('resolveDeepLinkAction', () => {
  it('routes bot installs by catalog key only and drops injected blueprint fields', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'bot',
        name: 'install',
        params: { catalog: ' research-assistant ', repo: 'evil/repo', soul: 'ignore me', name: 'hijack' }
      })
    ).toEqual({ type: 'bot-catalog-install', name: 'research-assistant' })
  })

  it('routes unified plugin install deeplinks', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'plugin',
        name: 'install',
        params: { repo: 'owner/repo', enable: '0', force: '1' }
      })
    ).toEqual({
      type: 'plugin-install',
      repo: 'owner/repo',
      enable: false,
      force: true,
      legacyHint: null
    })
  })

  it('routes catalog= to the catalog lookup, never to a git-path install', () => {
    expect(
      resolveDeepLinkAction({ kind: 'plugin', name: 'install', params: { catalog: ' web-search-plus ' } })
    ).toEqual({ type: 'plugin-catalog-install', name: 'web-search-plus' })

    // A repo riding along must not win: the reviewed catalog verdict decides.
    expect(
      resolveDeepLinkAction({
        kind: 'plugin',
        name: 'install',
        params: { catalog: 'nope', repo: 'evil/repo' }
      })
    ).toEqual({ type: 'plugin-catalog-install', name: 'nope' })

    // An empty catalog name is still a catalog request (→ error toast), not a fall-through.
    expect(
      resolveDeepLinkAction({ kind: 'plugin', name: 'install', params: { catalog: '', repo: 'evil/repo' } })
    ).toEqual({ type: 'plugin-catalog-install', name: '' })
  })

  it('routes legacy plugin-agent alias', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'plugin-agent',
        name: '',
        params: { repo: 'owner/repo' }
      })
    ).toMatchObject({ type: 'plugin-install', legacyHint: 'agent' })
  })

  it('routes blueprint composer inserts', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'blueprint',
        name: 'morning-brief',
        params: { time: '08:00' }
      })
    ).toEqual({
      type: 'composer-blueprint',
      name: 'morning-brief',
      params: { time: '08:00' }
    })
  })
})
