import { describe, expect, it } from 'vitest'

import { resolveDeepLinkAction } from './deeplink-routes'

describe('resolveDeepLinkAction', () => {
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

  it('routes legacy plugin-agent alias', () => {
    expect(
      resolveDeepLinkAction({
        kind: 'plugin-agent',
        name: '',
        params: { repo: 'owner/repo' }
      })
    ).toMatchObject({ type: 'plugin-install', legacyHint: 'agent' })
  })

  it('routes a returning connection to its operation, and carries no authority but the op id', () => {
    expect(
      resolveDeepLinkAction({ kind: 'connections', name: 'done', params: { op: 'op-7', status: 'connected' } })
    ).toEqual({ type: 'connection-done', op: 'op-7', status: 'connected' })

    // Without an op there is no operation to show, whatever the status claims.
    expect(resolveDeepLinkAction({ kind: 'connections', name: 'done', params: { status: 'connected' } })).toEqual({
      type: 'ignore'
    })
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
