import { describe, expect, it } from 'vitest'

import { LEGACY_OAUTH_PARTITION } from './oauth-partition'
import { planOauthPartitionMoves } from './oauth-partition-moves'

// ForgeGuard fork — a cookie-flow gateway's jar is decided from the registry's
// current shape (#92183), so it changes when the gateway's status changes.
// These cases pin the diff main.ts uses to carry the session across.

const registry = (primary: string, connections: any[]) => ({ primary, connections })

const remote = (id: string, url: string, extra: Record<string, unknown> = {}) => ({
  id,
  kind: 'remote',
  label: id,
  url,
  authMode: 'oauth',
  ...extra
})

const conn = (id: string) => `${LEGACY_OAUTH_PARTITION}:conn:${id}`

describe('planOauthPartitionMoves', () => {
  it('moves a session signed in before Save from the legacy jar to the new entry’s own jar', () => {
    const local = { id: 'local', kind: 'local', label: 'This Mac' }
    const before = { registry: registry('local', [local]) }
    const after = { registry: registry('local', [local, remote('edi', 'https://edi.example.test')]) }

    expect(planOauthPartitionMoves(before, after)).toEqual([
      { url: 'https://edi.example.test', from: LEGACY_OAUTH_PARTITION, to: conn('edi') }
    ])
  })

  it('make-primary moves both gateways, the one leaving the legacy jar first', () => {
    const a = remote('a', 'https://a.example.test')
    const b = remote('b', 'https://b.example.test')

    const moves = planOauthPartitionMoves({ registry: registry('a', [a, b]) }, { registry: registry('b', [a, b]) })

    expect(moves).toEqual([
      { url: 'https://a.example.test', from: LEGACY_OAUTH_PARTITION, to: conn('a') },
      { url: 'https://b.example.test', from: conn('b'), to: LEGACY_OAUTH_PARTITION }
    ])
  })

  it('plans nothing for a label-only edit', () => {
    const before = { registry: registry('local', [remote('a', 'https://a.example.test')]) }
    const after = { registry: registry('local', [remote('a', 'https://a.example.test', { label: 'renamed' })]) }

    expect(planOauthPartitionMoves(before, after)).toEqual([])
  })

  it('never moves token-auth or cloud entries — they have no gateway session cookies', () => {
    const before = { registry: registry('local', []) }

    const after = {
      registry: registry('local', [
        remote('tok', 'https://tok.example.test', { authMode: 'token' }),
        remote('cloud', 'https://cloud.example.test', { kind: 'cloud' })
      ])
    }

    expect(planOauthPartitionMoves(before, after)).toEqual([])
  })

  it('treats an apply that changes the v1 remote like a primary flip', () => {
    const a = remote('a', 'https://a.example.test')
    const b = remote('b', 'https://b.example.test')

    const before = { registry: registry('a', [a, b]), v1RemoteUrl: 'https://a.example.test' }
    const after = { registry: registry('b', [a, b]), v1RemoteUrl: 'https://b.example.test' }

    expect(planOauthPartitionMoves(before, after).map(m => [m.url, m.from, m.to])).toEqual([
      ['https://a.example.test', LEGACY_OAUTH_PARTITION, conn('a')],
      ['https://b.example.test', conn('b'), LEGACY_OAUTH_PARTITION]
    ])
  })

  it('leaves an entry that is the v1 remote on the legacy jar when it is saved', () => {
    const before = { registry: registry('local', []), v1RemoteUrl: 'https://a.example.test' }

    const after = {
      registry: registry('local', [remote('a', 'https://a.example.test')]),
      v1RemoteUrl: 'https://a.example.test'
    }

    expect(planOauthPartitionMoves(before, after)).toEqual([])
  })

  it('never carries a session that leaves the registry into the shared jar', () => {
    const a = remote('a', 'https://a.example.test')
    const primary = remote('p', 'https://a.example.test:8443')

    // Removed: its URL now resolves to the legacy jar, which the same-host
    // primary rides — moving the session there would present it to that
    // gateway (#92183). Its own jar is left behind instead.
    expect(
      planOauthPartitionMoves({ registry: registry('p', [primary, a]) }, { registry: registry('p', [primary]) })
    ).toEqual([])

    // URL edited: the same holds for the URL it used to have. The NEW URL is
    // a sign-in-before-save like any other and does move into the entry's jar.
    const edited = remote('a', 'https://a2.example.test')

    expect(
      planOauthPartitionMoves({ registry: registry('p', [primary, a]) }, { registry: registry('p', [primary, edited]) })
    ).toEqual([{ url: 'https://a2.example.test', from: LEGACY_OAUTH_PARTITION, to: conn('a') }])
  })

  it('moves a session when an entry switches from token to cookie auth', () => {
    const before = { registry: registry('local', [remote('a', 'https://a.example.test', { authMode: 'token' })]) }
    const after = { registry: registry('local', [remote('a', 'https://a.example.test')]) }

    expect(planOauthPartitionMoves(before, after)).toEqual([
      { url: 'https://a.example.test', from: LEGACY_OAUTH_PARTITION, to: conn('a') }
    ])
  })

  it('never throws on malformed snapshots', () => {
    expect(planOauthPartitionMoves(null, undefined)).toEqual([])
    expect(planOauthPartitionMoves({ registry: { connections: 'nope' } as any }, { registry: null })).toEqual([])
    expect(
      planOauthPartitionMoves(
        { registry: registry('x', [null, 7, { kind: 'remote', authMode: 'oauth', url: 'ftp://x' }]) },
        { registry: registry('x', [remote('', 'https://no-id.example.test')]) }
      )
    ).toEqual([])
  })
})
