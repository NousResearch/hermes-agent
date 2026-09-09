import { describe, expect, it } from 'vitest'

import { LEGACY_OAUTH_PARTITION, resolveOauthPartition } from './oauth-partition'

// #92183 — two basic-auth (cookie-flow) gateways registered in the v2
// connections registry must not share one cookie jar. Chromium cookie jars
// ignore the port, so two gateways on the same VPN host (different ports)
// evict each other's `hermes_session*` cookies when they ride the single
// shared `persist:hermes-remote-oauth` partition — and, worse, gateway A's
// cookie is silently PRESENTED to gateway B on every request. The resolver
// under test keys the jar on the registry connection's identity instead.

const registry = (primary: string, connections: any[]) => ({ primary, connections })

const remote = (id: string, url: string, extra: Record<string, unknown> = {}) => ({
  id,
  kind: 'remote',
  label: id,
  url,
  authMode: 'oauth',
  ...extra
})

describe('resolveOauthPartition (#92183 per-connection cookie jars)', () => {
  it('gives two same-host different-port registry gateways DISTINCT partitions (eviction fix)', () => {
    const reg = registry('local', [
      { id: 'local', kind: 'local' },
      remote('conn-a', 'https://10.27.27.7:9119'),
      remote('conn-b', 'https://10.27.27.7:9220')
    ])

    const a = resolveOauthPartition('https://10.27.27.7:9119', { registry: reg })
    const b = resolveOauthPartition('https://10.27.27.7:9220', { registry: reg })

    expect(a).not.toBe(LEGACY_OAUTH_PARTITION)
    expect(b).not.toBe(LEGACY_OAUTH_PARTITION)
    // Fail closed: B's requests must never ride a jar that can hold A's cookie.
    expect(a).not.toBe(b)
  })

  it('scopes a full request URL (REST/ws-ticket path) to its connection jar via longest base-url prefix', () => {
    const reg = registry('local', [
      remote('conn-a', 'https://gw.example.com'),
      remote('conn-b', 'https://gw.example.com/team-b')
    ])

    const a = resolveOauthPartition('https://gw.example.com/api/auth/ws-ticket', { registry: reg })
    const b = resolveOauthPartition('https://gw.example.com/team-b/api/auth/ws-ticket', { registry: reg })

    expect(a).not.toBe(b)
    expect(b).toContain('conn-b')
  })

  it('keeps the v1 primary remote on the LEGACY partition so upgrades do not sign the user out', () => {
    const reg = registry('mig-1', [remote('mig-1', 'https://gw-a.example.com')])

    expect(
      resolveOauthPartition('https://gw-a.example.com', {
        registry: reg,
        v1RemoteUrl: 'https://gw-a.example.com'
      })
    ).toBe(LEGACY_OAUTH_PARTITION)
  })

  it('keeps the registry PRIMARY connection on the legacy partition', () => {
    const reg = registry('conn-a', [
      remote('conn-a', 'https://gw-a.example.com'),
      remote('conn-b', 'https://gw-b.example.com')
    ])

    expect(resolveOauthPartition('https://gw-a.example.com/api/status', { registry: reg })).toBe(LEGACY_OAUTH_PARTITION)
    expect(resolveOauthPartition('https://gw-b.example.com/api/status', { registry: reg })).not.toBe(
      LEGACY_OAUTH_PARTITION
    )
  })

  it('keeps cloud connections on the legacy partition (silent portal cascade needs the shared jar)', () => {
    const reg = registry('local', [
      { id: 'cloud-1', kind: 'cloud', url: 'https://agent.nousresearch.com', authMode: 'oauth' }
    ])

    expect(resolveOauthPartition('https://agent.nousresearch.com/api/status', { registry: reg })).toBe(
      LEGACY_OAUTH_PARTITION
    )
  })

  it('keeps token-auth registry remotes on the legacy partition (no cookies involved)', () => {
    const reg = registry('local', [remote('tok-1', 'https://gw-t.example.com', { authMode: 'token' })])

    expect(resolveOauthPartition('https://gw-t.example.com', { registry: reg })).toBe(LEGACY_OAUTH_PARTITION)
  })

  it('falls back to the legacy partition for unmatched, portal, and malformed inputs', () => {
    const reg = registry('local', [remote('conn-a', 'https://gw-a.example.com')])

    expect(resolveOauthPartition('https://portal.nousresearch.com/api/agents', { registry: reg })).toBe(
      LEGACY_OAUTH_PARTITION
    )
    expect(resolveOauthPartition('not a url', { registry: reg })).toBe(LEGACY_OAUTH_PARTITION)
    expect(resolveOauthPartition('', { registry: reg })).toBe(LEGACY_OAUTH_PARTITION)
    expect(resolveOauthPartition('https://gw-a.example.com', { registry: null as any })).toBe(LEGACY_OAUTH_PARTITION)
    expect(
      resolveOauthPartition('https://gw-a.example.com', { registry: { primary: 'x', connections: 'junk' } as any })
    ).toBe(LEGACY_OAUTH_PARTITION)
  })

  it('does not treat a hostname PREFIX as a base-url match', () => {
    const reg = registry('local', [remote('conn-a', 'https://gw.example.com')])

    expect(resolveOauthPartition('https://gw.example.com.evil.tld/login', { registry: reg })).toBe(
      LEGACY_OAUTH_PARTITION
    )
  })

  it('normalizes trailing slashes and default ports when matching entry URLs', () => {
    const reg = registry('local', [remote('conn-a', 'https://gw-a.example.com:443/')])

    const got = resolveOauthPartition('https://gw-a.example.com/api/auth/ws-ticket', { registry: reg })

    expect(got).not.toBe(LEGACY_OAUTH_PARTITION)
    expect(got).toContain('conn-a')
  })

  it('produces a deterministic, partition-safe name from hostile connection ids', () => {
    const reg = registry('local', [remote('we ird/id:€', 'https://gw-a.example.com')])

    const got = resolveOauthPartition('https://gw-a.example.com', { registry: reg })

    expect(got.startsWith('persist:')).toBe(true)
    expect(got).not.toMatch(/[\s/€]/)
    expect(resolveOauthPartition('https://gw-a.example.com', { registry: reg })).toBe(got)
  })

  it('breaks same-URL ties deterministically (identical jar for identical gateway)', () => {
    const reg = registry('local', [
      remote('zeta', 'https://gw-a.example.com'),
      remote('alpha', 'https://gw-a.example.com')
    ])

    const got = resolveOauthPartition('https://gw-a.example.com', { registry: reg })

    expect(got).toContain('alpha')
  })
})

// The registry editor can sign a draft in BEFORE it is saved. The login window
// must then write to the jar the saved entry will read — an unsaved draft has
// no on-disk entry to URL-match, so identity (connectionId) decides instead.
describe('resolveOauthPartition with connectionId (pre-save sign-in identity)', () => {
  it('sends a pending (unsaved) draft sign-in to its own jar, not the legacy shared one', () => {
    const reg = registry('local', [{ id: 'local', kind: 'local' }])

    const got = resolveOauthPartition('https://macmini.lan:9119', { registry: reg, connectionId: 'macmini' })

    expect(got).not.toBe(LEGACY_OAUTH_PARTITION)
    expect(got).toContain('conn:macmini')
    // Deterministic across calls — the saved entry must read the jar the login wrote.
    expect(resolveOauthPartition('https://macmini.lan:9119', { registry: reg, connectionId: 'macmini' })).toBe(got)
  })

  it('keeps a pending same-host sign-in out of an existing gateway’s jar (no #92183 eviction)', () => {
    const reg = registry('local', [{ id: 'local', kind: 'local' }, remote('conn-a', 'https://10.27.27.7:9119')])

    const pending = resolveOauthPartition('https://10.27.27.7:9220', { registry: reg, connectionId: 'conn-b' })
    const existing = resolveOauthPartition('https://10.27.27.7:9119', { registry: reg })

    expect(pending).not.toBe(LEGACY_OAUTH_PARTITION)
    expect(pending).not.toBe(existing)
    expect(existing).toContain('conn-a')
    expect(pending).toContain('conn-b')
  })

  it('resolves an existing entry by identity even when the URL was edited but not yet saved', () => {
    const reg = registry('local', [{ id: 'local', kind: 'local' }, remote('conn-a', 'https://old.example.com')])

    const got = resolveOauthPartition('https://new.example.com', { registry: reg, connectionId: 'conn-a' })

    expect(got).toContain('conn-a')
    expect(got).not.toBe(LEGACY_OAUTH_PARTITION)
  })

  it('keeps the primary, the local entry, cloud, and token-auth identities on the legacy jar', () => {
    const reg = registry('conn-a', [
      { id: 'local', kind: 'local' },
      remote('conn-a', 'https://gw-a.example.com'),
      remote('tok-1', 'https://gw-t.example.com', { authMode: 'token' }),
      { id: 'cloud-1', kind: 'cloud', url: 'https://agent.nousresearch.com', authMode: 'oauth' }
    ])

    expect(resolveOauthPartition('https://gw-a.example.com', { registry: reg, connectionId: 'conn-a' })).toBe(
      LEGACY_OAUTH_PARTITION
    )
    expect(resolveOauthPartition('https://gw-a.example.com', { registry: reg, connectionId: 'local' })).toBe(
      LEGACY_OAUTH_PARTITION
    )
    expect(resolveOauthPartition('https://gw-t.example.com', { registry: reg, connectionId: 'tok-1' })).toBe(
      LEGACY_OAUTH_PARTITION
    )
    expect(resolveOauthPartition('https://agent.nousresearch.com', { registry: reg, connectionId: 'cloud-1' })).toBe(
      LEGACY_OAUTH_PARTITION
    )
  })

  it('keeps a v1-migrated entry on the legacy jar even when named by id', () => {
    const reg = registry('local', [{ id: 'local', kind: 'local' }, remote('mig-1', 'https://gw-a.example.com')])

    expect(
      resolveOauthPartition('https://gw-a.example.com', {
        registry: reg,
        connectionId: 'mig-1',
        v1RemoteUrl: 'https://gw-a.example.com'
      })
    ).toBe(LEGACY_OAUTH_PARTITION)
  })

  it('ignores blank or non-string connectionIds and falls back to URL matching', () => {
    const reg = registry('local', [remote('conn-a', 'https://gw-a.example.com')])

    for (const junk of ['', '   ', 42, null, undefined]) {
      expect(resolveOauthPartition('https://gw-a.example.com', { registry: reg, connectionId: junk })).toContain(
        'conn-a'
      )
    }
  })

  it('sanitizes hostile pending ids into partition-safe names', () => {
    const reg = registry('local', [{ id: 'local', kind: 'local' }])

    const got = resolveOauthPartition('https://gw.example.com', { registry: reg, connectionId: 'we ird/id:€' })

    expect(got.startsWith('persist:')).toBe(true)
    expect(got).not.toMatch(/[\s/€]/)
  })
})
