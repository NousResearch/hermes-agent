import { describe, expect, it } from 'vitest'

import { isHermesSessionCookieName, type JarCookie, moveOauthCookies } from './oauth-cookie-move'
import { LEGACY_OAUTH_PARTITION } from './oauth-partition'

// ForgeGuard fork — carrying a gateway's session cookies between Electron
// partitions (see oauth-partition-moves.ts). In-memory jars stand in for
// session.fromPartition(); the Chromium-side acceptance of the re-set is
// proven on a real Electron build (docs/maintainers/upstream-sync/patch-inventory.md).

interface FakeJar {
  cookies: JarCookie[]
  sets: Record<string, unknown>[]
  removes: [string, string][]
  failSet?: (details: Record<string, unknown>) => boolean
}

function fakeJars(initial: Record<string, JarCookie[]>) {
  const jars = new Map<string, FakeJar>()

  const jar = (partition: string) => {
    let j = jars.get(partition)

    if (!j) {
      j = { cookies: [...(initial[partition] || [])], sets: [], removes: [] }
      jars.set(partition, j)
    }

    return j
  }

  const fromPartition = (partition: string) => {
    const j = jar(partition)

    return {
      cookies: {
        get: async ({ url }: { url?: string }) => {
          const host = url ? new URL(url).hostname : ''

          return j.cookies.filter(c => !host || (c.domain || '').replace(/^\./, '') === host)
        },
        set: async (details: Record<string, unknown>) => {
          if (j.failSet?.(details)) {
            throw new Error('Failed to set cookie')
          }

          j.sets.push(details)
          j.cookies.push({
            name: String(details.name),
            value: String(details.value),
            domain: (details.domain as string) || new URL(String(details.url)).hostname,
            hostOnly: !details.domain,
            path: details.path as string
          })
        },
        remove: async (url: string, name: string) => {
          j.removes.push([url, name])
          j.cookies = j.cookies.filter(c => c.name !== name)
        },
        flushStore: async () => undefined
      }
    }
  }

  return { jar, fromPartition }
}

const URL_A = 'https://edi.example.test'
const CONN_A = `${LEGACY_OAUTH_PARTITION}:conn:edi`
const move = { url: URL_A, from: LEGACY_OAUTH_PARTITION, to: CONN_A }

const cookie = (name: string, extra: Partial<JarCookie> = {}): JarCookie => ({
  name,
  value: `${name}-value`,
  domain: 'edi.example.test',
  hostOnly: true,
  path: '/',
  secure: true,
  httpOnly: true,
  sameSite: 'lax',
  expirationDate: 1_900_000_000,
  ...extra
})

describe('moveOauthCookies', () => {
  it('re-sets a __Host- session cookie without a domain, keeping its attributes, and removes the source', async () => {
    const { jar, fromPartition } = fakeJars({ [LEGACY_OAUTH_PARTITION]: [cookie('__Host-hermes_session_at')] })

    const [result] = await moveOauthCookies([move], { fromPartition })

    expect(result).toMatchObject({ moved: 1, failed: 0 })
    expect(jar(CONN_A).sets).toEqual([
      {
        url: 'https://edi.example.test/',
        name: '__Host-hermes_session_at',
        value: '__Host-hermes_session_at-value',
        path: '/',
        secure: true,
        httpOnly: true,
        sameSite: 'lax',
        expirationDate: 1_900_000_000
      }
    ])
    expect(jar(LEGACY_OAUTH_PARTITION).removes).toEqual([['https://edi.example.test/', '__Host-hermes_session_at']])
  })

  it('writes a plain-HTTP-shaped cookie against the gateway’s https origin, not a scheme guessed from `secure`', async () => {
    // A dashboard behind a TLS terminator in another container writes bare,
    // non-Secure cookies even though the desktop reached it over https.
    const { jar, fromPartition } = fakeJars({
      [LEGACY_OAUTH_PARTITION]: [
        cookie('hermes_session_at', { secure: false }),
        cookie('hermes_session_rt', { secure: false })
      ]
    })

    await moveOauthCookies([move], { fromPartition })

    expect(jar(CONN_A).sets.map(s => [s.url, s.name, s.secure])).toEqual([
      ['https://edi.example.test/', 'hermes_session_at', false],
      ['https://edi.example.test/', 'hermes_session_rt', false]
    ])
  })

  it('keeps the domain of a cookie that was set with one, and the path of a prefixed gateway', async () => {
    const { jar, fromPartition } = fakeJars({
      [LEGACY_OAUTH_PARTITION]: [
        cookie('__Secure-hermes_session_at', { domain: '.edi.example.test', hostOnly: false, path: '/hermes' })
      ]
    })

    await moveOauthCookies([{ ...move, url: `${URL_A}/hermes` }], { fromPartition })

    expect(jar(CONN_A).sets[0]).toMatchObject({
      url: 'https://edi.example.test/hermes',
      domain: '.edi.example.test',
      path: '/hermes'
    })
  })

  it('leaves both jars alone when the destination already holds a live session', async () => {
    const { jar, fromPartition } = fakeJars({
      [LEGACY_OAUTH_PARTITION]: [cookie('hermes_session_at', { value: 'old' })],
      [CONN_A]: [cookie('hermes_session_rt', { value: 'newer' })]
    })

    const lines: string[] = []
    const [result] = await moveOauthCookies([move], { fromPartition, log: l => lines.push(l) })

    expect(result.skipped).toBe('destination-signed-in')
    expect(jar(CONN_A).sets).toEqual([])
    expect(jar(LEGACY_OAUTH_PARTITION).removes).toEqual([])
    expect(lines.join('\n')).toContain('already signed in')
  })

  it('moves only the gateway’s own session cookies', async () => {
    const { jar, fromPartition } = fakeJars({
      [LEGACY_OAUTH_PARTITION]: [
        cookie('hermes_session_at'),
        cookie('hermes_session_provider'),
        cookie('hermes_sso_attempt'),
        cookie('privy-token'),
        cookie('hermes_sessionx')
      ]
    })

    await moveOauthCookies([move], { fromPartition })

    expect(jar(CONN_A).sets.map(s => s.name)).toEqual(['hermes_session_at', 'hermes_session_provider'])
    expect(jar(LEGACY_OAUTH_PARTITION).cookies.map(c => c.name)).toEqual([
      'hermes_sso_attempt',
      'privy-token',
      'hermes_sessionx'
    ])
  })

  it('logs a rejected cookie, keeps its source, and carries on with the rest and the next move', async () => {
    const { jar, fromPartition } = fakeJars({
      [LEGACY_OAUTH_PARTITION]: [cookie('hermes_session_at'), cookie('hermes_session_rt')],
      [`${LEGACY_OAUTH_PARTITION}:conn:b`]: [cookie('hermes_session_at', { domain: 'b.example.test' })]
    })

    jar(CONN_A).failSet = details => details.name === 'hermes_session_at'

    const lines: string[] = []

    const results = await moveOauthCookies(
      [move, { url: 'https://b.example.test', from: `${LEGACY_OAUTH_PARTITION}:conn:b`, to: LEGACY_OAUTH_PARTITION }],
      { fromPartition, log: l => lines.push(l) }
    )

    expect(results.map(r => [r.moved, r.failed])).toEqual([
      [1, 1],
      [1, 0]
    ])
    expect(jar(LEGACY_OAUTH_PARTITION).removes.map(r => r[1])).toEqual(['hermes_session_rt'])
    expect(lines.some(l => l.includes('could not move hermes_session_at'))).toBe(true)
  })

  it('reports rather than throws when a jar is unavailable or there is nothing to move', async () => {
    const { fromPartition } = fakeJars({})

    expect((await moveOauthCookies([move], { fromPartition }))[0].skipped).toBe('nothing-to-move')
    expect((await moveOauthCookies([move], { fromPartition: () => null }))[0].skipped).toBe('jar-unavailable')
    expect((await moveOauthCookies([{ ...move, url: 'not a url' }], { fromPartition }))[0].skipped).toBe('bad-url')
  })
})

describe('isHermesSessionCookieName', () => {
  it.each([
    ['hermes_session_at', true],
    ['__Secure-hermes_session_rt', true],
    ['__Host-hermes_session_provider', true],
    ['hermes_session_pkce', true],
    ['hermes_sso_attempt', false],
    ['hermes_sessionx', false],
    ['__Host-privy-token', false],
    ['', false],
    [undefined, false]
  ])('%s -> %s', (name, expected) => {
    expect(isHermesSessionCookieName(name)).toBe(expected)
  })
})
