import { describe, expect, it } from 'vitest'

import {
  cookiePathMatches,
  jarWriteFromSetCookie,
  originKeyFor,
  parseSetCookie,
  RemoteSessionCookieStore
} from './remote-session-cookies'

const LEGACY = 'persist:hermes-remote-oauth'
const CONN_A = 'persist:hermes-remote-oauth-conn-a'
const CONN_B = 'persist:hermes-remote-oauth-conn-b'

describe('parseSetCookie', () => {
  it('takes the first Name=Value pair and keeps the effective path', () => {
    expect(parseSetCookie('hermes_session=abc123; Path=/; HttpOnly; SameSite=Lax')).toEqual({
      name: 'hermes_session',
      value: 'abc123',
      path: '/',
      expired: false,
      attributes: { secure: false, httpOnly: true, sameSite: 'lax' }
    })
    expect(parseSetCookie('hermes_session=abc; Path=/gw-a; HttpOnly')?.path).toBe('/gw-a')
    // A missing or non-absolute Path falls back to "/".
    expect(parseSetCookie('a=1')?.path).toBe('/')
    expect(parseSetCookie('a=1; Path=relative')?.path).toBe('/')
  })

  it('flags deletions: Max-Age<=0, a past Expires, or an empty value', () => {
    expect(parseSetCookie('hermes_session=; Path=/; Max-Age=0')?.expired).toBe(true)
    expect(parseSetCookie('hermes_session=x; Max-Age=-1')?.expired).toBe(true)
    expect(parseSetCookie('hermes_session=x; Expires=Thu, 01 Jan 1970 00:00:00 GMT')?.expired).toBe(true)
    expect(parseSetCookie('hermes_session=x; Max-Age=900')?.expired).toBe(false)
    expect(parseSetCookie('hermes_session=; Path=/')?.expired).toBe(true)
  })

  it('returns null for unparsable headers', () => {
    expect(parseSetCookie('')).toBeNull()
    expect(parseSetCookie('nonsense')).toBeNull()
    expect(parseSetCookie('=value')).toBeNull()
  })
})

describe('originKeyFor', () => {
  it('keys on protocol+host, ignoring path and port-implicit forms', () => {
    expect(originKeyFor('https://gw.example.com/api/status')).toBe('https://gw.example.com')
    expect(originKeyFor('http://localhost:8734/ws-ticket')).toBe('http://localhost:8734')
    expect(originKeyFor('ftp://gw.example.com')).toBeNull()
    expect(originKeyFor('not a url')).toBeNull()
  })
})

describe('cookiePathMatches', () => {
  it('implements RFC 6265 path-match', () => {
    expect(cookiePathMatches('/', '/api/status')).toBe(true)
    expect(cookiePathMatches('/gw-a', '/gw-a')).toBe(true)
    expect(cookiePathMatches('/gw-a', '/gw-a/api')).toBe(true)
    expect(cookiePathMatches('/gw-a/', '/gw-a/api')).toBe(true)
    expect(cookiePathMatches('/gw-a', '/gw-ab/api')).toBe(false)
    expect(cookiePathMatches('/gw-a', '/gw-b/api')).toBe(false)
  })
})

describe('RemoteSessionCookieStore', () => {
  it('records Set-Cookie headers and serializes a Cookie header per partition+origin', () => {
    const store = new RemoteSessionCookieStore()

    store.record(LEGACY, 'https://gw.example.com/api/auth/login', 'hermes_session=abc; Path=/; HttpOnly')
    store.record(LEGACY, 'https://gw.example.com/api/other', ['refresh=xyz; Path=/', 'broken'])

    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/api/auth/ws-ticket')).toBe(
      'hermes_session=abc; refresh=xyz'
    )
    // Origin-scoped: another gateway sees nothing.
    expect(store.cookieHeaderFor(LEGACY, 'https://other.example.com/api')).toBeNull()
    // Partition-scoped: the same origin on a different jar sees nothing.
    expect(store.cookieHeaderFor(CONN_A, 'https://gw.example.com/api/auth/ws-ticket')).toBeNull()
  })

  it('keeps same-origin sub-path gateways on separate partitions apart (A→B→A)', () => {
    const store = new RemoteSessionCookieStore()
    const a = 'https://gw.example.com/gw-a/api/auth/ws-ticket'
    const b = 'https://gw.example.com/gw-b/api/auth/ws-ticket'

    store.record(CONN_A, a, 'hermes_session=session-a; Path=/')
    store.record(CONN_B, b, 'hermes_session=session-b; Path=/')
    // Back to A: B's later credential must not have replaced A's.
    expect(store.cookieHeaderFor(CONN_A, a)).toBe('hermes_session=session-a')
    expect(store.cookieHeaderFor(CONN_B, b)).toBe('hermes_session=session-b')

    // A B-only 401 clears only B.
    store.clear(CONN_B, b)
    expect(store.cookieHeaderFor(CONN_B, b)).toBeNull()
    expect(store.cookieHeaderFor(CONN_A, a)).toBe('hermes_session=session-a')
  })

  it('honours cookie Path scope within one partition+origin', () => {
    const store = new RemoteSessionCookieStore()

    store.record(LEGACY, 'https://gw.example.com/gw-a/', 'hermes_session=scoped; Path=/gw-a')
    store.record(LEGACY, 'https://gw.example.com/', 'shared=1; Path=/')

    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/gw-a/api/status')).toBe(
      'hermes_session=scoped; shared=1'
    )
    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/gw-b/api/status')).toBe('shared=1')
  })

  it('overwrites a cookie when the gateway rotates it and deletes it on an expiring Set-Cookie', () => {
    const store = new RemoteSessionCookieStore()

    store.record(LEGACY, 'https://gw.example.com', 'hermes_session=old; Path=/')
    store.record(LEGACY, 'https://gw.example.com', 'hermes_session=new; Path=/')
    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/')).toBe('hermes_session=new')

    store.record(LEGACY, 'https://gw.example.com', 'hermes_session=; Path=/; Max-Age=0')
    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/')).toBeNull()
  })

  it('seeds from a session jar read, keeping the jar path', () => {
    const store = new RemoteSessionCookieStore()

    store.recordFromJar(LEGACY, 'https://gw.example.com', [
      { name: 'hermes_session', value: 'jar-value', path: '/' },
      { name: 'scoped', value: 's', path: '/gw-a' },
      { name: '', value: 'x' },
      { value: 'no-name' }
    ])

    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/x')).toBe('hermes_session=jar-value')
    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/gw-a/x')).toBe('hermes_session=jar-value; scoped=s')
  })

  it('clear drops the target partition+origin, a whole partition, or everything', () => {
    const store = new RemoteSessionCookieStore()

    store.record(LEGACY, 'https://gw.example.com', 'a=1; Path=/')
    store.record(LEGACY, 'https://other.example.com', 'b=2; Path=/')
    store.record(CONN_A, 'https://gw.example.com', 'c=3; Path=/')

    store.clear(LEGACY, 'https://gw.example.com/api/status')
    expect(store.cookieHeaderFor(LEGACY, 'https://gw.example.com/')).toBeNull()
    expect(store.cookieHeaderFor(LEGACY, 'https://other.example.com/')).toBe('b=2')
    expect(store.cookieHeaderFor(CONN_A, 'https://gw.example.com/')).toBe('c=3')

    store.clear(LEGACY)
    expect(store.cookieHeaderFor(LEGACY, 'https://other.example.com/')).toBeNull()
    expect(store.cookieHeaderFor(CONN_A, 'https://gw.example.com/')).toBe('c=3')

    store.clear()
    expect(store.cookieHeaderFor(CONN_A, 'https://gw.example.com/')).toBeNull()
  })
})

describe('RemoteSessionCookieStore.followJarWrite', () => {
  const AGENT = 'https://agent.agents.example.com'
  const TICKET = `${AGENT}/api/auth/ws-ticket`

  it('a refresh-token rotation in the jar replaces the mirrored token, so the mirror never replays it', () => {
    const store = new RemoteSessionCookieStore()

    store.record(LEGACY, TICKET, '__Host-hermes_session_rt=rt-1; Path=/; HttpOnly; Secure')
    store.followJarWrite(LEGACY, {
      name: '__Host-hermes_session_rt',
      value: 'rt-2',
      domain: 'agent.agents.example.com',
      path: '/',
      hostOnly: true
    })

    expect(store.cookieHeaderFor(LEGACY, TICKET)).toBe('__Host-hermes_session_rt=rt-2')
  })

  it('never adds a cookie the mirror did not hold, nor crosses partition, host or path', () => {
    const store = new RemoteSessionCookieStore()
    const other = 'https://other.agents.example.com/api'

    store.record(LEGACY, TICKET, 'rt=old; Path=/')
    store.record(CONN_A, TICKET, 'rt=conn-a; Path=/')
    store.record(LEGACY, other, 'rt=other; Path=/')

    store.followJarWrite(LEGACY, { name: 'fresh', value: 'x', domain: 'agent.agents.example.com', path: '/' })
    store.followJarWrite(LEGACY, { name: 'rt', value: 'wrong-path', domain: 'agent.agents.example.com', path: '/gw' })
    store.followJarWrite(LEGACY, {
      name: 'rt',
      value: 'new',
      domain: 'agent.agents.example.com',
      path: '/',
      hostOnly: true
    })

    expect(store.cookieHeaderFor(LEGACY, TICKET)).toBe('rt=new')
    expect(store.cookieHeaderFor(CONN_A, TICKET)).toBe('rt=conn-a')
    expect(store.cookieHeaderFor(LEGACY, other)).toBe('rt=other')
  })

  it('a domain cookie reaches subdomain origins; a host-only one does not', () => {
    const store = new RemoteSessionCookieStore()

    store.record(LEGACY, TICKET, 'rt=old; Path=/')
    store.followJarWrite(LEGACY, {
      name: 'rt',
      value: 'host-only',
      domain: 'agents.example.com',
      path: '/',
      hostOnly: true
    })
    expect(store.cookieHeaderFor(LEGACY, TICKET)).toBe('rt=old')

    store.followJarWrite(LEGACY, { name: 'rt', value: 'domain', domain: '.agents.example.com', path: '/' })
    expect(store.cookieHeaderFor(LEGACY, TICKET)).toBe('rt=domain')
  })
})

describe('jarWriteFromSetCookie', () => {
  const URL_ = 'https://agent.agents.example.com/api/auth/ws-ticket'

  it("turns the gateway's rotated refresh cookie into jar details", () => {
    const now = Date.UTC(2026, 10, 12)

    expect(
      jarWriteFromSetCookie(
        URL_,
        '__Host-hermes_session_rt=rt-2; HttpOnly; Max-Age=2592000; Path=/; SameSite=lax; Secure',
        now
      )
    ).toEqual({
      kind: 'set',
      details: {
        url: URL_,
        name: '__Host-hermes_session_rt',
        value: 'rt-2',
        path: '/',
        secure: true,
        httpOnly: true,
        sameSite: 'lax',
        expirationDate: now / 1000 + 2592000
      }
    })
  })

  it('keeps Domain, honours Expires, and leaves a cookie without either a session cookie', () => {
    const set = jarWriteFromSetCookie(URL_, 'a=1; Domain=.example.com; Expires=Wed, 01 Jan 2031 00:00:00 GMT')

    expect(set?.kind === 'set' && set.details.domain).toBe('.example.com')
    expect(set?.kind === 'set' && set.details.expirationDate).toBe(Date.UTC(2031, 0, 1) / 1000)

    const session = jarWriteFromSetCookie(URL_, 'a=1; Path=/')

    expect(session?.kind === 'set' && session.details.expirationDate).toBeUndefined()
  })

  it('a deletion removes the jar cookie; junk is ignored', () => {
    expect(jarWriteFromSetCookie(URL_, '__Host-hermes_session_rt=; Max-Age=0; Path=/')).toEqual({
      kind: 'remove',
      name: '__Host-hermes_session_rt'
    })
    expect(jarWriteFromSetCookie(URL_, 'nonsense')).toBeNull()
  })
})
