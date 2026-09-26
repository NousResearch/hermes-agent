import { describe, expect, it } from 'vitest'

import { originKeyFor, parseSetCookie, RemoteSessionCookieStore } from './remote-session-cookies'

describe('parseSetCookie', () => {
  it('takes the first Name=Value pair and ignores attributes', () => {
    expect(parseSetCookie('hermes_session=abc123; Path=/; HttpOnly; SameSite=Lax')).toEqual({
      name: 'hermes_session',
      value: 'abc123'
    })
  })

  it('returns null for unparsable headers', () => {
    expect(parseSetCookie('')).toBeNull()
    expect(parseSetCookie('nonsense')).toBeNull()
    expect(parseSetCookie('=value')).toBeNull()
    expect(parseSetCookie('name=')).toBeNull()
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

describe('RemoteSessionCookieStore', () => {
  it('records Set-Cookie headers and serializes a Cookie header per origin', () => {
    const store = new RemoteSessionCookieStore()

    store.record('https://gw.example.com/api/auth/login', 'hermes_session=abc; Path=/; HttpOnly')
    store.record('https://gw.example.com/api/other', ['refresh=xyz; Path=/', 'broken'])

    expect(store.cookieHeaderFor('https://gw.example.com/api/auth/ws-ticket')).toBe('hermes_session=abc; refresh=xyz')
    // Origin-scoped: another gateway sees nothing.
    expect(store.cookieHeaderFor('https://other.example.com/api')).toBeNull()
  })

  it('overwrites a cookie when the gateway rotates it', () => {
    const store = new RemoteSessionCookieStore()

    store.record('https://gw.example.com', 'hermes_session=old; Path=/')
    store.record('https://gw.example.com', 'hermes_session=new; Path=/')

    expect(store.cookieHeaderFor('https://gw.example.com/')).toBe('hermes_session=new')
  })

  it('seeds from a session jar read', () => {
    const store = new RemoteSessionCookieStore()

    store.recordFromJar('https://gw.example.com', [
      { name: 'hermes_session', value: 'jar-value' },
      { name: '', value: 'x' },
      { value: 'no-name' }
    ])

    expect(store.cookieHeaderFor('https://gw.example.com/x')).toBe('hermes_session=jar-value')
  })

  it('clear drops only the target origin (or everything)', () => {
    const store = new RemoteSessionCookieStore()

    store.record('https://gw.example.com', 'a=1; Path=/')
    store.record('https://other.example.com', 'b=2; Path=/')

    store.clear('https://gw.example.com/api/status')
    expect(store.cookieHeaderFor('https://gw.example.com/')).toBeNull()
    expect(store.cookieHeaderFor('https://other.example.com/')).toBe('b=2')

    store.clear()
    expect(store.cookieHeaderFor('https://other.example.com/')).toBeNull()
  })
})
