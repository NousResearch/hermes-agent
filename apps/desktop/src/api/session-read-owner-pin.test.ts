import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionMessagesResponse } from '@/types/hermes'

import {
  sessionReadOwnerPin,
  setApiRequestConnection,
  setApiRequestLocalMode,
  setApiRequestProfile,
  setSessionOwnerResolver
} from './client'
import { getSessionMessages } from './sessions'

// #160309: transcript hydration for a session owned by a NON-default profile
// could issue its reads with no `profile` parameter at all. The ambient scope
// starts null, and `sessionReadOwnerPin` dropped a KNOWN owner's profile when
// the owner rode the ambient connection (or named no connection) — the read
// then hit the serving process's own default store, missed the id, and 404'd
// a session the desktop can see on screen. These tests pin the contract that
// the owner's profile rides along on EVERY owner shape.

const page = (sessionId: string): SessionMessagesResponse => ({
  session_id: sessionId,
  profile: 'work',
  messages: [],
  pagination: { limit: 120, offset: 0, order: 'latest', returned: 0 }
})

describe('sessionReadOwnerPin owner profile retention', () => {
  beforeEach(() => {
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    setApiRequestLocalMode(true)
  })

  afterEach(() => {
    setSessionOwnerResolver(null)
    setApiRequestLocalMode(false)
    setApiRequestConnection(null)
    setApiRequestProfile(null)
  })

  it('keeps a known owner profile on the ambient connection (the #160309 404)', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'work', targetProfile: 'work' }))

    // Ambient connection is 'local' (local mode, no registry source): the owner
    // IS on the ambient connection, and the old code returned {} here — the
    // request carried no profile and the serving home 404'd the id.
    expect(sessionReadOwnerPin('stored-session')).toEqual({ profile: 'work' })
  })

  it('keeps a known owner profile when the owner names no connection', () => {
    setSessionOwnerResolver(() => ({ profile: 'work' }))

    expect(sessionReadOwnerPin('stored-session')).toEqual({ profile: 'work' })
  })

  it('resolves a string owner to its profile', () => {
    setSessionOwnerResolver(() => 'work')

    expect(sessionReadOwnerPin('stored-session')).toEqual({ profile: 'work' })
  })

  it('still pins the owner connection when it differs from the ambient one', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'hermes-pi', profile: 'work', targetProfile: 'work' }))

    expect(sessionReadOwnerPin('stored-session')).toEqual({ connectionId: 'hermes-pi', profile: 'work' })
  })

  it('prefers targetProfile over profile on the owner route', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'default', targetProfile: 'work' }))

    expect(sessionReadOwnerPin('stored-session')).toEqual({ profile: 'work' })
  })

  it('returns no profile when no owner is known', () => {
    setSessionOwnerResolver(() => undefined)

    expect(sessionReadOwnerPin('stored-session')).toEqual({})
  })

  it('keeps the caller-named profile authoritative (string scope)', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'hermes-pi', profile: 'work', targetProfile: 'work' }))

    // The caller deliberately pointed this read at another profile: the owner
    // may still route the connection, but must not rewrite the profile.
    expect(sessionReadOwnerPin('stored-session', 'default')).toEqual({ connectionId: 'hermes-pi' })
  })

  it('keeps the caller-named profile authoritative (object scope)', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'work', targetProfile: 'work' }))

    expect(sessionReadOwnerPin('stored-session', { profile: 'default' })).toEqual({})
  })

  it('keeps the caller-named object profile while pinning the owner connection', () => {
    setSessionOwnerResolver(() => ({ connectionId: 'hermes-pi', profile: 'work', targetProfile: 'work' }))

    // The caller pointed the read at profile 'default' without naming a
    // connection: the owner still routes the connection, the profile stays the
    // caller's (capabilityScoped adds no connection for a bare object profile).
    expect(sessionReadOwnerPin('stored-session', { profile: 'default' })).toEqual({ connectionId: 'hermes-pi' })
  })
})

describe('getSessionMessages owner profile in the URL', () => {
  const api = vi.fn()

  beforeEach(() => {
    api.mockReset()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
    api.mockResolvedValue(page('stored-session'))
    setApiRequestProfile(null)
    setApiRequestConnection(null)
    setApiRequestLocalMode(true)
  })

  afterEach(() => {
    setSessionOwnerResolver(null)
    setApiRequestLocalMode(false)
    setApiRequestConnection(null)
    setApiRequestProfile(null)
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('stamps the owning profile on an unscoped transcript read', async () => {
    setSessionOwnerResolver(() => ({ connectionId: 'local', profile: 'work', targetProfile: 'work' }))

    await getSessionMessages('stored-session')

    // RED on main: the path carried no ?profile= and the serving home 404'd.
    expect(api).toHaveBeenCalled()
    const request = api.mock.calls[0][0] as { path: string }
    expect(request.path).toBe('/api/sessions/stored-session/messages?profile=work')
  })

  it('leaves the URL unscoped when no owner is known', async () => {
    setSessionOwnerResolver(() => undefined)

    await getSessionMessages('stored-session')

    const request = api.mock.calls[0][0] as { path: string }
    expect(request.path).toBe('/api/sessions/stored-session/messages')
  })
})
