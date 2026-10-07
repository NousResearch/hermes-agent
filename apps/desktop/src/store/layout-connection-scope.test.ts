import { beforeEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'
import { readKey } from '@/lib/storage'

import { $pinnedSessionIds, $sidebarSessionOrderIds, $sidebarSessionOrderManual, pinSession } from './layout'
import { getRememberedSessionId, setConnection, setRememberedSessionId } from './session'

// Two Desktop windows share one renderer origin (and therefore one
// localStorage area) while each can be connected to a DIFFERENT gateway.
// Session/pin lists reconcile against each gateway's own state.db, so any of
// these lists persisted under a single global key bleeds between windows on
// different backends (#77318). These tests pin the scope contract: the
// gateway-bound sidebar lists must be keyed per connection.

const localConn = {
  baseUrl: 'http://127.0.0.1:8000',
  mode: 'local',
  profile: 'default'
} as unknown as HermesConnection

const remoteA = {
  baseUrl: 'https://vps-a.example:8443',
  mode: 'remote',
  profile: 'default'
} as unknown as HermesConnection

const remoteB = {
  baseUrl: 'https://vps-b.example:8443',
  mode: 'remote',
  profile: 'default'
} as unknown as HermesConnection

beforeEach(() => {
  window.localStorage.clear()
  // Return to the local scope and empty every list under it.
  setConnection(localConn)
  $pinnedSessionIds.set([])
  $sidebarSessionOrderIds.set([])
  $sidebarSessionOrderManual.set(false)
})

describe('connection-scoped sidebar lists (#77318)', () => {
  it('does not leak the local pin set into a remote connection', () => {
    pinSession('local-1')
    pinSession('local-2')

    setConnection(remoteA)

    // The remote gateway has its own state.db; its pin list starts from its
    // own scope (empty here), never from the local window's pins.
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('keeps pin sets isolated between two different remote gateways', () => {
    setConnection(remoteA)
    pinSession('a-1')

    setConnection(remoteB)
    expect($pinnedSessionIds.get()).toEqual([])
    pinSession('b-1')

    // Each scope round-trips its own set.
    setConnection(remoteA)
    expect($pinnedSessionIds.get()).toEqual(['a-1'])

    setConnection(remoteB)
    expect($pinnedSessionIds.get()).toEqual(['b-1'])
  })

  it('restores the local pin set when returning to the local connection', () => {
    pinSession('local-1')

    setConnection(remoteA)
    pinSession('a-1')

    setConnection(localConn)
    expect($pinnedSessionIds.get()).toEqual(['local-1'])
  })

  it('writes remote pins to a connection-scoped key, leaving the legacy key intact', () => {
    pinSession('local-1')

    setConnection(remoteA)
    pinSession('a-1')

    // The bare key still belongs to the local connection.
    expect(readKey('hermes.desktop.pinnedSessions')).toBe(JSON.stringify(['local-1']))

    // The remote pin landed under its own gateway scope, not the shared key
    // and not a per-profile fragment.
    const scoped = readKey(`hermes.desktop.pinnedSessions.remote.${encodeURIComponent('https://vps-a.example:8443')}`)

    expect(scoped).toBe(JSON.stringify(['a-1']))
  })

  it('never adopts legacy globally-keyed pins into a remote scope', () => {
    // Pre-fix installs persisted every window's pins under the one global
    // key. Ownership of those rows is unknowable, so a remote scope must
    // start empty (the gateway's own `pinned` rows re-adopt what belongs
    // there) rather than inherit the mixed set wholesale.
    pinSession('mixed-1')
    pinSession('mixed-2')

    setConnection(remoteA)
    expect($pinnedSessionIds.get()).toEqual([])
  })

  it('a null connection (reconnect blip) keeps the current scope', () => {
    setConnection(remoteA)
    pinSession('a-1')

    // Disconnects publish null; that is a connectivity state, not a switch —
    // the remote window must not repaint the local scope's lists.
    setConnection(null)
    expect($pinnedSessionIds.get()).toEqual(['a-1'])
  })

  it('keeps pins across a profile switch on the same gateway, but not session order', () => {
    setConnection(remoteA)
    pinSession('a-1')
    $sidebarSessionOrderIds.set(['s1'])
    $sidebarSessionOrderManual.set(true)

    setConnection({ ...remoteA, profile: 'k9' } as unknown as HermesConnection)

    expect($pinnedSessionIds.get()).toEqual(['a-1'])
    expect($sidebarSessionOrderIds.get()).toEqual([])
    expect($sidebarSessionOrderManual.get()).toBe(false)
  })

  it('scopes the manual session order and its flag per connection', () => {
    $sidebarSessionOrderIds.set(['s1', 's2'])
    $sidebarSessionOrderManual.set(true)

    setConnection(remoteA)
    expect($sidebarSessionOrderIds.get()).toEqual([])
    expect($sidebarSessionOrderManual.get()).toBe(false)

    $sidebarSessionOrderIds.set(['r1'])
    $sidebarSessionOrderManual.set(true)

    setConnection(localConn)
    expect($sidebarSessionOrderIds.get()).toEqual(['s1', 's2'])
    expect($sidebarSessionOrderManual.get()).toBe(true)

    setConnection(remoteA)
    expect($sidebarSessionOrderIds.get()).toEqual(['r1'])
  })

  it('scopes remembered session navigation per connection, not just per profile', () => {
    setRememberedSessionId('local-session', 'default')

    setConnection(remoteA)
    // Same profile name on another gateway is a different backend: restoring
    // the local window's session id here would navigate to a session this
    // gateway has never seen.
    expect(getRememberedSessionId('default')).toBeNull()

    setRememberedSessionId('remote-session', 'default')
    expect(getRememberedSessionId('default')).toBe('remote-session')

    setConnection(localConn)
    expect(getRememberedSessionId('default')).toBe('local-session')
  })
})
