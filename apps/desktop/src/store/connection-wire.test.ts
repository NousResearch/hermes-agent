import { afterEach, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import { serverOwnsComposerQueue } from './composer-queue'
import { $connectionWires, connectionWire, primaryWireConnectionId, recordConnectionWire, recordSecondaryWire } from './connection-wire'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from './session'

afterEach(() => {
  $connectionWires.set({})
  $connection.set(null)
  _resetSessionOwnerHintsForTests({ storage: true })
})

// A foregrounded secondary keeps `$connection` while the primary reconnects in the background. An
// anonymous (local) primary's ready frame must not overwrite the foreground connection's wire.
it('records an anonymous primary under the local source, not the foreground connection', () => {
  $connection.set({ connectionId: 'vps', mode: 'remote', wsUrl: 'wss://vps.example/api/ws?ticket=public' } as HermesConnection)
  setSessionOwnerHint('chat', { connectionId: 'vps', mode: 'remote', profile: 'default' })
  recordConnectionWire('vps', 'legacy')

  recordConnectionWire(primaryWireConnectionId({ connectionId: null, mode: 'local' }, $connection.get()?.connectionId), 'canonical')

  expect(connectionWire('vps')).toBe('legacy')
  expect(connectionWire('local')).toBe('canonical')
  expect(serverOwnsComposerQueue('chat')).toBe(false)
})

it('names an established primary by its own route and an unrecorded one by the window', () => {
  expect(primaryWireConnectionId({ connectionId: 'home', mode: 'remote' }, 'vps')).toBe('home')
  expect(primaryWireConnectionId({ connectionId: null, mode: 'remote' }, 'vps')).toBeNull()
  expect(primaryWireConnectionId({ connectionId: null, mode: null }, 'vps')).toBe('vps')
})

// A profile-only secondary names no connection, but its descriptor names the backend it reached.
it('records a secondary under its registry id, else the backend its profile resolved to', () => {
  recordConnectionWire('local', 'canonical')

  recordSecondaryWire('home', { connectionId: 'vps', mode: 'remote' }, 'legacy')
  recordSecondaryWire(null, { connectionId: 'vps', mode: 'remote' }, 'canonical')
  recordSecondaryWire(null, { mode: 'remote' }, 'legacy')

  expect(connectionWire('home')).toBe('legacy')
  expect(connectionWire('vps')).toBe('canonical')
  expect(connectionWire('local')).toBe('canonical')

  recordSecondaryWire(null, { mode: 'local' }, 'legacy')
  expect(connectionWire('local')).toBe('legacy')
})
