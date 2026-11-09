import { afterEach, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import { serverOwnsComposerQueue } from './composer-queue'
import { $connectionWires, connectionWire, primaryWireSource, recordConnectionWire, recordSecondaryWire, UNNAMED_REMOTE_WIRE_KEY } from './connection-wire'
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

  recordConnectionWire(primaryWireSource({ connectionId: null, mode: 'local' }, $connection.get()), 'canonical')

  expect(connectionWire('vps')).toBe('legacy')
  expect(connectionWire('local')).toBe('canonical')
  expect(serverOwnsComposerQueue('chat')).toBe(false)
})

it('names an established primary by its own route and an unrecorded one by the window', () => {
  const vps = { connectionId: 'vps', mode: 'remote' }
  expect(primaryWireSource({ connectionId: 'home', mode: 'remote' }, vps)).toEqual({ connectionId: 'home', mode: 'remote' })
  expect(primaryWireSource({ connectionId: null, mode: 'remote' }, vps)).toEqual({ connectionId: null, mode: 'remote' })
  expect(primaryWireSource({ connectionId: null, mode: null }, vps)).toBe(vps)
})

// `HERMES_DESKTOP_REMOTE_URL` gives the primary a remote with no registry id, while a real local
// source can stay connected as a secondary. Neither may overwrite the other's wire.
it('keeps an unnamed remote primary apart from the local source', () => {
  recordConnectionWire('local', 'canonical')
  recordConnectionWire(primaryWireSource({ connectionId: null, mode: 'remote' }, null), 'legacy')

  expect(connectionWire('local')).toBe('canonical')
  expect($connectionWires.get()[UNNAMED_REMOTE_WIRE_KEY]).toBe('legacy')
  setSessionOwnerHint('native', { connectionId: 'local', mode: 'local', profile: 'default' })
  expect(serverOwnsComposerQueue('native')).toBe(true)

  // A chat with no recorded owner on the unnamed remote window reads that remote's wire.
  $connection.set({ mode: 'remote', wsUrl: 'wss://remote.example/api/ws?token=t' } as HermesConnection)
  expect(serverOwnsComposerQueue('remote-chat')).toBe(false)
  recordConnectionWire({ mode: 'remote' }, 'canonical')
  expect(serverOwnsComposerQueue('remote-chat')).toBe(true)
  expect(connectionWire('local')).toBe('canonical')
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

// A profile pointed at a remote backend (connection.json `profiles.<name>`) dials under the local
// scope, so its secondary reports `local` for both ids. A legacy remote must not flip the local
// source's canonical wire, or local image attaches fall back to `image.attach` and fail.
it('keeps a profile remote override from overwriting the local source wire', () => {
  recordConnectionWire('local', 'canonical')
  $connection.set({ connectionId: 'local', mode: 'local', wsUrl: 'ws://127.0.0.1:1/api/ws?native_dial=x' } as HermesConnection)
  setSessionOwnerHint('chat', { connectionId: 'local', profile: 'mishu' })

  recordSecondaryWire('local', { connectionId: 'local', mode: 'remote' }, 'legacy')

  expect(connectionWire('local')).toBe('canonical')
  expect(serverOwnsComposerQueue('chat')).toBe(true)
})
