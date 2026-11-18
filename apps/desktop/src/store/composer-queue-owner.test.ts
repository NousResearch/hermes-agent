import { afterEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import { serverOwnsComposerQueue } from './composer-queue'
import { $connectionWires, recordConnectionWire } from './connection-wire'
import { $connection, _resetSessionOwnerHintsForTests, setSessionOwnerHint } from './session'

const browser = (wsUrl: string) => ({ mode: 'remote', wsUrl }) as HermesConnection

afterEach(() => $connection.set(null))

// The browser-hosted Desktop attaches to the canonical socket of the gateway serving it: its
// owner admits queued input into the durable FIFO, so the composer must hand its queue to the
// server exactly as local Electron does, not keep a renderer-local queue the owner never sees.
describe('composer queue ownership for the browser-hosted Desktop', () => {
  it('belongs to the gateway when the window dials its canonical socket', () => {
    $connection.set(browser('wss://host.example/api/ws?profile=work&native_dial=browser'))

    expect(serverOwnsComposerQueue('a-session-with-no-recorded-owner')).toBe(true)
  })

  it('stays in the renderer for a backend still on the legacy contract', () => {
    $connection.set(browser('wss://host.example/api/ws?token=served-token'))

    expect(serverOwnsComposerQueue('a-session-with-no-recorded-owner')).toBe(false)
  })
})

// A public remote connection learns its wire from gateway.ready. The composer asks the owner's
// connection, so a canonical remote chat hands images, /undo, /retry and busy input to the owner
// instead of keeping them in a renderer path the canonical wire refuses or silently drops.
describe('composer queue ownership follows the negotiated wire', () => {
  afterEach(() => {
    $connectionWires.set({})
    _resetSessionOwnerHintsForTests({ storage: true })
  })

  const remote = { connectionId: 'vps', mode: 'remote', wsUrl: 'wss://vps.example/api/ws?ticket=public' } as HermesConnection

  it('belongs to the owner of a public connection that negotiated the canonical wire', () => {
    $connection.set(remote)
    expect(serverOwnsComposerQueue('chat')).toBe(false)
    recordConnectionWire('vps', 'canonical')
    expect(serverOwnsComposerQueue('chat')).toBe(true)
  })

  it('reads a remote owner route\'s own connection, not the early mode', () => {
    $connection.set(remote)
    setSessionOwnerHint('chat', { connectionId: 'vps', mode: 'remote', profile: 'default' })
    recordConnectionWire('vps', 'canonical')
    expect(serverOwnsComposerQueue('chat')).toBe(true)
    recordConnectionWire('vps', 'legacy')
    expect(serverOwnsComposerQueue('chat')).toBe(false)
  })

  it('keeps a background connection\'s wire apart from the foreground one', () => {
    $connection.set(remote)
    setSessionOwnerHint('chat', { connectionId: 'other', mode: 'remote', profile: 'default' })
    recordConnectionWire('vps', 'canonical')
    expect(serverOwnsComposerQueue('chat')).toBe(false)
    recordConnectionWire('other', 'canonical')
    expect(serverOwnsComposerQueue('chat')).toBe(true)
  })

  // Pre-existing on main: the webapp stamps Bot Chat owners `mode: 'remote'` on its own native dial.
  it('follows a native dial for a remote-stamped owner of the same connection before ready', () => {
    $connection.set({ connectionId: 'served', mode: 'remote', wsUrl: 'wss://host.example/api/ws?native_dial=browser' } as HermesConnection)
    setSessionOwnerHint('chat', { connectionId: 'served', mode: 'remote', profile: 'bot' })
    expect(serverOwnsComposerQueue('chat')).toBe(true)
  })
})
