import { afterEach, describe, expect, it } from 'vitest'

import type { HermesConnection } from '@/global'

import { serverOwnsComposerQueue } from './composer-queue'
import { $connection } from './session'

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
