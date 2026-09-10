import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  createDesktopUpdateIpcHandlers,
  MACOS_SAFE_PUBLISHER_CAPABILITY,
  parseMacosSafePublisherCapability,
  registerDesktopUpdateIpc,
  SAFE_PUBLISHER_UNAVAILABLE_MESSAGE
} from './macos-update-containment'

test('macOS apply IPC rechecks the safe publisher while check-only stays available', async () => {
  let capabilityVersion: number | null = null
  let applies = 0
  let checks = 0

  const handlers = createDesktopUpdateIpcHandlers({
    applyUpdates: async () => {
      applies += 1

      return { ok: true }
    },
    checkUpdates: async () => {
      checks += 1

      return { supported: true }
    },
    isMac: true,
    publisherCapability: () =>
      capabilityVersion === null
        ? null
        : { capability: MACOS_SAFE_PUBLISHER_CAPABILITY, version: capabilityVersion }
  })

  assert.deepEqual(await handlers.check(), { supported: true })
  assert.equal(checks, 1, 'check-only must not depend on mutation capability')

  assert.deepEqual(await handlers.apply({ stopSafeBlockers: true }), {
    error: 'safe-publisher-unavailable',
    message: SAFE_PUBLISHER_UNAVAILABLE_MESSAGE,
    ok: false
  })
  assert.equal(applies, 0, 'direct IPC must not reach mutation without the capability')

  capabilityVersion = 1
  assert.deepEqual(await handlers.apply({ stopSafeBlockers: true }), { ok: true })
  assert.equal(applies, 1)

  capabilityVersion = null
  assert.equal((await handlers.apply({ staleRendererRequest: true } as any)).ok, false)
  assert.equal(applies, 1, 'a stale renderer cannot reuse an earlier positive capability result')
})

test('registers the authoritative check and apply handlers at the Electron main boundary', async () => {
  const registered = new Map<string, (...args: any[]) => Promise<any>>()
  registerDesktopUpdateIpc(
    {
      handle: (channel: string, handler: (...args: any[]) => Promise<any>) => {
        registered.set(channel, handler)
      }
    },
    createDesktopUpdateIpcHandlers({
      applyUpdates: async () => ({ ok: true }),
      checkUpdates: async () => ({ supported: true }),
      isMac: true,
      publisherCapability: () => null
    })
  )

  const check = registered.get('hermes:updates:check')
  const apply = registered.get('hermes:updates:apply')

  if (!check || !apply) {
    throw new Error('update IPC handlers were not registered')
  }

  assert.deepEqual(await check({}), { supported: true })
  assert.equal((await apply({}, {})).ok, false)
})

test('macOS publisher capability parsing fails closed and requires a compatible version', () => {
  assert.deepEqual(
    parseMacosSafePublisherCapability(
      JSON.stringify({ capability: MACOS_SAFE_PUBLISHER_CAPABILITY, version: 1, additiveField: 'ignored' })
    ),
    { capability: MACOS_SAFE_PUBLISHER_CAPABILITY, version: 1 }
  )
  assert.equal(
    parseMacosSafePublisherCapability(JSON.stringify({ capability: MACOS_SAFE_PUBLISHER_CAPABILITY, version: 0 })),
    null
  )
  assert.equal(parseMacosSafePublisherCapability(JSON.stringify({ capability: 'legacy-posix-swap', version: 99 })), null)
  assert.equal(parseMacosSafePublisherCapability('not-json'), null)
})

test('Linux and Windows update apply behavior does not probe the macOS publisher', async () => {
  for (const isMac of [false]) {
    let probes = 0

    const handlers = createDesktopUpdateIpcHandlers({
      applyUpdates: async payload => ({ ok: true, payload }),
      checkUpdates: async () => ({ supported: true }),
      isMac,
      publisherCapability: () => {
        probes += 1

        return null
      }
    })

    assert.equal((await handlers.apply({ platform: 'preserved' })).ok, true)
    assert.equal(probes, 0)
  }
})
