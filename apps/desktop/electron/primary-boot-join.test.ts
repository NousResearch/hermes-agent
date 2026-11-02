import assert from 'node:assert/strict'

import { test } from 'vitest'

import { resolveProfileBackendRoute } from './connection-config'
import { desktopConnectionRequestInheritsPrimary, resolveDesktopConnectionRequest } from './desktop-profile'
import { joinPrimaryBoot, primaryBootRequest, primaryProfileRestorable } from './primary-boot-join'
import { PrimaryProfilePin } from './primary-profile-pin'
import { connectPrimaryRemote } from './primary-remote-startup'

const remote = {
  baseUrl: 'https://offline.invalid',
  wsUrl: 'wss://offline.invalid/api/ws',
  connectionId: 'homelab',
  source: 'registry',
  token: 'placeholder'
}

// main.ts's hermes:connection → ensureBackend path, minus the IPC: a renderer
// asking with no profile, against a global remote primary whose inherited
// profile ('windows-only') that gateway does not serve.
function rendererRequest(profile: string | undefined, pin: PrimaryProfilePin) {
  const route = resolveDesktopConnectionRequest(profile, null, pin.resolve(() => 'windows-only'))

  return primaryBootRequest(route.profile, pin.resolve(() => 'windows-only'), desktopConnectionRequestInheritsPrimary(profile, null))
}

const routeFor = (pin: PrimaryProfilePin) => (key: string) =>
  resolveProfileBackendRoute(key, { globalRemote: true, primaryProfile: pin.resolve(() => 'windows-only') })

test('a renderer that joins a remote boot mid-roster read adopts the served profile it re-pinned to', async () => {
  for (const asked of [undefined, 'windows-only'] as const) {
    const pin = new PrimaryProfilePin()
    pin.pin('windows-only')
    const request = rendererRequest(asked, pin)
    // Resolved before the wait, as ensureBackend does: it names the stale profile.
    const route = routeFor(pin)(request.key)
    assert.equal(route.descriptorProfile, 'windows-only')

    const joined = await joinPrimaryBoot(request, route, {
      primaryKey: () => pin.resolve(() => 'windows-only'),
      resolveRoute: routeFor(pin),
      // The in-flight boot's /api/profiles read lands while this caller waits.
      start: async () => {
        pin.pin('default')

        return { mode: 'remote' }
      }
    })

    // A profile-less caller follows the primary; one that named a profile keeps it.
    const want = asked ? 'windows-only' : 'default'
    assert.equal(joined.key, want, String(asked))
    assert.equal(joined.route.descriptorProfile, want, String(asked))
    // The descriptor ensureBackend returns: the global remote's shared primary, tagged with that profile.
    assert.deepEqual(joined.descriptor, { mode: 'remote', profile: want, sharedPrimary: true }, String(asked))
  }
})

test('a renderer retry after a failed remote boot still restores a served profile', async () => {
  const pin = new PrimaryProfilePin()
  // The failed boot cleared its pin; the retry resolves the stored preference.
  const request = rendererRequest(undefined, pin)
  assert.deepEqual(request, { key: 'windows-only', inheritsPrimary: true })

  // startHermes(request.key, { inheritsPrimary }) gates restoration on this.
  const restorable = () => primaryProfileRestorable(request.key, request.inheritsPrimary, null, {})

  pin.pin(request.key)

  const connection = await connectPrimaryRemote(remote, {
    profile: request.key,
    restoreRememberedProfile: restorable,
    assertCurrentAttempt: () => undefined,
    advanceBootProgress: async () => undefined,
    waitForRemote: async () => undefined,
    readProfiles: async () => ({ profiles: [{ name: 'default' }, { name: 'marina' }] }),
    onProfileRestored: profile => {
      pin.pin(profile)
    },
    updateBootProgress: () => undefined,
    logs: () => [],
    windowState: () => ({})
  })

  assert.equal(connection.profile, 'default')
  assert.equal(pin.booted, 'default')

  // An explicitly requested profile, a --profile launch or a per-profile
  // route is intent and is never swapped.
  for (const [inherits, launchArg, config] of [
    [false, null, {}],
    [true, 'windows-only', {}],
    [true, null, { profiles: { default: { mode: 'remote', url: 'https://work.invalid' } } }],
    [true, null, { profiles: { default: { mode: 'ssh', host: 'work.invalid' } } }]
  ] as const) {
    assert.equal(primaryProfileRestorable('windows-only', inherits, launchArg, config), false, JSON.stringify(config))
  }
})

test('a default profile kept local with only remembered SSH settings still repairs the global remote primary', async () => {
  // Settings → "This device" on the default profile parks its SSH form as
  // savedSsh; the global remote stays the primary's effective route.
  const config = {
    mode: 'remote',
    remote: { url: 'https://offline.invalid' },
    profiles: { default: { mode: 'local', savedSsh: { mode: 'ssh', host: 'box.invalid', user: 'me' } } }
  }

  const reads: string[] = []

  const connection = await connectPrimaryRemote(
    { ...remote, source: 'settings' },
    {
      profile: 'work',
      restoreRememberedProfile: () => primaryProfileRestorable('work', true, null, config),
      assertCurrentAttempt: () => undefined,
      advanceBootProgress: async () => undefined,
      waitForRemote: async () => undefined,
      readProfiles: async () => {
        reads.push('profiles')

        return { profiles: [{ name: 'default' }] }
      },
      onProfileRestored: () => undefined,
      updateBootProgress: () => undefined,
      logs: () => [],
      windowState: () => ({})
    }
  )

  assert.deepEqual(reads, ['profiles'])
  assert.equal(connection.profile, 'default')
})
