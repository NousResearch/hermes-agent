/**
 * Tests for electron/ix-auto-provision.ts — the zero-touch provisioning
 * client (fetch the portal payload with the OTP session, fill only the
 * empty settings slots).
 *
 * Run with: node --test electron/ix-auto-provision.test.ts
 *
 * Why this matters: this path writes real secrets into safeStorage-backed
 * settings, so it must NEVER clobber a manual value, must be idempotent
 * across repeated logins, and must reject junk WireGuard confs.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { DEFAULT_IX_AGENCY_SETTINGS, type IxAgencySettings } from './ix-agency'
import {
  applyIxProvisionToSettings,
  fetchIxDesktopProvision,
  type IxDesktopProvisionPayload
} from './ix-auto-provision'

const PORTAL = 'https://admin.example.test'

const WG_CONF = '[Interface]\nPrivateKey = abc\nAddress = 10.8.0.9/24\n\n[Peer]\nPublicKey = def\n'

function payload(overrides: Partial<IxDesktopProvisionPayload> = {}): IxDesktopProvisionPayload {
  return {
    gatewayToken: 'gw-token',
    litellm: { url: 'https://litellm.example.test', key: 'sk-user-key', source: 'per-user' },
    cognito: { clientId: 'client-from-portal', clientSecret: 'cog-secret' },
    wireguard: { conf: WG_CONF, source: 'per-user' },
    ...overrides
  }
}

function settings(overrides: Partial<IxAgencySettings> = {}): IxAgencySettings {
  return { ...DEFAULT_IX_AGENCY_SETTINGS, ...overrides }
}

function fakeFetch(handler: (url: string, init?: RequestInit) => { body?: unknown; status?: number }): typeof fetch {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const { body = {}, status = 200 } = handler(String(input), init)

    return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  }) as typeof fetch
}

/* ── fetch ────────────────────────────────────────────────────────────────── */

test('fetch hits the provision endpoint with credentials include', async () => {
  let seenUrl = ''
  let seenInit: RequestInit | undefined

  const result = await fetchIxDesktopProvision(
    PORTAL,
    fakeFetch((url, init) => {
      seenUrl = url
      seenInit = init

      return { body: { ok: true, ...payload() } }
    })
  )

  assert.equal(seenUrl, `${PORTAL}/api/portal/desktop/provision`)
  assert.equal(seenInit?.credentials, 'include')
  assert.equal(result.gatewayToken, 'gw-token')
  assert.equal(result.litellm.key, 'sk-user-key')
  assert.equal(result.wireguard.conf, WG_CONF)
})

test('fetch maps 401 to a sign-in-first error', async () => {
  await assert.rejects(fetchIxDesktopProvision(PORTAL, fakeFetch(() => ({ status: 401 }))), /sign in/i)
})

test('fetch rejects a payload without ok:true', async () => {
  await assert.rejects(fetchIxDesktopProvision(PORTAL, fakeFetch(() => ({ body: {} }))), /did not return/)
})

test('fetch tolerates missing slots (degraded server config)', async () => {
  const result = await fetchIxDesktopProvision(
    PORTAL,
    fakeFetch(() => ({ body: { ok: true, gatewayToken: '', litellm: { key: '', source: 'unavailable' } } }))
  )

  assert.equal(result.gatewayToken, '')
  assert.equal(result.cognito.clientSecret, '')
  assert.equal(result.wireguard.conf, '')
})

/* ── apply ────────────────────────────────────────────────────────────────── */

test('fills every empty slot on a fresh install', () => {
  const { next, filled } = applyIxProvisionToSettings(settings(), payload())

  assert.deepEqual(filled.sort(), ['cognitoClientSecret', 'gatewayToken', 'litellmKey', 'vpnConfSecret'])
  assert.equal(next.gatewayToken, 'gw-token')
  assert.equal(next.litellmKey, 'sk-user-key')
  assert.equal(next.litellmUrl, 'https://litellm.example.test')
  assert.equal(next.cognitoClientSecret, 'cog-secret')
  assert.equal(next.cognitoClientId, 'client-from-portal')
  assert.equal(next.vpnConfSecret, WG_CONF)
})

test('never clobbers manual values', () => {
  const manual = settings({
    gatewayToken: 'manual-gw',
    litellmKey: 'manual-key',
    litellmUrl: 'https://my-litellm.example',
    cognitoClientSecret: 'manual-cog',
    cognitoClientId: 'manual-client',
    vpnConfSecret: '[Interface]\nPrivateKey = manual\n[Peer]\n'
  })

  const { next, filled } = applyIxProvisionToSettings(manual, payload())

  assert.deepEqual(filled, [])
  assert.deepEqual(next, manual)
})

test('is idempotent — a second pass after filling changes nothing', () => {
  const first = applyIxProvisionToSettings(settings(), payload())
  const second = applyIxProvisionToSettings(first.next, payload())

  assert.deepEqual(second.filled, [])
  assert.deepEqual(second.next, first.next)
})

test('a manually-configured .conf PATH blocks the keychain VPN import', () => {
  const { next, filled } = applyIxProvisionToSettings(
    settings({ vpnConfPath: '/home/user/usa-vpn.conf' }),
    payload()
  )

  assert.ok(!filled.includes('vpnConfSecret'))
  assert.equal(next.vpnConfSecret, '')
})

test('rejects junk WireGuard material', () => {
  const { filled } = applyIxProvisionToSettings(
    settings(),
    payload({ wireguard: { conf: 'not a wireguard conf', source: 'shared' } })
  )

  assert.ok(!filled.includes('vpnConfSecret'))
})

test('custom LiteLLM URL survives even when the key is filled', () => {
  const { next, filled } = applyIxProvisionToSettings(settings({ litellmUrl: 'https://my-litellm.example' }), payload())

  assert.ok(filled.includes('litellmKey'))
  assert.equal(next.litellmUrl, 'https://my-litellm.example')
})

test('custom Cognito client id survives even when the secret is filled', () => {
  const { next, filled } = applyIxProvisionToSettings(settings({ cognitoClientId: 'my-own-client' }), payload())

  assert.ok(filled.includes('cognitoClientSecret'))
  assert.equal(next.cognitoClientId, 'my-own-client')
})

test('partial payloads fill only what they carry', () => {
  const { next, filled } = applyIxProvisionToSettings(
    settings(),
    payload({
      gatewayToken: '',
      litellm: { url: '', key: '', source: 'unavailable' },
      wireguard: { conf: '', source: 'unavailable' }
    })
  )

  assert.deepEqual(filled, ['cognitoClientSecret'])
  assert.equal(next.gatewayToken, '')
  assert.equal(next.vpnConfSecret, '')
})
