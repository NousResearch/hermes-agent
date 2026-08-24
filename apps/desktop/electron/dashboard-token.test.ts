/**
 * Tests for electron/dashboard-token.ts.
 *
 * Run with: node --test electron/dashboard-token.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  adoptServedDashboardToken,
  dashboardIndexUrl,
  extractInjectedDashboardToken,
  fetchPublicText,
  isForeignBackendToken,
  resolveServedDashboardToken
} from './dashboard-token'

test('extractInjectedDashboardToken reads the JSON-encoded dashboard token', () => {
  const html = '<script>window.__HERMES_SESSION_TOKEN__="served-token";window.__HERMES_BASE_PATH__=""</script>'
  assert.equal(extractInjectedDashboardToken(html), 'served-token')
})

test('extractInjectedDashboardToken handles escaped token strings', () => {
  const html = '<script>window.__HERMES_SESSION_TOKEN__="served\\\\token\\"quoted";</script>'
  assert.equal(extractInjectedDashboardToken(html), 'served\\token"quoted')
})

test('extractInjectedDashboardToken returns null for missing or malformed values', () => {
  assert.equal(extractInjectedDashboardToken('<html></html>'), null)
  assert.equal(extractInjectedDashboardToken('<script>window.__HERMES_SESSION_TOKEN__={bad}</script>'), null)
})

test('dashboardIndexUrl preserves dashboard path prefixes', () => {
  assert.equal(dashboardIndexUrl('http://127.0.0.1:9120'), 'http://127.0.0.1:9120/')
  assert.equal(dashboardIndexUrl('https://host.example/hermes/'), 'https://host.example/hermes/')
})

test('resolveServedDashboardToken uses the served token and logs when it differs', async () => {
  const logs = []

  const token = await resolveServedDashboardToken('http://127.0.0.1:9120', 'spawn-token', {
    fetchText: async url => {
      assert.equal(url, 'http://127.0.0.1:9120/')

      return '<script>window.__HERMES_SESSION_TOKEN__="served-token";</script>'
    },
    rememberLog: line => logs.push(line)
  })

  assert.equal(token, 'served-token')
  assert.equal(logs.length, 1)
  assert.match(logs[0], /served a different session token/)
})

test('resolveServedDashboardToken falls back when the served HTML has no token', async () => {
  const token = await resolveServedDashboardToken('http://127.0.0.1:9120', 'spawn-token', {
    fetchText: async () => '<html></html>',
    rememberLog: () => {
      throw new Error('should not log when no served token is present')
    }
  })

  assert.equal(token, 'spawn-token')
})

test('resolveServedDashboardToken does not log when served token matches fallback', async () => {
  const token = await resolveServedDashboardToken('http://127.0.0.1:9120', 'same-token', {
    fetchText: async () => '<script>window.__HERMES_SESSION_TOKEN__="same-token";</script>',
    rememberLog: () => {
      throw new Error('should not log when token already matches')
    }
  })

  assert.equal(token, 'same-token')
})

test('resolveServedDashboardToken propagates fetch errors so callers can fall back explicitly', async () => {
  await assert.rejects(
    () =>
      resolveServedDashboardToken('http://127.0.0.1:9120', 'spawn-token', {
        fetchText: async () => {
          throw new Error('boom')
        }
      }),
    /boom/
  )
})

test('fetchPublicText rejects unsupported protocols', async () => {
  await assert.rejects(() => fetchPublicText('file:///tmp/index.html'), /Unsupported Hermes backend URL protocol/)
})

test('isForeignBackendToken only flags a mismatched token from a dead child', () => {
  const cases = [
    [{ servedToken: 'other', spawnToken: 'mine', childAlive: false }, true],
    // Live child + drift = our backend regenerated the token (env pin lost).
    [{ servedToken: 'other', spawnToken: 'mine', childAlive: true }, false],
    [{ servedToken: 'mine', spawnToken: 'mine', childAlive: false }, false],
    [{ servedToken: 'mine', spawnToken: 'mine', childAlive: true }, false],
    [{ servedToken: null, spawnToken: 'mine', childAlive: false }, false],
    [{ servedToken: '', spawnToken: 'mine', childAlive: false }, false]
  ]

  for (const [input, expected] of cases) {
    assert.equal(isForeignBackendToken(input as any), expected, JSON.stringify(input))
  }
})

test('adoptServedDashboardToken adopts drift from a live child', async () => {
  const token = await adoptServedDashboardToken('http://127.0.0.1:9120', 'spawn-token', {
    childAlive: () => true,
    fetchText: async () => '<script>window.__HERMES_SESSION_TOKEN__="served-token";</script>'
  })

  assert.equal(token, 'served-token')
})

test('adoptServedDashboardToken refuses a foreign token when our child is dead', async () => {
  await assert.rejects(
    () =>
      adoptServedDashboardToken('http://127.0.0.1:9120', 'spawn-token', {
        childAlive: () => false,
        fetchText: async () => '<script>window.__HERMES_SESSION_TOKEN__="squatter-token";</script>',
        label: 'Hermes backend for profile "work"'
      }),
    /profile "work".*process we did not spawn/
  )
})

test('adoptServedDashboardToken falls back to the spawn token when the fetch fails', async () => {
  const logs = []

  const token = await adoptServedDashboardToken('http://127.0.0.1:9120', 'spawn-token', {
    childAlive: () => true,
    fetchText: async () => {
      throw new Error('boom')
    },
    rememberLog: line => logs.push(line)
  })

  assert.equal(token, 'spawn-token')
  assert.equal(logs.length, 1)
  assert.match(logs[0], /could not read served dashboard token \(Hermes backend\): boom/)
})
