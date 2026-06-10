/**
 * Tests for electron/dashboard-token.cjs.
 *
 * Run with: node --test electron/dashboard-token.test.cjs
 * (Wired into npm test:desktop:platforms in package.json.)
 */

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  dashboardIndexUrl,
  extractInjectedDashboardToken,
  fetchPublicText,
  resolveServedDashboardToken
} = require('./dashboard-token.cjs')

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
