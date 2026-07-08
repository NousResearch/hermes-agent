/**
 * Regression coverage for the Desktop-owned local backend auth handshake.
 *
 * Run with: node --test electron/desktop-backend-auth.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const mainPath = path.join(__dirname, 'main.cjs')
const source = fs.readFileSync(mainPath, 'utf8')

function functionSource(functionName, followingFunctionName) {
  const start = source.indexOf(`async function ${functionName}`)
  assert.notEqual(start, -1, `main.cjs should define ${functionName}`)
  const end = source.indexOf(`function ${followingFunctionName}`, start)
  assert.notEqual(end, -1, `${functionName} should be followed by ${followingFunctionName}`)
  return source.slice(start, end)
}

function assertAdoptsBeforeReadiness(body, label) {
  const adoptIndex = body.indexOf('const authToken = await adoptServedDashboardToken(baseUrl, token')
  const waitWithAuthIndex = body.indexOf('waitForHermes(baseUrl, authToken)')

  assert.notEqual(adoptIndex, -1, `${label} should adopt the served dashboard token`)
  assert.notEqual(waitWithAuthIndex, -1, `${label} should probe readiness with authToken`)
  assert.ok(
    adoptIndex < waitWithAuthIndex,
    `${label} token adoption must happen before the readiness API probe`
  )
  assert.doesNotMatch(
    body,
    /waitForHermes\(baseUrl,\s*token\)/,
    `${label} must not probe with the pre-adoption spawn token`
  )
}

test('primary desktop backend starts isolated to avoid unified-dashboard token drift', () => {
  const body = functionSource('startHermes()', 'wireCommonWindowHandlers')
  assert.match(
    body,
    /const backendArgs = \['serve', '--host', '127\.0\.0\.1', '--port', '0', '--isolated'\]/,
    'desktop-spawned serve backend should pass --isolated'
  )
})

test('primary desktop backend adopts served dashboard token before readiness API probe', () => {
  assertAdoptsBeforeReadiness(functionSource('startHermes()', 'wireCommonWindowHandlers'), 'startHermes()')
})

test('profile pool backend starts isolated to avoid unified-dashboard token drift', () => {
  const body = functionSource('spawnPoolBackend(profile, entry)', 'stopPoolBackend')
  assert.match(
    body,
    /const backendArgs = \['--profile', profile, 'serve', '--host', '127\.0\.0\.1', '--port', '0', '--isolated'\]/,
    'profile pool serve backend should pass --isolated'
  )
})

test('profile pool backend adopts served dashboard token before readiness API probe', () => {
  assertAdoptsBeforeReadiness(functionSource('spawnPoolBackend(profile, entry)', 'stopPoolBackend'), 'spawnPoolBackend()')
})
