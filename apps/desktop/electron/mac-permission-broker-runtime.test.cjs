'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const {
  brokerAvailable,
  brokerExecutableForAppBundle,
  brokerExecutableFromProcess,
  brokerServiceStatus,
  brokerStatus,
  openBrokerSettings,
  parseBrokerJson,
  registerBrokerLoginItem,
  unregisterBrokerLoginItem
} = require('./mac-permission-broker-runtime.cjs')

test('brokerExecutableForAppBundle points at LoginItems helper executable', () => {
  assert.equal(
    brokerExecutableForAppBundle('/Applications/Hermes.app'),
    '/Applications/Hermes.app/Contents/Library/LoginItems/HermesMacBroker.app/Contents/MacOS/HermesMacBroker'
  )
  assert.equal(brokerExecutableForAppBundle(''), null)
})

test('brokerExecutableFromProcess resolves packaged app paths and ignores non-macOS', () => {
  assert.equal(
    brokerExecutableFromProcess({ platform: 'darwin', execPath: '/Applications/Hermes.app/Contents/MacOS/Hermes' }),
    '/Applications/Hermes.app/Contents/Library/LoginItems/HermesMacBroker.app/Contents/MacOS/HermesMacBroker'
  )
  assert.equal(
    brokerExecutableFromProcess({ platform: 'darwin', resourcesPath: '/Applications/Hermes.app/Contents/Resources', execPath: '/usr/bin/node' }),
    '/Applications/Hermes.app/Contents/Library/LoginItems/HermesMacBroker.app/Contents/MacOS/HermesMacBroker'
  )
  assert.equal(brokerExecutableFromProcess({ platform: 'linux', execPath: '/opt/Hermes' }), null)
})

test('brokerAvailable uses the supplied filesystem implementation', () => {
  assert.equal(brokerAvailable('/x', { existsSync: () => true }), true)
  assert.equal(brokerAvailable('/x', { existsSync: () => false }), false)
  assert.equal(brokerAvailable(null, { existsSync: () => true }), false)
})

test('parseBrokerJson returns structured errors for invalid output', () => {
  assert.deepEqual(parseBrokerJson('{"ok":true}'), { ok: true })
  const bad = parseBrokerJson('not-json')
  assert.equal(bad.ok, false)
  assert.match(bad.error, /invalid broker JSON/)
  assert.equal(bad.raw, 'not-json')
})

test('brokerStatus and openBrokerSettings call the expected helper commands', () => {
  const calls = []
  const execFileSync = (exe, args) => {
    calls.push([exe, args])
    return '{"ok":true}'
  }
  const fsImpl = { existsSync: () => true }

  assert.deepEqual(brokerStatus('/broker', { execFileSync, fsImpl }), { ok: true })
  assert.deepEqual(brokerServiceStatus('/broker', { execFileSync, fsImpl }), { ok: true })
  assert.deepEqual(registerBrokerLoginItem('/broker', { execFileSync, fsImpl }), { ok: true })
  assert.deepEqual(unregisterBrokerLoginItem('/broker', { execFileSync, fsImpl }), { ok: true })
  assert.deepEqual(openBrokerSettings('/broker', 'accessibility', { execFileSync, fsImpl }), { ok: true })
  assert.deepEqual(calls, [
    ['/broker', ['--status-json']],
    ['/broker', ['--service-status']],
    ['/broker', ['--register-login-item']],
    ['/broker', ['--unregister-login-item']],
    ['/broker', ['--open-settings', 'accessibility']]
  ])
})

test('brokerStatus reports missing helper without throwing', () => {
  const result = brokerStatus('/missing', { fsImpl: { existsSync: () => false } })
  assert.equal(result.ok, false)
  assert.match(result.error, /not installed/)
})
