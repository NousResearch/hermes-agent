'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  BROKER_APP_NAME,
  BROKER_BUNDLE_ID,
  BROKER_EXECUTABLE_NAME,
  brokerBundleInfoPlist,
  brokerLoginItemsDir,
  installBrokerIntoApp,
  shouldStageBroker,
  stageMacPermissionBroker,
  swiftcArgs
} = require('./stage-mac-permission-broker.cjs')

test('brokerBundleInfoPlist declares a stable helper identity and usage descriptions', () => {
  const plist = brokerBundleInfoPlist()

  assert.match(plist, new RegExp(`<string>${BROKER_BUNDLE_ID}</string>`))
  assert.match(plist, new RegExp(`<string>${BROKER_EXECUTABLE_NAME}</string>`))
  assert.match(plist, /NSMicrophoneUsageDescription/)
  assert.match(plist, /NSAppleEventsUsageDescription/)
  assert.match(plist, /LSUIElement/)
})

test('swiftcArgs compiles the helper with required macOS frameworks', () => {
  const args = swiftcArgs({ source: '/tmp/main.swift', output: '/tmp/HermesMacBroker' })

  assert.deepEqual(args.slice(0, 3), ['/tmp/main.swift', '-o', '/tmp/HermesMacBroker'])
  assert.ok(args.includes('AppKit'))
  assert.ok(args.includes('ApplicationServices'))
  assert.ok(args.includes('AVFoundation'))
  assert.ok(args.includes('UserNotifications'))
})

test('shouldStageBroker only stages on darwin unless explicitly skipped', () => {
  assert.equal(shouldStageBroker('darwin', {}), true)
  assert.equal(shouldStageBroker('linux', {}), false)
  assert.equal(shouldStageBroker('win32', {}), false)
  assert.equal(shouldStageBroker('darwin', { HERMES_DESKTOP_SKIP_MAC_BROKER: '1' }), false)
})

test('installBrokerIntoApp copies the helper into Contents/Library/LoginItems', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-broker-test-'))
  const app = path.join(tmp, 'Hermes.app')
  const broker = path.join(tmp, BROKER_APP_NAME)
  fs.mkdirSync(path.join(broker, 'Contents', 'MacOS'), { recursive: true })
  fs.writeFileSync(path.join(broker, 'Contents', 'MacOS', BROKER_EXECUTABLE_NAME), 'binary')

  const installed = installBrokerIntoApp({ appBundlePath: app, brokerBundle: broker })

  assert.equal(installed, path.join(brokerLoginItemsDir(app), BROKER_APP_NAME))
  assert.equal(fs.readFileSync(path.join(installed, 'Contents', 'MacOS', BROKER_EXECUTABLE_NAME), 'utf8'), 'binary')
})

test('stageMacPermissionBroker builds and installs when platform is darwin', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-broker-stage-'))
  const desktopRoot = path.join(tmp, 'desktop')
  const source = path.join(desktopRoot, 'macos', 'HermesMacBroker', 'Sources', 'HermesMacBroker')
  fs.mkdirSync(source, { recursive: true })
  fs.writeFileSync(path.join(source, 'main.swift'), 'print("stub")')

  const calls = []
  const execFile = (command, args) => {
    calls.push([command, args])
    const output = args[args.indexOf('-o') + 1]
    fs.writeFileSync(output, '#!/bin/sh\necho stub\n')
  }

  const context = {
    electronPlatformName: 'darwin',
    appOutDir: path.join(tmp, 'release', 'mac-arm64'),
    packager: { appInfo: { productFilename: 'Hermes' } }
  }
  fs.mkdirSync(path.join(context.appOutDir, 'Hermes.app'), { recursive: true })

  const result = await stageMacPermissionBroker(context, { desktopRoot, execFile, env: {} })

  assert.match(result.installed, /Hermes\.app\/Contents\/Library\/LoginItems\/HermesMacBroker\.app$/)
  assert.equal(calls[0][0], 'swiftc')
  assert.equal(fs.existsSync(path.join(result.installed, 'Contents', 'Info.plist')), true)
})
