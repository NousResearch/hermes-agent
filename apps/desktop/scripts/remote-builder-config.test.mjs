import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { test } from 'vitest'

import { electronBundleDefines } from './bundle-electron-main-config.mjs'

const require = createRequire(import.meta.url)

test('remote builder has a separate app identity and never packages the local install stamp', () => {
  const normal = require('../package.json')
  const remote = require('../electron-builder.remote.cjs')

  assert.notEqual(remote.appId, normal.build.appId)
  assert.equal(remote.appId, 'com.nousresearch.hermes.remote')
  assert.equal(remote.productName, 'Hermes Remote')
  assert.equal(remote.executableName, 'hermes-remote')
  assert.equal(remote.extraMetadata.name, 'hermes-remote')
  assert.equal(remote.extraMetadata.productName, 'Hermes Remote')
  assert.equal(remote.directories.output, 'release/remote')
  assert.deepEqual(remote.linux.target, ['AppImage', 'flatpak'])
  assert.deepEqual(remote.mac.target, normal.build.mac.target)
  assert.equal(remote.mac.extendInfo.CFBundleDisplayName, 'Hermes Remote')
  assert.equal(remote.mac.extendInfo.CFBundleName, 'Hermes Remote')
  assert.equal(remote.mac.extendInfo.CFBundleExecutable, 'hermes-remote')
  assert.deepEqual(remote.win.target, normal.build.win.target)
  assert.equal(remote.dmg.title, 'Install Hermes Remote')
  assert.equal(remote.nsis.shortcutName, 'Hermes Remote')
  assert.equal(remote.nsis.uninstallDisplayName, 'Hermes Remote')
  assert.equal(remote.protocols, undefined)
  assert.equal(
    remote.extraResources.some(resource => resource.to === 'install-stamp.json'),
    false
  )
  const forbiddenPayload = /(?:^|[\\/])(?:agent|hermes-agent|venv|bootstrap)(?:[\\/]|$)|install-stamp/i

  assert.equal(
    remote.files.some(pattern => forbiddenPayload.test(String(pattern))),
    false
  )
  assert.equal(
    remote.extraResources.some(resource =>
      forbiddenPayload.test(`${String(resource.from || '')}/${String(resource.to || '')}`)
    ),
    false
  )
  assert.deepEqual(remote.files, normal.build.files)
  assert.equal(normal.build.appId, 'com.nousresearch.hermes')
  assert.equal(normal.build.directories.output, 'release')
})

test('production Electron bundles bake mutually exclusive remote-only identities', () => {
  const normal = electronBundleDefines({ isDev: false, isRemoteOnly: false })
  const remote = electronBundleDefines({ isDev: false, isRemoteOnly: true })

  assert.equal(normal['process.env.HERMES_DESKTOP_IS_PACKAGED'], 'true')
  assert.equal(normal['process.env.HERMES_DESKTOP_REMOTE_ONLY'], '"0"')
  assert.equal(remote['process.env.HERMES_DESKTOP_IS_PACKAGED'], 'true')
  assert.equal(remote['process.env.HERMES_DESKTOP_REMOTE_ONLY'], '"1"')
  assert.notEqual(normal['process.env.HERMES_DESKTOP_REMOTE_ONLY'], remote['process.env.HERMES_DESKTOP_REMOTE_ONLY'])
})
