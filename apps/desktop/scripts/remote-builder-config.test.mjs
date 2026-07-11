import assert from 'node:assert/strict'
import test from 'node:test'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const config = require('../electron-builder.remote.cjs')

test('remote package has a distinct identity and only standalone Linux targets', () => {
  assert.equal(config.appId, 'com.nousresearch.hermes.remote')
  assert.equal(config.productName, 'Hermes Remote')
  assert.deepEqual(config.linux.target, ['AppImage', 'flatpak'])
  assert.equal(config.linux.syncDesktopName, true)
  assert.equal(config.protocols, undefined)
})

test('remote package omits local bootstrap resources', () => {
  assert.equal(config.extraResources.some(resource => resource.to === 'install-stamp.json'), false)
})

test('Flatpak can reach gateways and secret storage without broad home access', () => {
  assert.equal(config.flatpak.baseVersion, config.flatpak.runtimeVersion)
  assert.ok(config.flatpak.finishArgs.includes('--share=network'))
  assert.ok(config.flatpak.finishArgs.includes('--talk-name=org.freedesktop.FileManager1'))
  assert.ok(config.flatpak.finishArgs.includes('--talk-name=org.freedesktop.secrets'))
  assert.equal(config.flatpak.finishArgs.some(arg => arg.startsWith('--filesystem=home')), false)
})
