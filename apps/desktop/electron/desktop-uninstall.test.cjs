'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  UNINSTALL_MODES,
  assertAppUninstallMode,
  buildAppOnlyCleanupScript,
  resolveRemovableAppPath,
  shouldRemoveAppBundle
} = require('./desktop-uninstall.cjs')

test('only app-only uninstall mode is supported', () => {
  assert.deepEqual(UNINSTALL_MODES, ['app'])
  assert.doesNotThrow(() => assertAppUninstallMode('app'))
  assert.throws(() => assertAppUninstallMode('gui'), /Unknown uninstall mode/)
  assert.throws(() => assertAppUninstallMode('lite'), /Unknown uninstall mode/)
  assert.throws(() => assertAppUninstallMode('full'), /Unknown uninstall mode/)
})

test('resolveRemovableAppPath finds the Reuben .app bundle on macOS', () => {
  assert.equal(
    resolveRemovableAppPath('/Applications/Reuben.app/Contents/MacOS/Reuben', 'darwin'),
    '/Applications/Reuben.app'
  )
  assert.equal(resolveRemovableAppPath('/usr/bin/electron', 'darwin'), null)
})

test('resolveRemovableAppPath finds the Reuben install dir on Windows', () => {
  assert.equal(
    resolveRemovableAppPath('C:\\Users\\x\\AppData\\Local\\Programs\\Reuben\\Reuben.exe', 'win32'),
    'C:\\Users\\x\\AppData\\Local\\Programs\\Reuben'
  )
  assert.equal(resolveRemovableAppPath('C:\\Temp\\Hermes\\Hermes.exe', 'win32'), null)
})

test('resolveRemovableAppPath handles Linux AppImage and unpacked dirs', () => {
  assert.equal(
    resolveRemovableAppPath('/tmp/.mount_ReubenXXXX/reuben', 'linux', { APPIMAGE: '/home/x/Apps/Reuben.AppImage' }),
    '/home/x/Apps/Reuben.AppImage'
  )
  assert.equal(resolveRemovableAppPath('/opt/reuben/linux-unpacked/reuben', 'linux', {}), '/opt/reuben/linux-unpacked')
  assert.equal(resolveRemovableAppPath('/usr/bin/reuben', 'linux', {}), null)
})

test('shouldRemoveAppBundle requires packaged app and resolved path', () => {
  assert.equal(shouldRemoveAppBundle(true, '/Applications/Reuben.app'), true)
  assert.equal(shouldRemoveAppBundle(false, '/Applications/Reuben.app'), false)
  assert.equal(shouldRemoveAppBundle(true, null), false)
})

test('buildAppOnlyCleanupScript creates POSIX app-bundle cleanup without backend commands', () => {
  const script = buildAppOnlyCleanupScript({
    appPath: '/Applications/Reuben.app',
    desktopPid: 1234,
    platform: 'darwin'
  })
  assert.match(script, /^#!\/bin\/bash/)
  assert.match(script, /pid=1234/)
  assert.match(script, /rm -rf '\/Applications\/Reuben\.app'/)
  assert.doesNotMatch(script, /HERMES_HOME/)
  assert.doesNotMatch(script, /hermes_cli/)
})

test('buildAppOnlyCleanupScript creates Windows app-dir cleanup without backend commands', () => {
  const script = buildAppOnlyCleanupScript({
    appPath: 'C:\\Users\\x\\AppData\\Local\\Programs\\Reuben',
    desktopPid: 9988,
    platform: 'win32'
  })
  assert.match(script, /@echo off/)
  assert.match(script, /set "PID=9988"/)
  assert.match(script, /rmdir \/s \/q "C:\\Users\\x\\AppData\\Local\\Programs\\Reuben"/)
  assert.doesNotMatch(script, /HERMES_HOME/)
  assert.doesNotMatch(script, /hermes_cli/)
})
