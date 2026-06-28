/**
 * Tests for electron/workspace-cwd.cjs.
 *
 * Run with: node --test electron/workspace-cwd.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { isPackagedInstallPath, resolveWorkspaceCwd } = require('./workspace-cwd.cjs')

const installRoot = path.resolve('/opt/Hermes')

test('isPackagedInstallPath returns false when not packaged', () => {
  assert.equal(isPackagedInstallPath(installRoot, { isPackaged: false, installRoots: [installRoot] }), false)
})

test('isPackagedInstallPath flags the install root itself', () => {
  assert.equal(isPackagedInstallPath(installRoot, { isPackaged: true, installRoots: [installRoot] }), true)
})

test('isPackagedInstallPath flags paths nested under the install root', () => {
  const nested = path.join(installRoot, 'resources', 'app.asar')

  assert.equal(isPackagedInstallPath(nested, { isPackaged: true, installRoots: [installRoot] }), true)
})

test('isPackagedInstallPath ignores paths outside the install root', () => {
  const homeProject = path.resolve('/home/user/projects/demo')

  assert.equal(isPackagedInstallPath(homeProject, { isPackaged: true, installRoots: [installRoot] }), false)
})

test('resolveWorkspaceCwd prefers explicit launch cwd over saved default project dir', () => {
  const launchCwd = path.resolve('/home/user/Math')
  const savedDefault = path.resolve('/home/user/408')
  const existing = new Set([launchCwd, savedDefault, path.resolve('/home/user')])

  assert.equal(
    resolveWorkspaceCwd({
      explicitCwd: launchCwd,
      defaultProjectDir: savedDefault,
      homeDir: path.resolve('/home/user'),
      isPackaged: true,
      installRoots: [installRoot],
      directoryExists: dir => existing.has(dir)
    }),
    launchCwd
  )
})

test('resolveWorkspaceCwd falls back to saved default project dir without explicit launch cwd', () => {
  const savedDefault = path.resolve('/home/user/408')
  const existing = new Set([savedDefault, path.resolve('/home/user')])

  assert.equal(
    resolveWorkspaceCwd({
      explicitCwd: '',
      defaultProjectDir: savedDefault,
      homeDir: path.resolve('/home/user'),
      isPackaged: true,
      installRoots: [installRoot],
      directoryExists: dir => existing.has(dir)
    }),
    savedDefault
  )
})

test('resolveWorkspaceCwd skips packaged install paths', () => {
  const launchCwd = path.join(installRoot, 'resources')
  const home = path.resolve('/home/user')
  const existing = new Set([path.resolve(launchCwd), home])

  assert.equal(
    resolveWorkspaceCwd({
      explicitCwd: launchCwd,
      homeDir: home,
      isPackaged: true,
      installRoots: [installRoot],
      directoryExists: dir => existing.has(dir)
    }),
    home
  )
})
