/**
 * Tests for electron/workspace-cwd.cjs.
 *
 * Run with: node --test electron/workspace-cwd.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('node:path')

const { isPackagedInstallPath } = require('./workspace-cwd.cjs')

const installRoot = path.resolve('/opt/Hermes')

test('isPackagedInstallPath returns false when not packaged', () => {
  assert.equal(
    isPackagedInstallPath(installRoot, { isPackaged: false, installRoots: [installRoot] }),
    false
  )
})

test('isPackagedInstallPath flags the install root itself', () => {
  assert.equal(
    isPackagedInstallPath(installRoot, { isPackaged: true, installRoots: [installRoot] }),
    true
  )
})

test('isPackagedInstallPath flags paths nested under the install root', () => {
  const nested = path.join(installRoot, 'resources', 'app.asar')

  assert.equal(
    isPackagedInstallPath(nested, { isPackaged: true, installRoots: [installRoot] }),
    true
  )
})

test('isPackagedInstallPath ignores paths outside the install root', () => {
  const homeProject = path.resolve('/home/user/projects/demo')

  assert.equal(
    isPackagedInstallPath(homeProject, { isPackaged: true, installRoots: [installRoot] }),
    false
  )
})
