/**
 * Tests for electron/workspace-cwd.ts.
 *
 * Run with: node --test electron/workspace-cwd.test.ts
 */

import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { cwdCandidates, isPackagedInstallPath } from './workspace-cwd'

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

test('cwdCandidates does not select source-run directories without an explicit preference', () => {
  assert.deepEqual(
    cwdCandidates({
      defaultProjectDir: null,
      desktopCwd: null,
      homeDir: '/home/user'
    }),
    ['/home/user']
  )
})

test('cwdCandidates keeps configured directories ahead of the home fallback', () => {
  assert.deepEqual(
    cwdCandidates({
      defaultProjectDir: '/projects/saved',
      desktopCwd: '/projects/explicit',
      homeDir: '/home/user'
    }),
    ['/projects/saved', '/projects/explicit', '/home/user']
  )
})

test('configured packaged install paths remain detectable for fallback', () => {
  const [candidate] = cwdCandidates({
    defaultProjectDir: installRoot,
    desktopCwd: undefined,
    homeDir: '/home/user'
  })

  assert.equal(
    isPackagedInstallPath(candidate, { isPackaged: true, installRoots: [installRoot] }),
    true
  )
})
