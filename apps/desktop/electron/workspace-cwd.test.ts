/**
 * Tests for electron/workspace-cwd.ts.
 *
 * Run with: node --test electron/workspace-cwd.test.ts
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { isPackagedInstallPath, resolveWorkspaceCwd } from './workspace-cwd'

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

test('cwd precedence retains a terminal hint only when it wins the validated ladder', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-launch-cwd-'))
  const home = fs.mkdirSync(path.join(root, 'home'), { recursive: true })!
  const project = fs.mkdirSync(path.join(root, 'project'), { recursive: true })!
  const configured = fs.mkdirSync(path.join(root, 'configured'), { recursive: true })!
  const accepts = (cwd: string) => fs.existsSync(cwd) && fs.statSync(cwd).isDirectory()

  try {
    assert.deepEqual(resolveWorkspaceCwd([{ cwd: project, source: 'launch' }], home, accepts), {
      cwd: project,
      launchCwd: project
    })
    assert.deepEqual(resolveWorkspaceCwd([{ cwd: configured }, { cwd: project, source: 'launch' }], home, accepts), {
      cwd: configured,
      launchCwd: ''
    })
    assert.deepEqual(resolveWorkspaceCwd([{ cwd: home }], home, accepts), { cwd: home, launchCwd: '' })
    // An explicit terminal launch from home differs from an implicit home fallback.
    assert.deepEqual(resolveWorkspaceCwd([{ cwd: home, source: 'launch' }], home, accepts), {
      cwd: home,
      launchCwd: home
    })
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('missing, invalid and packaged-install launch hints fall back without attaching a workspace', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-launch-cwd-'))
  const home = path.join(root, 'home')
  const install = path.join(root, 'install')
  const file = path.join(root, 'file')
  fs.mkdirSync(home)
  fs.mkdirSync(install)
  fs.writeFileSync(file, '')

  const accepts = (cwd: string) =>
    !isPackagedInstallPath(cwd, { isPackaged: true, installRoots: [install] }) &&
    fs.existsSync(cwd) &&
    fs.statSync(cwd).isDirectory()

  try {
    for (const cwd of [undefined, '', path.join(root, 'missing'), file, install]) {
      assert.deepEqual(resolveWorkspaceCwd([{ cwd, source: 'launch' }, { cwd: home }], home, accepts), {
        cwd: home,
        launchCwd: ''
      })
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
