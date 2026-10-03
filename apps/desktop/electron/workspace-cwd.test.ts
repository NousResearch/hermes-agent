/**
 * Tests for electron/workspace-cwd.ts.
 *
 * Run with: node --test electron/workspace-cwd.test.ts
 */

import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { isPackagedInstallPath, resolveTerminalCwd } from './workspace-cwd'

const installRoot = path.resolve('/opt/Hermes')
const sourceRoot = path.resolve('/home/user/checkouts/hermes-agent')
const homeDir = path.resolve('/home/user')

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

// #121259: a source-run desktop pins TERMINAL_CWD to the repo root, so the
// agent's install-tree context guard (cwd is None first conjunct) never runs
// and the contributor AGENTS.md loads into every session.
test('resolveTerminalCwd pins home when the spawn cwd is the dev-run source-root fallback', () => {
  assert.equal(
    resolveTerminalCwd(sourceRoot, { chosenCwd: null, homeDir, isPackaged: false, sourceRoot }),
    homeDir
  )
})

test('resolveTerminalCwd pins home for a subdirectory of the source tree without an explicit choice', () => {
  const subdir = path.join(sourceRoot, 'apps', 'desktop')

  assert.equal(
    resolveTerminalCwd(subdir, { chosenCwd: null, homeDir, isPackaged: false, sourceRoot }),
    homeDir
  )
})

test('resolveTerminalCwd honors a deliberately chosen source-tree workspace', () => {
  assert.equal(
    resolveTerminalCwd(sourceRoot, { chosenCwd: sourceRoot, homeDir, isPackaged: false, sourceRoot }),
    sourceRoot
  )
})

test('resolveTerminalCwd keeps a non-tree spawn cwd without an explicit choice', () => {
  const project = path.join(homeDir, 'projects', 'demo')

  assert.equal(
    resolveTerminalCwd(project, { chosenCwd: null, homeDir, isPackaged: false, sourceRoot }),
    project
  )
})

test('resolveTerminalCwd leaves packaged spawns untouched', () => {
  assert.equal(
    resolveTerminalCwd(sourceRoot, { chosenCwd: null, homeDir, isPackaged: true, sourceRoot }),
    sourceRoot
  )
})
