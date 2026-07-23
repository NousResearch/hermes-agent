import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { afterEach, beforeEach, test } from 'vitest'

import { repoRootPkg } from '../scripts/run-electron-builder.mjs'

// PR #2: electron-builder OOM in monorepos. repoRootPkg() must walk
// UP from a given start dir and resolve the MONOREPO ROOT package.json --
// the one carrying `workspaces` -- not apps/desktop (which has no
// `workspaces` field). If it resolved apps/desktop, the workspace
// neutralize would be a no-op and the OOM would persist.

const { tmpdir } = os
let root

beforeEach(() => {
  root = fs.mkdtempSync(path.join(tmpdir(), 'eb-oom-'))
})

afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true })
})

test('resolves the monorepo root carrying workspaces, not apps/desktop', () => {
  // monorepo root package.json WITH workspaces
  fs.writeFileSync(
    path.join(root, 'package.json'),
    JSON.stringify({ name: 'hermes-agent', version: '1.0.0', workspaces: ['apps/*'] })
  )
  // apps/desktop has NO workspaces field
  const appDir = path.join(root, 'apps', 'desktop')
  fs.mkdirSync(path.join(appDir, 'scripts'), { recursive: true })
  fs.writeFileSync(
    path.join(appDir, 'package.json'),
    JSON.stringify({ name: 'desktop', version: '1.0.0' })
  )
  // Start the walk from apps/desktop/scripts (what the real script does
  // via its own import.meta.url).
  const startDir = path.join(appDir, 'scripts')
  const resolved = repoRootPkg(startDir)
  // resolved path must be the monorepo root, not apps/desktop
  assert.strictEqual(
    path.dirname(resolved),
    root,
    `expected ${root}, got ${path.dirname(resolved)}`
  )
  const json = JSON.parse(fs.readFileSync(resolved, 'utf8'))
  assert.ok(json.workspaces, 'resolved package.json must carry workspaces')
})

test('falls back to monorepo root when no workspaces root exists', () => {
  // No workspaces anywhere: should not throw and should return a path.
  fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'plain' }))
  const startDir = path.join(root, 'apps', 'desktop', 'scripts')
  fs.mkdirSync(startDir, { recursive: true })
  // real repo has apps/desktop/package.json; fallback walks past it
  fs.writeFileSync(path.join(root, 'apps', 'desktop', 'package.json'), JSON.stringify({ name: 'desktop' }))
  const resolved = repoRootPkg(startDir)
  assert.ok(resolved, 'repoRootPkg must return a path even without workspaces')
  // fallback resolves to the monorepo root (3 levels up from scripts)
  assert.strictEqual(path.dirname(resolved), root, `expected ${root}, got ${path.dirname(resolved)}`)
  assert.ok(fs.existsSync(resolved), 'resolved path must exist')
})
