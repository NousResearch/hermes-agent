import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { resolveWindowsGit } from './resolve-windows-git'

const roots: string[] = []
afterEach(() => roots.splice(0).forEach(root => rmSync(root, { recursive: true, force: true })))

function fixture() {
  const root = mkdtempSync(path.join(tmpdir(), 'hermes-managed-git-'))
  roots.push(root)
  const store = path.join(root, 'hermes', 'tools')
  mkdirSync(store, { recursive: true })
  const env = {
    LOCALAPPDATA: root,
    ProgramFiles: path.join(root, 'system'),
    'ProgramFiles(x86)': path.join(root, 'x86')
  }
  const touch = (...parts: string[]) => {
    const name = path.join(root, ...parts)
    mkdirSync(path.dirname(name), { recursive: true })
    writeFileSync(name, '')
    return name
  }
  const facts = (value: unknown) => writeFileSync(path.join(store, 'facts.json'), JSON.stringify(value))
  const resolve = (onPath: string | null = null) =>
    resolveWindowsGit({ env, fileExists: existsSync, findOnPath: () => onPath })
  return { root, store, env, touch, facts, resolve }
}

test.each(['cmd', 'bin'])('bootstrap-only Git uses the PM-selected entry (%s)', folder => {
  const f = fixture()
  const selected = f.touch('hermes', 'tools', 'git-selected-win32-x64', folder, 'git.exe')
  f.touch('hermes', 'tools', 'git-newer-but-unselected', 'cmd', 'git.exe')
  f.facts({ schema: 1, packages: { git: { entry: 'git-selected-win32-x64' } } })
  assert.equal(f.resolve(), selected)
})

test('legacy PortableGit retains precedence over the PM store and system install', () => {
  const f = fixture()
  const legacy = f.touch('hermes', 'git', 'cmd', 'git.exe')
  f.touch('hermes', 'tools', 'git-selected', 'cmd', 'git.exe')
  f.touch('system', 'Git', 'cmd', 'git.exe')
  f.facts({ schema: 1, packages: { git: { entry: 'git-selected' } } })
  assert.equal(f.resolve(), legacy)
})

test.each([
  null,
  {},
  { schema: 2, packages: { git: { entry: 'git-selected' } } },
  { schema: 1, packages: { git: { entry: '../escape' } } },
  { schema: 1, packages: { git: { entry: 42 } } },
  { schema: 1, packages: { git: { entry: 'missing-entry' } } }
])('invalid or stale facts preserve system/PATH fallback: %j', value => {
  const f = fixture()
  f.touch('hermes', 'tools', 'git-selected', 'cmd', 'git.exe')
  f.facts(value)
  assert.equal(f.resolve('path-git'), 'path-git')
  const system = f.touch('system', 'Git', 'cmd', 'git.exe')
  assert.equal(f.resolve('path-git'), system)
})

test('missing and unreadable/malformed facts preserve bare git fallback', () => {
  const f = fixture()
  assert.equal(f.resolve(), 'git')
  writeFileSync(path.join(f.store, 'facts.json'), '{')
  assert.equal(f.resolve(), 'git')
  rmSync(path.join(f.store, 'facts.json'))
  mkdirSync(path.join(f.store, 'facts.json'))
  assert.equal(f.resolve(), 'git')
})
