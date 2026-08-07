import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

import { DEFAULT_ELECTRON_MIRROR, electronMirrorEnv, inspectBuildEnvironment } from './build-environment.mjs'

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, JSON.stringify(value))
}

function fixture() {
  const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-build-'))
  writeJson(path.join(repoRoot, 'package.json'), { engines: { node: '>=20.0.0' } })
  writeJson(path.join(repoRoot, 'package-lock.json'), { lockfileVersion: 3 })
  writeJson(path.join(repoRoot, 'node_modules', '.package-lock.json'), { lockfileVersion: 3 })
  writeJson(path.join(repoRoot, 'apps', 'desktop', 'package.json'), {
    build: { electronVersion: '40.10.2' },
    devDependencies: { electron: '40.10.2' }
  })
  for (const [name, version] of [
    ['vite', '8.0.10'],
    ['electron-builder', '26.8.1'],
    ['electron', '40.10.2']
  ]) {
    writeJson(path.join(repoRoot, 'node_modules', name, 'package.json'), { version })
  }
  const electron = path.join(repoRoot, 'node_modules', 'electron', 'dist', 'electron')
  fs.mkdirSync(path.dirname(electron), { recursive: true })
  fs.writeFileSync(electron, '')
  return repoRoot
}

test('electronMirrorEnv preserves an explicit mirror and supplies a default', () => {
  assert.equal(electronMirrorEnv({}).ELECTRON_MIRROR, DEFAULT_ELECTRON_MIRROR)
  assert.equal(
    electronMirrorEnv({ ELECTRON_MIRROR: 'https://mirror.example/' }).ELECTRON_MIRROR,
    'https://mirror.example/'
  )
})

test('inspectBuildEnvironment accepts a complete locked install', () => {
  const repoRoot = fixture()
  assert.deepEqual(inspectBuildEnvironment({ repoRoot, platform: 'linux', nodeVersion: '22.12.0' }), [])
})

test('inspectBuildEnvironment rejects unsupported Node releases', () => {
  const repoRoot = fixture()
  const problems = inspectBuildEnvironment({ repoRoot, platform: 'linux', nodeVersion: '21.7.0' })
  assert.ok(problems.some(problem => problem.cause.includes('Node 21.7.0 is incompatible')))
})

test('inspectBuildEnvironment explains stale and incomplete Electron installs', () => {
  const repoRoot = fixture()
  writeJson(path.join(repoRoot, 'node_modules', 'electron', 'package.json'), { version: '39.0.0' })
  fs.rmSync(path.join(repoRoot, 'node_modules', 'electron', 'dist'), { recursive: true })

  const problems = inspectBuildEnvironment({ repoRoot, platform: 'linux', nodeVersion: '22.12.0' })
  assert.ok(problems.some(problem => problem.code === 'electron-install-mismatch'))
  assert.ok(problems.some(problem => problem.code === 'electron-binary-missing'))
})
