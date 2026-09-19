import assert from 'node:assert/strict'
import { join } from 'node:path'

import { test } from 'vitest'

import {
  ensureSpawnHelperExecutable,
  needsExecBit,
  spawnHelperCandidates,
  type SpawnHelperFs,
  withExecBits,
  writableNodePtyRoot
} from './spawn-helper-perms'

interface FakeFile {
  mode: number
  statThrows?: boolean
  chmodThrows?: boolean
}

function fakeFs(
  files: Record<string, FakeFile>,
  dirs: Record<string, string[]> = {}
): SpawnHelperFs & { chmods: { path: string; mode: number }[] } {
  const chmods: { path: string; mode: number }[] = []

  return {
    chmods,
    existsSync(path) {
      return path in files || path in dirs
    },
    readdirSync(path) {
      return dirs[path] ?? []
    },
    statSync(path) {
      const file = files[path]

      if (!file || file.statThrows) {
        throw new Error(`stat failed: ${path}`)
      }

      return { mode: file.mode }
    },
    chmodSync(path, mode) {
      const file = files[path]

      if (file?.chmodThrows) {
        throw new Error(`chmod failed: ${path}`)
      }

      chmods.push({ path, mode })

      if (file) {
        file.mode = mode
      }
    }
  }
}

test('rewrites an archived node-pty root to the matching unpacked tree exactly once', () => {
  assert.equal(
    writableNodePtyRoot('/Hermes.app/Contents/Resources/app.asar/dist/node_modules/node-pty'),
    '/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/node-pty'
  )
  assert.equal(
    writableNodePtyRoot('/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/node-pty'),
    '/Hermes.app/Contents/Resources/app.asar.unpacked/dist/node_modules/node-pty'
  )
})

test('uses the unpacked helper when resolution reports an app.asar node-pty root', () => {
  const archivedRoot = '/Hermes.app/Contents/Resources/app.asar/dist/node_modules/node-pty'
  const unpackedRoot = writableNodePtyRoot(archivedRoot)
  const helper = join(unpackedRoot, 'prebuilds', 'darwin-arm64', 'spawn-helper')
  const fs = fakeFs({ [helper]: { mode: 0o644 } }, { [join(unpackedRoot, 'prebuilds')]: ['darwin-arm64'] })

  const result = ensureSpawnHelperExecutable(archivedRoot, fs)

  assert.deepEqual(result.fixed, [helper])
  assert.deepEqual(result.errors, [])
  assert.deepEqual(fs.chmods, [{ path: helper, mode: 0o755 }])
})

test('needsExecBit / withExecBits treat any missing exec bit as non-executable', () => {
  assert.equal(needsExecBit(0o644), true)
  assert.equal(needsExecBit(0o755), false)
  // Partial exec bits (owner only) still count as needing repair.
  assert.equal(needsExecBit(0o744), true)
  // Preserves read/write bits while adding exec for all three classes.
  assert.equal(withExecBits(0o644), 0o755)
  assert.equal(withExecBits(0o600), 0o711)
})

test('candidates cover every prebuild dir plus build/Release', () => {
  const root = '/pkg/node-pty'

  const fs = fakeFs({}, { [join(root, 'prebuilds')]: ['darwin-arm64', 'darwin-x64', 'linux-x64'] })

  assert.deepEqual(spawnHelperCandidates(root, fs), [
    join(root, 'prebuilds', 'darwin-arm64', 'spawn-helper'),
    join(root, 'prebuilds', 'darwin-x64', 'spawn-helper'),
    join(root, 'prebuilds', 'linux-x64', 'spawn-helper'),
    join(root, 'build', 'Release', 'spawn-helper')
  ])
})

test('chmods only the non-executable spawn-helpers, leaving 0755 copies alone', () => {
  const root = '/pkg/node-pty'
  const arm = join(root, 'prebuilds', 'darwin-arm64', 'spawn-helper')
  const x64 = join(root, 'prebuilds', 'darwin-x64', 'spawn-helper')

  const fs = fakeFs(
    {
      [arm]: { mode: 0o644 },
      [x64]: { mode: 0o755 }
    },
    { [join(root, 'prebuilds')]: ['darwin-arm64', 'darwin-x64'] }
  )

  const result = ensureSpawnHelperExecutable(root, fs)

  assert.deepEqual(result.fixed, [arm])
  assert.deepEqual(result.errors, [])
  assert.deepEqual(fs.chmods, [{ path: arm, mode: 0o755 }])
})

test('missing spawn-helpers are skipped without error', () => {
  const root = '/pkg/node-pty'
  const fs = fakeFs({}, { [join(root, 'prebuilds')]: ['darwin-arm64'] })

  const result = ensureSpawnHelperExecutable(root, fs)

  assert.deepEqual(result.fixed, [])
  assert.deepEqual(result.errors, [])
  assert.deepEqual(fs.chmods, [])
})

test('chmod failures are collected, not thrown', () => {
  const root = '/pkg/node-pty'
  const arm = join(root, 'prebuilds', 'darwin-arm64', 'spawn-helper')

  const fs = fakeFs({ [arm]: { mode: 0o644, chmodThrows: true } }, { [join(root, 'prebuilds')]: ['darwin-arm64'] })

  const result = ensureSpawnHelperExecutable(root, fs)

  assert.deepEqual(result.fixed, [])
  assert.equal(result.errors.length, 1)
  assert.equal(result.errors[0].path, arm)
})

test('no prebuilds dir (Windows layout) is a clean no-op', () => {
  const root = '/pkg/node-pty'
  const fs = fakeFs({}, {})

  const result = ensureSpawnHelperExecutable(root, fs)

  assert.deepEqual(result.fixed, [])
  assert.deepEqual(result.errors, [])
})
