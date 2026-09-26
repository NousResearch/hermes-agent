import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, test } from 'vitest'

import {
  PTY_SPAWN_HELPER_ENV,
  PTY_SPAWN_HELPER_MODE,
  resolveSpawnHelperSource,
  stageExternalSpawnHelper
} from './pty-spawn-helper-stage'

function tempRoot(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'pty-helper-stage-'))
}

function makeNodePtyFixture(root: string, helperContent = 'fake spawn-helper'): string {
  const nodePtyRoot = path.join(root, 'app.asar.unpacked', 'dist', 'node_modules', 'node-pty')
  const prebuildDir = path.join(nodePtyRoot, 'prebuilds', 'darwin-arm64')
  fs.mkdirSync(prebuildDir, { recursive: true })
  fs.writeFileSync(path.join(prebuildDir, 'spawn-helper'), helperContent)
  fs.chmodSync(path.join(prebuildDir, 'spawn-helper'), 0o755)

  return nodePtyRoot
}

describe('resolveSpawnHelperSource', () => {
  test('finds the staged helper under an asar-unpacked root', () => {
    const root = tempRoot()
    try {
      const nodePtyRoot = makeNodePtyFixture(root)

      expect(resolveSpawnHelperSource(nodePtyRoot, fs)).toBe(
        path.join(nodePtyRoot, 'prebuilds', 'darwin-arm64', 'spawn-helper')
      )
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('normalizes an app.asar-archived root to the unpacked tree', () => {
    const root = tempRoot()
    try {
      const unpackedRoot = makeNodePtyFixture(root)
      const archivedRoot = unpackedRoot.replace('app.asar.unpacked', 'app.asar')

      expect(resolveSpawnHelperSource(archivedRoot, fs)).toBe(
        path.join(unpackedRoot, 'prebuilds', 'darwin-arm64', 'spawn-helper')
      )
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('returns null when no helper is staged', () => {
    const root = tempRoot()
    try {
      expect(resolveSpawnHelperSource(path.join(root, 'node-pty'), fs)).toBeNull()
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
})

describe('stageExternalSpawnHelper', () => {
  const savedEnv = process.env[PTY_SPAWN_HELPER_ENV]

  afterEach(() => {
    if (savedEnv === undefined) {
      delete process.env[PTY_SPAWN_HELPER_ENV]
    } else {
      process.env[PTY_SPAWN_HELPER_ENV] = savedEnv
    }
  })

  test('copies the helper outside the bundle, chmods 0755, and exports the env var', () => {
    const root = tempRoot()
    try {
      const nodePtyRoot = makeNodePtyFixture(root)
      const env: Record<string, string | undefined> = {}

      const result = stageExternalSpawnHelper({
        nodePtyRoot,
        destDir: path.join(root, 'userData', 'bin'),
        env
      })

      const dest = path.join(root, 'userData', 'bin', 'spawn-helper')
      expect(result.staged).toBe(dest)
      expect(result.reused).toBe(false)
      expect(result.errors).toEqual([])
      expect(fs.readFileSync(dest, 'utf8')).toBe('fake spawn-helper')
      expect(fs.statSync(dest).mode & 0o777).toBe(PTY_SPAWN_HELPER_MODE)
      expect(env[PTY_SPAWN_HELPER_ENV]).toBe(dest)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('reuses an up-to-date copy instead of rewriting it', () => {
    const root = tempRoot()
    try {
      const nodePtyRoot = makeNodePtyFixture(root)
      const env: Record<string, string | undefined> = {}

      stageExternalSpawnHelper({ nodePtyRoot, destDir: path.join(root, 'bin'), env })
      const first = fs.statSync(path.join(root, 'bin', 'spawn-helper')).mtimeMs

      const result = stageExternalSpawnHelper({ nodePtyRoot, destDir: path.join(root, 'bin'), env })

      expect(result.reused).toBe(true)
      expect(fs.statSync(path.join(root, 'bin', 'spawn-helper')).mtimeMs).toBe(first)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('refreshes a stale copy when the staged helper changed', () => {
    const root = tempRoot()
    try {
      const nodePtyRoot = makeNodePtyFixture(root)
      const env: Record<string, string | undefined> = {}
      const destDir = path.join(root, 'bin')

      stageExternalSpawnHelper({ nodePtyRoot, destDir, env })
      fs.writeFileSync(path.join(nodePtyRoot, 'prebuilds', 'darwin-arm64', 'spawn-helper'), 'rebuilt helper')

      const result = stageExternalSpawnHelper({ nodePtyRoot, destDir, env })

      expect(result.reused).toBe(false)
      expect(fs.readFileSync(path.join(destDir, 'spawn-helper'), 'utf8')).toBe('rebuilt helper')
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('non-darwin platforms are a no-op', () => {
    const root = tempRoot()
    try {
      const nodePtyRoot = makeNodePtyFixture(root)
      const env: Record<string, string | undefined> = {}

      const result = stageExternalSpawnHelper({ nodePtyRoot, destDir: path.join(root, 'bin'), platform: 'linux', env })

      expect(result.staged).toBeNull()
      expect(env[PTY_SPAWN_HELPER_ENV]).toBeUndefined()
      expect(fs.existsSync(path.join(root, 'bin'))).toBe(false)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('a missing helper fails soft without exporting the env var', () => {
    const root = tempRoot()
    try {
      const env: Record<string, string | undefined> = {}

      const result = stageExternalSpawnHelper({ nodePtyRoot: path.join(root, 'node-pty'), destDir: path.join(root, 'bin'), env })

      expect(result.staged).toBeNull()
      expect(result.sourcePath).toBeNull()
      expect(result.errors).toHaveLength(1)
      expect(env[PTY_SPAWN_HELPER_ENV]).toBeUndefined()
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  test('a pre-existing env override is left untouched when staging fails', () => {
    const root = tempRoot()
    try {
      const env: Record<string, string | undefined> = { [PTY_SPAWN_HELPER_ENV]: '/stale/leftover' }

      stageExternalSpawnHelper({ nodePtyRoot: path.join(root, 'node-pty'), destDir: path.join(root, 'bin'), env })

      // A stale override is harmless: the patched unixTerminal.js only honors
      // it when fs.existsSync passes, otherwise it uses the default path.
      expect(env[PTY_SPAWN_HELPER_ENV]).toBe('/stale/leftover')
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })
})
