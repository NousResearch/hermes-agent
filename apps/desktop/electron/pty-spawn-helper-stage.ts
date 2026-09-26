// macOS 26 hardened-runtime fix for the packaged desktop app (#63784).
//
// node-pty 1.1.0's darwin `pty.fork` uses posix_spawn with POSIX_SPAWN_SETSID
// to exec `spawn-helper`. On macOS 26 the hardened runtime rejects that spawn
// when the helper binary lives inside the sealed .app bundle subtree — even at
// mode 0755 — so the packaged terminal dies with `posix_spawnp failed.`
// (Demonstrated in the #63784 repro: `pty.spawn` succeeds with the helper
// outside the bundle and fails inside.) stage-native-deps.mjs patches the
// staged unixTerminal.js to honor HERMES_NODE_PTY_SPAWN_HELPER, so at first
// terminal start the main process copies the staged helper to a short path in
// the user's Application Support tree (outside the bundle), chmods it 0755,
// and exports the env var node-pty reads when its module first loads.
//
// The copy is done via read+write rather than copyFileSync so filesystem
// xattrs (e.g. com.apple.quarantine propagated from an installed app) do not
// ride along. Everything is best-effort and fail-soft: any failure leaves the
// env var unset and node-pty falls back to its default in-bundle path, which
// is exactly the pre-fix behavior (worse case, no regression). Dev flow is
// unaffected: the env override is only injected into the *staged* unixTerminal
// copy, and raw node_modules keeps the #66734 lazy-chmod repair.

import fs from 'node:fs'
import path from 'node:path'

import { spawnHelperCandidates, writableNodePtyRoot } from './spawn-helper-perms'

export const PTY_SPAWN_HELPER_ENV = 'HERMES_NODE_PTY_SPAWN_HELPER'
export const PTY_SPAWN_HELPER_MODE = 0o755

export interface StageExternalHelperFs {
  existsSync(path: string): boolean
  readdirSync(path: string): string[]
  readFileSync(path: string): Buffer
  writeFileSync(path: string, data: Buffer): void
  chmodSync(path: string, mode: number): void
  statSync(path: string): { size: number }
  mkdirSync(path: string, options: { recursive: true }): void
}

export interface StageExternalHelperOptions {
  nodePtyRoot: string
  destDir: string
  platform?: NodeJS.Platform
  env?: Record<string, string | undefined>
  fs?: StageExternalHelperFs
}

export interface StageExternalHelperResult {
  /** Destination helper path when one is staged and exported via the env var. */
  staged: null | string
  /** Staged in-bundle helper the copy came from (null when nothing found). */
  sourcePath: null | string
  /** True when an existing destination copy was reused without rewriting. */
  reused: boolean
  /** Collected non-fatal failures; empty on success. */
  errors: string[]
}

const defaultFs: StageExternalHelperFs = {
  existsSync: p => fs.existsSync(p),
  readdirSync: p => fs.readdirSync(p),
  readFileSync: p => fs.readFileSync(p),
  writeFileSync: (p, data) => fs.writeFileSync(p, data),
  chmodSync: (p, mode) => fs.chmodSync(p, mode),
  statSync: p => fs.statSync(p),
  mkdirSync: (p, options) => fs.mkdirSync(p, options)
}

// First existing spawn-helper under the packaged (asar-unpacked) node-pty tree.
export function resolveSpawnHelperSource(
  nodePtyRoot: string,
  fsDeps: Pick<StageExternalHelperFs, 'existsSync' | 'readdirSync'>
): null | string {
  const writableRoot = writableNodePtyRoot(nodePtyRoot)

  for (const candidate of spawnHelperCandidates(writableRoot, fsDeps)) {
    if (fsDeps.existsSync(candidate)) {
      return candidate
    }
  }

  return null
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/**
 * Stage the packaged node-pty spawn-helper OUTSIDE the sealed .app bundle
 * subtree (destDir, e.g. `<userData>/bin`) and export `PTY_SPAWN_HELPER_ENV`
 * pointing at it. macOS-only; every other platform (and every failure) is a
 * no-op that leaves `env` untouched so node-pty uses its default path.
 */
export function stageExternalSpawnHelper({
  nodePtyRoot,
  destDir,
  platform = process.platform,
  env = process.env,
  fs: fsDeps = defaultFs
}: StageExternalHelperOptions): StageExternalHelperResult {
  const result: StageExternalHelperResult = { staged: null, sourcePath: null, reused: false, errors: [] }

  if (platform !== 'darwin') {
    return result
  }

  try {
    const sourcePath = resolveSpawnHelperSource(nodePtyRoot, fsDeps)
    if (!sourcePath) {
      result.errors.push(`no spawn-helper found under ${nodePtyRoot}`)
      return result
    }

    result.sourcePath = sourcePath
    const destPath = path.join(destDir, 'spawn-helper')
    const source = fsDeps.readFileSync(sourcePath)

    if (fsDeps.existsSync(destPath)) {
      try {
        if (fsDeps.statSync(destPath).size === source.length) {
          result.staged = destPath
          result.reused = true
        }
      } catch (error) {
        result.errors.push(`stat ${destPath}: ${errorMessage(error)}`)
      }
    }

    if (!result.staged) {
      fsDeps.mkdirSync(destDir, { recursive: true })
      fsDeps.writeFileSync(destPath, source)
      fsDeps.chmodSync(destPath, PTY_SPAWN_HELPER_MODE)
      result.staged = destPath
    }

    env[PTY_SPAWN_HELPER_ENV] = result.staged
  } catch (error) {
    result.staged = null
    result.errors.push(errorMessage(error))
    delete env[PTY_SPAWN_HELPER_ENV]
  }

  return result
}
