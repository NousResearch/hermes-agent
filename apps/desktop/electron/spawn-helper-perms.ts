// node-pty ships its POSIX `spawn-helper` inside the published npm tarball with
// mode 0644 (no exec bit). node-pty `posix_spawnp`s that helper on macOS/Linux,
// so a non-executable helper fails every terminal spawn with
// `Error: posix_spawnp failed.`. Packaged builds are covered because
// stage-native-deps.mjs chmods the staged copy, but the dev flow
// (`npm run dev` → `electron .`) resolves node-pty straight from
// `node_modules/`, which nobody chmods. This restores the exec bits at runtime,
// best-effort, so both dev and any environment that stripped the bit keep
// working. Idempotent: files that are already executable are left untouched.

import {
  chmodSync as realChmodSync,
  existsSync as realExistsSync,
  readdirSync as realReaddirSync,
  statSync as realStatSync
} from 'node:fs'
import { join } from 'node:path'

const EXEC_BITS = 0o111

export interface SpawnHelperFs {
  existsSync(path: string): boolean
  readdirSync(path: string): string[]
  statSync(path: string): { mode: number }
  chmodSync(path: string, mode: number): void
}

export interface EnsureSpawnHelperResult {
  fixed: string[]
  errors: { path: string; error: string }[]
}

const defaultFs: SpawnHelperFs = {
  existsSync: realExistsSync,
  readdirSync: (path: string) => realReaddirSync(path),
  statSync: (path: string) => realStatSync(path),
  chmodSync: realChmodSync
}

// True when any of the owner/group/other execute bits are missing.
export function needsExecBit(mode: number): boolean {
  return (mode & EXEC_BITS) !== EXEC_BITS
}

// Preserve existing permission bits, adding execute for owner/group/other.
export function withExecBits(mode: number): number {
  return mode | EXEC_BITS
}

// Every place a `spawn-helper` can live under a node-pty package root: one per
// bundled prebuild (`prebuilds/<platform>-<arch>/`) plus a locally compiled
// `build/Release/` copy. Windows layouts have no spawn-helper, so the list is
// naturally empty there.
export function spawnHelperCandidates(
  nodePtyRoot: string,
  fs: Pick<SpawnHelperFs, 'existsSync' | 'readdirSync'> = defaultFs
): string[] {
  const candidates: string[] = []
  const prebuilds = join(nodePtyRoot, 'prebuilds')

  if (fs.existsSync(prebuilds)) {
    for (const entry of fs.readdirSync(prebuilds)) {
      candidates.push(join(prebuilds, entry, 'spawn-helper'))
    }
  }

  candidates.push(join(nodePtyRoot, 'build', 'Release', 'spawn-helper'))

  return candidates
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

// Best-effort: ensure every existing spawn-helper under `nodePtyRoot` is
// executable. Never throws — missing files are skipped, and chmod/stat failures
// are collected so the caller can log them without breaking terminal startup.
export function ensureSpawnHelperExecutable(
  nodePtyRoot: string,
  fs: SpawnHelperFs = defaultFs
): EnsureSpawnHelperResult {
  const result: EnsureSpawnHelperResult = { fixed: [], errors: [] }

  for (const path of spawnHelperCandidates(nodePtyRoot, fs)) {
    if (!fs.existsSync(path)) {
      continue
    }

    let mode: number

    try {
      mode = fs.statSync(path).mode
    } catch (error) {
      result.errors.push({ path, error: errorMessage(error) })

      continue
    }

    if (!needsExecBit(mode)) {
      continue
    }

    try {
      fs.chmodSync(path, withExecBits(mode))
      result.fixed.push(path)
    } catch (error) {
      result.errors.push({ path, error: errorMessage(error) })
    }
  }

  return result
}
