/**
 * payload-backend.ts
 *
 * The bundled-install backend: a pm payload staged by `hermes pm bundle`
 * and shipped under resources/agent-payload. The payload carries the repo
 * snapshot, the tool store (with facts.json), and a relocatable venv
 * built on the staged python-build-standalone interpreter.
 *
 * Electron's whole job here is finding the interpreter and verifying the
 * payload can boot. Bundled builds run the store python directly —
 * self-relative, no pyvenv.cfg write, so read-only installs (MSIX) work.
 * Everything else — managed tool PATHs, env composition — happens
 * in-process via pm when the backend runs.
 */

import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

export interface PayloadInfo {
  root: string
  repoDir: string
  toolsDir: string
  /** The payload's own CPython (tools/<python-entry>/python(.exe)). */
  storePython: string
  /** The venv's site-packages (Lib/site-packages on win, lib/python3.11/site-packages on posix). */
  sitePackages: string
  /** The self-relative CLI trampoline (bin/hermes(.exe)) — the bundled entry point. */
  shim: string
}

export function resolvePayload(
  resourcesPath: string | undefined,
  deps: {
    fileExists: (p: string) => boolean
    directoryExists: (p: string) => boolean
    isWindows: boolean
  }
): PayloadInfo | null {
  if (!resourcesPath) {
    return null
  }

  const root = path.join(resourcesPath, 'agent-payload')
  const manifestPath = path.join(root, 'manifest.json')

  if (!deps.fileExists(manifestPath)) {
    return null
  }

  let manifest: any

  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  } catch {
    return null
  }

  if (!manifest || manifest.external === true) {
    return null
  }

  // The manifest names the payload layout; a bundle that omits the fields
  // is malformed — refuse it rather than guess at cmd_bundle's layout.
  if (typeof manifest.repo !== 'string' || typeof manifest.store !== 'string' || typeof manifest.venv !== 'string') {
    return null
  }

  const repoDir = path.join(root, manifest.repo)
  const toolsDir = path.join(root, manifest.store)
  const venvDir = path.join(root, manifest.venv)
  // The CLI trampoline staged into bin/ (hermes/hermes-agent/hermes-acp —
  // build-bundled-desktop.mjs 5b: distlib-minted launchers on win32,
  // $0-relative bash trampolines on POSIX). It execs the store python with
  // the payload's own PYTHONPATH, so it is the single bundled entry point.
  const shim = path.join(root, 'bin', deps.isWindows ? 'hermes.exe' : 'hermes')

  // The store CPython + the venv's site-packages. Bundled builds run the
  // STORE python (self-relative, no pyvenv.cfg write — works on read-only
  // MSIX) with PYTHONPATH pointing at the venv site-packages (where the
  // project deps are installed). The venv python itself is NOT used in
  // bundled builds.
  let storePython = ''
  let sitePackages = ''

  try {
    const facts = JSON.parse(fs.readFileSync(path.join(toolsDir, 'facts.json'), 'utf8'))
    const entry = facts?.packages?.python?.entry

    if (typeof entry === 'string') {
      storePython = path.join(toolsDir, entry, deps.isWindows ? 'python.exe' : 'bin', deps.isWindows ? '' : 'python3')
      sitePackages = deps.isWindows
        ? path.join(venvDir, 'Lib', 'site-packages')
        : path.join(venvDir, 'lib', `python${process.env.PYTHON_VER || '3.11'}`, 'site-packages')
    }
  } catch {
    // fall through to the existence checks below
  }

  if (!deps.directoryExists(repoDir) || !deps.fileExists(storePython) || !deps.directoryExists(sitePackages) || !deps.fileExists(shim)) {
    return null
  }

  return { root, repoDir, toolsDir, storePython, sitePackages, shim }
}

/**
 * "Is this artifact a bundled install?" — the app ships its own Hermes payload.
 * True whenever resources/agent-payload/manifest.json exists and is not the
 * external stub (before-build.mjs writes {schema:1, external:true} for
 * non-bundled builds). Deliberately does NOT verify payload usability: a
 * damaged bundle still must never install — callers use this to refuse the
 * installer, not to decide the payload can boot.
 */
export function isBundledInstall(
  resourcesPath: string | undefined,
  deps: { fileExists: (p: string) => boolean }
): boolean {
  if (!resourcesPath) {
    return false
  }

  const manifestPath = path.join(resourcesPath, 'agent-payload', 'manifest.json')

  if (!deps.fileExists(manifestPath)) {
    return false
  }

  try {
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))

    return Boolean(manifest) && manifest.external !== true
  } catch {
    return false
  }
}

/**
 * Verify the payload is usable — the store python + venv site-packages
 * resolve. NO pyvenv.cfg write: bundled builds run the store python
 * directly (self-relative, works on read-only MSIX), so there is nothing
 * to re-point. Returns true when the payload can boot.
 */
export function adoptPayloadVenv(
  payload: PayloadInfo,
  deps: { isWindows: boolean; log?: (m: string) => void }
): boolean {
  if (!payload.storePython || !payload.sitePackages) {
    deps.log?.('[payload] missing store python or site-packages')

    return false
  }

  return true
}

// ─── update channel ─────────────────────────────────────────────────────────
//
// The CLI owns the channel records; Electron only reads the install id for
// `update.installs.<sha16>/` bookkeeping. Channel resolution itself lives in
// hermes_cli/update_channel.py — main.ts keys canary/stable off the baked
// install stamp tag directly.

export type UpdateChannel = 'stable' | 'main' | 'canary'

/**
 * The install id of the tree at `root`: sha16 of the canonical PATH,
 * byte-identical to Python's install id (sha256 of the resolved root,
 * first 16 hex chars — `boot_bootstrap._install_key` /
 * `update_channel._install_key_sha16`). Path-derived so it survives
 * artifact replacement at the same location; the same key names
 * `installs/<sha16>/`.
 */
export function installIdForRoot(root: string, canonicalize: (p: string) => string = p => p): string {
  return createHash('sha256').update(canonicalize(root), 'utf8').digest('hex').slice(0, 16)
}
