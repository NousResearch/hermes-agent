/** Electron consumes the launch contract completed by the PM bundle builder. */
import { createHash } from 'node:crypto'
import os from 'node:os'
import path from 'node:path'

import { type BootProgress, stablePayloadResources } from './appimage-payload'
import { INSTALL_STAMP, type InstallStamp, type PayloadRuntime } from './install-stamp'

export interface PayloadInfo extends PayloadRuntime {
  root: string
  shim: string
}

export function bundledPayload(
  resourcesPath: string,
  stamp: Readonly<InstallStamp> | null = INSTALL_STAMP
): PayloadInfo | null {
  if (stamp?.payload !== 'bundled') {
    return null
  }

  // The builder validates these paths before baking the stamp. There is no
  // discovery, filesystem validation or alternative payload at runtime.
  const runtime = stamp.runtime!
  const root = path.join(resourcesPath, 'agent-payload')

  const commands = Object.fromEntries(
    Object.entries(runtime.commands).map(([name, relative]) => [name, path.join(root, relative)])
  )

  return {
    root,
    repoDir: path.join(root, runtime.repoDir),
    toolsDir: path.join(root, runtime.toolsDir),
    storePython: path.join(root, runtime.storePython),
    sitePackages: path.join(root, runtime.sitePackages),
    commands,
    shim: commands.hermes
  }
}

/** The payload this launch runs: an AppImage's stable per-build copy, else the artifact's own. */
export async function runnableBundledPayload(
  log: (line: string) => void,
  progress?: BootProgress
): Promise<PayloadInfo | null> {
  return bundledPayload(await stablePayloadResources(log, progress))
}

/**
 * PYTHONPATH for running a bundled core module with the payload python: the
 * bundled core code first, then the venv's dependencies. This is the order the
 * bin/ launchers give sys.path (scripts/build/launcher_wrapper.py).
 */
export function payloadPythonPath(payload: Pick<PayloadInfo, 'repoDir' | 'sitePackages'>): string {
  return [payload.repoDir, payload.sitePackages].join(path.delimiter)
}

/**
 * The bytecode cache the POSIX bin/ launchers give the payload python
 * (scripts/build/launchers.py): a pyc written into a macOS bundle breaks its
 * code signature. The Windows launcher sets none for a baked payload, whose
 * install tree is read-only.
 */
export function payloadPycacheEnv(
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): NodeJS.ProcessEnv {
  if (platform === 'win32') {
    return {}
  }

  const cache = env.XDG_CACHE_HOME || path.posix.join(env.HOME || os.homedir(), '.cache')

  return { PYTHONPYCACHEPREFIX: env.PYTHONPYCACHEPREFIX || path.posix.join(cache, 'hermes-pycache') }
}

/**
 * The bundled shim is the Hermes CLI, not an interpreter, so it rejects the
 * ticket bridge's `-P -u -c`. Ticket minting runs the payload python against
 * the same import closure and bytecode cache the shim gives the CLI.
 */
export function payloadTicketRuntime(
  payload: Pick<PayloadInfo, 'storePython' | 'repoDir' | 'sitePackages'>,
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): {
  ticketCommand: string
  ticketEnv: NodeJS.ProcessEnv
} {
  return {
    ticketCommand: payload.storePython,
    ticketEnv: { PYTHONPATH: payloadPythonPath(payload), ...payloadPycacheEnv(env, platform) }
  }
}

/**
 * The bundled Python payload for a strategy whose update check runs inside
 * it. A stamp that names such a mechanism without a payload is a broken
 * install; say so instead of failing on `payload.storePython`.
 */
export function requireBundledPayload(mechanism: string, resourcesPath = process.resourcesPath): PayloadInfo {
  const payload: PayloadInfo | null = bundledPayload(resourcesPath)

  if (!payload) {
    throw new Error(`${mechanism} updates require the bundled application payload, which this install has none of`)
  }

  return payload
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
