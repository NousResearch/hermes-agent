/** Electron consumes the launch contract completed by the PM bundle builder. */
import { createHash } from 'node:crypto'
import path from 'node:path'

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
