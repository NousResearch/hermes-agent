import fs from 'node:fs'
import path from 'node:path'

import { classifyActiveRuntime } from './active-runtime-state'
import { canImportHermesCli } from './backend-probes'

interface DesktopBootstrapMarkerRuntimeDeps {
  ACTIVE_HERMES_ROOT: string
  VENV_ROOT: string
  BOOTSTRAP_COMPLETE_MARKER: string
  BOOTSTRAP_MARKER_SCHEMA_VERSION: number
  app: { getVersion: () => string }
  getVenvPython: (venvRoot: string) => string
  isHermesSourceRoot: (root: string) => boolean
  fileExists: (filePath: string) => boolean
  writeFileAtomic: (targetPath: string, data: string, encoding?: BufferEncoding) => void
}

export function createDesktopBootstrapMarkerRuntime(deps: DesktopBootstrapMarkerRuntimeDeps) {
  const {
    ACTIVE_HERMES_ROOT,
    VENV_ROOT,
    BOOTSTRAP_COMPLETE_MARKER,
    BOOTSTRAP_MARKER_SCHEMA_VERSION,
    app,
    getVenvPython,
    isHermesSourceRoot,
    fileExists,
    writeFileAtomic
  } = deps

  function readJson(filePath) {
    try {
      return JSON.parse(fs.readFileSync(filePath, 'utf8'))
    } catch {
      return null
    }
  }

  // Bootstrap-complete marker helpers. The marker is written by whichever
  // installer ran: install.ps1, install.sh, the Rust bootstrap installer, or the
  // first-launch bootstrap runner. It is provenance ("a bootstrap finished
  // here"), NOT the launch gate -- activeRuntimeState() decides that, because a
  // healthy runtime can predate the marker or outlive a repair that cleared it.
  //
  // Marker schema (version 1):
  //   {
  //     schemaVersion: 1,
  //     pinnedCommit: "<40-char SHA>",       // what install.ps1 was driven against
  //     pinnedBranch: "<branch name>" | null,
  //     completedAt:  "<ISO 8601>",
  //     desktopVersion: "<app.getVersion()>"  // for forensics
  //   }
  function readBootstrapMarker() {
    return readJson(BOOTSTRAP_COMPLETE_MARKER)
  }

  // Marker-independent: is the canonical install at ACTIVE_HERMES_ROOT actually
  // runnable right now? A complete CLI install (`install.sh --include-desktop`)
  // or a DMG launch over a prior CLI install satisfies this WITHOUT the desktop
  // ever having written the bootstrap marker -- so we must be able to recognise
  // "already installed" off the filesystem alone, not just the marker.
  async function isActiveRuntimeUsable() {
    const venvPython = getVenvPython(VENV_ROOT)

    return (
      isHermesSourceRoot(ACTIVE_HERMES_ROOT) &&
      fileExists(venvPython) &&
      // Explicit await: a bare promise as the last `&&` operand only works via
      // async-return flattening; any operand appended after it would make the
      // expression truthy regardless of the probe result.
      (await canImportHermesCli(venvPython, {
        env: {
          PYTHONPATH: [ACTIVE_HERMES_ROOT, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter)
        }
      }))
    )
  }

  async function activeRuntimeState() {
    // We DELIBERATELY do NOT verify that the checkout is currently at the
    // pinned commit -- users update via the in-app update path or `hermes
    // update`, which moves HEAD legitimately. The marker only attests "a
    // desktop-managed bootstrap ran here at least once"; runtime usability is
    // what decides whether we can actually launch.
    return classifyActiveRuntime(readBootstrapMarker(), BOOTSTRAP_MARKER_SCHEMA_VERSION, await isActiveRuntimeUsable())
  }

  function writeBootstrapMarker(payload) {
    fs.mkdirSync(path.dirname(BOOTSTRAP_COMPLETE_MARKER), { recursive: true })

    const merged = {
      schemaVersion: BOOTSTRAP_MARKER_SCHEMA_VERSION,
      pinnedCommit: payload.pinnedCommit || null,
      pinnedBranch: payload.pinnedBranch || null,
      completedAt: new Date().toISOString(),
      desktopVersion: app.getVersion()
    }

    writeFileAtomic(BOOTSTRAP_COMPLETE_MARKER, JSON.stringify(merged, null, 2) + '\n', 'utf8')

    return merged
  }

  return { readBootstrapMarker, isActiveRuntimeUsable, activeRuntimeState, writeBootstrapMarker }
}
