// cosmic.ts — window enumeration for the COSMIC desktop (cosmic-comp).
//
// COSMIC does not expose X11-style `_NET_CLIENT_LIST`, and it runs every app as
// a native-Wayland client — so the generic `get-windows`/xprop path enumerates
// nothing under a native-Wayland session. Instead COSMIC speaks the Wayland
// protocol `ext_foreign_toplevel_list_v1` (plus its own
// `zcosmic_toplevel_info_v1`). Hermes ships a small helper,
// `cosmic-toplevel-list`, that connects to the compositor and prints every
// open toplevel as JSON (`title`, `app_id`, `identifier`; `geometry` is null
// because cosmic-comp 1.0 does not serve geometry over Wayland).
//
// Two COSMIC paths therefore exist, and the user may choose either:
//
//   1. Native Wayland (default on COSMIC): `cosmic-toplevel-list` is shelled
//      out and returns title/app_id/identifier. Geometry is unavailable, so the
//      HUD uses name-based awareness. This is what makes COSMIC a first-class,
//      fully-working platform without forcing X11.
//
//   2. XWayland: when Hermes is launched with `desktop.ozone_platform_hint: x11`
//      (see issue #84011 / PR #84013, which bridges
//      `ELECTRON_OZONE_PLATFORM_HINT`), it becomes an X11 client and the shared
//      `get-windows`/xprop enumerator returns *full* geometry. Use this when the
//      HUD needs pixel-exact positions under COSMIC.
//
// This provider only ever answers on COSMIC and otherwise returns null, so the
// established Hyprland → COSMIC → X11 fallback chain is preserved everywhere
// else. Same contract as `hyprland.ts`.

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import path from 'node:path'
import { existsSync } from 'node:fs'
import type { EnumeratedWindow } from './window-below'

const execFileAsync = promisify(execFile)

/** True when the active session is the COSMIC desktop. */
export function isCosmic(env: NodeJS.ProcessEnv): boolean {
  const current = (env.XDG_CURRENT_DESKTOP ?? '').toLowerCase()
  const session = (env.XDG_SESSION_DESKTOP ?? '').toLowerCase()

  return current.includes('cosmic') || session.includes('cosmic')
}

interface CosmicToplevel {
  title: string | null
  app_id: string | null
  identifier: string | null
  geometry: null
}

function toEnumerated(w: CosmicToplevel): EnumeratedWindow {
  // COSMIC reliably gives `app_id` (the stable launcher id) and `title`. It does
  // not serve geometry or pid over Wayland in 1.0, so bounds/pid are
  // placeholders; the HUD's name-based awareness still works, and pixel-exact
  // positioning is available under XWayland (see ozone_platform_hint).
  const id = w.identifier ? hashString(w.identifier) : 0
  return {
    app: w.app_id ?? '',
    bounds: { x: 0, y: 0, width: 0, height: 0 },
    id: id || 0,
    pid: 0,
    title: w.title ?? w.app_id ?? '',
  }
}

function resolveCosmicHelperPath(): string {
  // electron-builder extraResources ships the staged binary into
  // process.resourcesPath. The exact on-disk location depends on whether the
  // extraResources `from` entry is treated as a file or a directory, so we
  // probe both shapes and fall back to a bare name (PATH) for dev builds and
  // manual installs. Returns the first candidate that exists on disk.
  const name = 'cosmic-toplevel-list'
  const candidates: string[] = []
  if (typeof process !== 'undefined' && process.resourcesPath) {
    // Directory entry: <resources>/cosmic-toplevel-list/cosmic-toplevel-list
    candidates.push(path.join(process.resourcesPath, name, name))
    // File entry: <resources>/cosmic-toplevel-list
    candidates.push(path.join(process.resourcesPath, name))
  }
  candidates.push(name) // bare name → resolved via PATH

  for (const candidate of candidates) {
    try {
      if (existsSync(candidate)) return candidate
    } catch {
      // ignore and try the next candidate
    }
  }
  // None exist; return the most likely path so the exec error is actionable.
  return candidates[0]
}

function hashString(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0
  }
  return h >>> 0
}

/**
 * Enumerate COSMIC windows via the `cosmic-toplevel-list` helper, or null when
 * this is not a COSMIC session or the helper is unavailable.
 *
 // The helper is built from `apps/desktop/cosmic-toplevel-list` by the desktop
 // pack pipeline (scripts/stage-native-deps.mjs -> cargo build) and shipped via
 // electron-builder `extraResources` into `process.resourcesPath`. If it is
 // missing we return null and the caller falls through to `get-windows`
 // (which works under XWayland) and finally to the COSMIC guidance note.
 */
export async function readCosmicWindows(
  selfPid: number,
  titlesAvailable: boolean,
  env: NodeJS.ProcessEnv,
  enumerate: (titlesAvailable: boolean) => Promise<EnumeratedWindow[] | null>
): Promise<EnumeratedWindow[] | null> {
  if (!isCosmic(env)) {
    return null
  }

  // Native Wayland: try the COSMIC helper first. It speaks the compositor's own
  // Wayland protocol and needs no X11. The binary is shipped via electron-builder
  // extraResources into process.resourcesPath; fall back to a bare name (PATH)
  // for dev builds / manual installs.
  const helperPath = resolveCosmicHelperPath()
  try {
    const { stdout } = await execFileAsync(helperPath, [], {
      env,
      timeout: 5000,
      maxBuffer: 10 * 1024 * 1024,
    })
    const parsed = JSON.parse(stdout) as CosmicToplevel[]
    if (Array.isArray(parsed) && parsed.length > 0) {
      const windows = parsed.map(toEnumerated).filter((w) => w.title.length > 0)
      if (windows.length > 0) {
        return windows
      }
    }
  } catch {
    // Helper missing or failed — fall through to the X11 enumerator (works
    // under XWayland) and ultimately to the COSMIC guidance note.
  }

  return enumerate(titlesAvailable)
}
