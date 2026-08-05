/**
 * Profile share: export/import a profile as a portable bundle.
 *
 * The archive is the CLI's own `hermes profile export` tar.gz (config, skills,
 * SOUL.md, cron — credentials always excluded), plus one desktop-only file at
 * the root: `desktop.json`, the appearance/interface overlay (skin + mode,
 * any user-theme definitions the skin needs, the profile rail color, and the
 * layout tree). A CLI import of the same archive simply carries the file
 * along; the desktop import applies it so the receiving user gets the whole
 * look — theme, layout, skills — as a ready-to-use profile.
 *
 * Paths, not bytes, cross the renderer↔backend boundary: the native save/open
 * dialogs and the backend share the filesystem for local and pooled backends.
 */

import { isLayoutNode, normalize } from '@/components/pane-shell/tree/model'
import { $layoutTree, markActivePreset, persistTree } from '@/components/pane-shell/tree/store'
import { exportProfileArchive, importProfileArchive } from '@/hermes'
import { translateNow } from '@/i18n'
import { modePref, skinPref, type ThemeMode } from '@/themes/context'
import { BUILTIN_THEMES } from '@/themes/presets'
import type { DesktopTheme } from '@/themes/types'
import { $userThemes, installUserTheme, resolveTheme } from '@/themes/user-themes'
import type { ProfileDesktopOverlay } from '@/types/hermes'

import { notify, notifyError } from './notifications'
import {
  $activeGatewayProfile,
  $profileColors,
  normalizeProfileKey,
  refreshActiveProfile,
  selectProfile,
  setProfileColor
} from './profile'

/** Filename of the overlay inside the archive (profile root). */
export const DESKTOP_OVERLAY_FILENAME = 'desktop.json'

const OVERLAY_VERSION = 1

/**
 * Snapshot the desktop appearance/interface for `profile` into the overlay.
 * The layout tree is global (one window layout, not per-profile) — it rides
 * along so the receiver can opt into the sender's whole interface.
 */
export function buildDesktopOverlay(profile: string): ProfileDesktopOverlay {
  const key = normalizeProfileKey(profile)
  const skin = skinPref.resolve(key)
  const mode = modePref.resolve(key)

  // Bundle the full definition of any non-built-in theme the skin points at,
  // so the receiver's picker can resolve it. Built-ins resolve by name.
  const themes: Record<string, unknown> = {}
  const userTheme = BUILTIN_THEMES[skin] ? undefined : $userThemes.get()[skin]

  if (userTheme) {
    themes[userTheme.name] = userTheme
  }

  return {
    version: OVERLAY_VERSION,
    skin,
    mode,
    ...(Object.keys(themes).length ? { themes } : {}),
    profileColor: $profileColors.get()[key] ?? null,
    layoutTree: $layoutTree.get()
  }
}

/** Export `profile` (backend archive + desktop overlay) to `output` (or the
 *  backend's staging dir when omitted). Returns the archive path. */
export async function exportProfileBundle(profile: string, output?: string): Promise<string> {
  const overlay = buildDesktopOverlay(profile)

  const { archive } = await exportProfileArchive(profile, {
    extraFiles: { [DESKTOP_OVERLAY_FILENAME]: JSON.stringify(overlay, null, 2) },
    output
  })

  return archive
}

const isThemeMode = (value: unknown): value is ThemeMode =>
  value === 'light' || value === 'dark' || value === 'system'

/**
 * Apply an imported overlay: install bundled themes, assign the new profile's
 * skin + mode + rail color, and (when present) adopt the sender's layout tree.
 * Every step is independent and best-effort — a malformed half never blocks
 * the rest, and a missing overlay is a plain CLI-exported archive (no-op).
 */
export function applyDesktopOverlay(profile: string, overlay: null | ProfileDesktopOverlay | undefined): void {
  if (!overlay || typeof overlay !== 'object') {
    return
  }

  const key = normalizeProfileKey(profile)

  // 1. Bundled theme definitions. installUserTheme validates shape and refuses
  //    built-in collisions; a bad entry just doesn't install.
  for (const theme of Object.values(overlay.themes ?? {})) {
    try {
      installUserTheme(theme as DesktopTheme)
    } catch {
      // Invalid/colliding theme — the skin assignment below falls back.
    }
  }

  // 2. Appearance assignment for the new profile. Only assign a skin that
  //    actually resolves so the pref never points at nothing.
  if (typeof overlay.skin === 'string' && resolveTheme(overlay.skin)) {
    skinPref.assign(key, overlay.skin)
  }

  if (isThemeMode(overlay.mode)) {
    modePref.assign(key, overlay.mode)
  }

  // 3. Rail color.
  if (typeof overlay.profileColor === 'string' && overlay.profileColor) {
    setProfileColor(key, overlay.profileColor)
  }

  // 4. Layout tree — global by design (one window layout). Normalize through
  //    the same canonicalizer the boot load uses; a null result means the
  //    tree was junk, so the current layout stays.
  if (overlay.layoutTree != null && isLayoutNode(overlay.layoutTree)) {
    const tree = normalize(overlay.layoutTree)

    if (tree) {
      $layoutTree.set(tree)
      persistTree()
      markActivePreset('custom')
    }
  }
}

/** Import an archive, apply its desktop overlay, return the new profile name. */
export async function importProfileBundle(archive: string, name?: string): Promise<string> {
  const result = await importProfileArchive(archive, name)
  applyDesktopOverlay(result.name, result.desktop)

  return result.name
}

/** The profile the export pickers should default to — the active one. */
export function activeProfileKey(): string {
  return normalizeProfileKey($activeGatewayProfile.get())
}

// ── Dialog-driven flows ──────────────────────────────────────────────────────
// One store function per user verb (⌘K row, rail button, and any future menu
// item all funnel here). Toasts via the shared notification store; strings via
// translateNow so the flows stay callable from non-React surfaces.

const ARCHIVE_FILTERS = [{ extensions: ['tar.gz', 'tgz'], name: 'Hermes profile' }]

/** Pick a save location and export `profile` (default: the active one).
 *  Returns the archive path, or null when the user cancelled. */
export async function runExportProfileFlow(profile?: string): Promise<null | string> {
  const target = normalizeProfileKey(profile ?? activeProfileKey())
  const pick = window.hermesDesktop?.selectSavePath

  if (!pick) {
    return null
  }

  const output = await pick({
    title: translateNow('profiles.exportProfile'),
    defaultPath: `${target}.tar.gz`,
    filters: ARCHIVE_FILTERS
  })

  if (!output) {
    return null
  }

  try {
    const archive = await exportProfileBundle(target, output)
    notify({ kind: 'success', title: translateNow('profiles.exported'), message: archive })

    return archive
  } catch (error) {
    notifyError(error, translateNow('profiles.failedExport'))

    return null
  }
}

/** Pick an archive and import it as a new profile; lands the user in it on a
 *  fresh chat. Returns the new profile name, or null when cancelled/failed. */
export async function runImportProfileFlow(): Promise<null | string> {
  const paths = await window.hermesDesktop?.selectPaths?.({
    title: translateNow('profiles.importProfile'),
    multiple: false,
    filters: ARCHIVE_FILTERS
  })

  const archive = paths?.[0]

  if (!archive) {
    return null
  }

  try {
    const name = await importProfileBundle(archive)
    notify({ kind: 'success', title: translateNow('profiles.imported'), message: name })
    // Same landing as CreateProfileDialog's onCreated: refresh the list, then
    // switch into the new profile on a fresh chat.
    await refreshActiveProfile()
    selectProfile(name)

    return name
  } catch (error) {
    notifyError(error, translateNow('profiles.failedImport'))

    return null
  }
}
