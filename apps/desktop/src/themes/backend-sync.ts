/**
 * Live skin sync from the Hermes backend.
 *
 * The backend resolves the active skin (built-in or `$HERMES_HOME/skins/*.yaml`)
 * and announces it on `gateway.ready` / `skin.changed`, and answers `config.get
 * skin` with the same payload. `ingestBackendSkin` folds that into the desktop:
 *
 *   1. Registers the converted theme in `$backendThemes` so it appears wherever a
 *      built-in does — Appearance, Cmd-K, `/skin` — with no per-surface wiring
 *      (`listAllThemes` merges this store).
 *   2. When asked to apply (an explicit change), requests the switch via
 *      `$pendingSkinApply`, which the ThemeProvider drains through `setTheme`.
 *
 * `gateway.ready` seeds the baseline WITHOUT applying, so a fresh connect never
 * stomps the user's persisted desktop theme; only a genuine name change (Hermes
 * authoring/activating a skin from a prompt, or `/skin` elsewhere) repaints.
 */

import type { HermesSkin } from '@hermes/shared/skin'
import { atom } from 'nanostores'

import { BUILTIN_THEMES } from './presets'
import { skinToDesktopTheme } from './skin'
import type { DesktopTheme } from './types'

/** Skins pushed by the backend, keyed by name. Merged by `listAllThemes`. */
export const $backendThemes = atom<Record<string, DesktopTheme>>({})

/** One-shot skin name the ThemeProvider should switch to (it clears this). */
export const $pendingSkinApply = atom<string | null>(null)

// Last skin name synced from the backend + whether it was ever APPLIED (vs
// merely seeded at connect). Once applied, only a name change applies again —
// no re-apply on repeat events, no snap-back after a manual desktop switch.
// A `skin.changed` matching a seed-only baseline still applies: the seed
// records without painting, so if the activation event was missed (backend
// restart / disconnected), an explicit re-affirm must repaint, not no-op.
let lastSynced: { applied: boolean; name: string } | null = null

/** Test-only: reset the module's apply guard + registry between cases. */
export function __resetBackendSkinSync(): void {
  lastSynced = null
  $backendThemes.set({})
  $pendingSkinApply.set(null)
}

/**
 * Fold a resolved skin into the desktop. `apply: false` (connect-time seed) only
 * records the baseline; `apply: true` (runtime change / poll) repaints on a name
 * change. Built-in names keep the desktop's own palette but can still be applied.
 */
export function ingestBackendSkin(skin: HermesSkin | undefined | null, { apply }: { apply: boolean }): void {
  const name = (skin && typeof skin === 'object' ? (skin.name ?? '') : '').trim()

  if (!name) {
    return
  }

  // Built-in names (mono/slate/…) already have a hand-tuned desktop palette —
  // never shadow them, but they remain valid apply targets. The CLI `default`
  // skin ("Classic Hermes — gold and kawaii") is *not* a desktop built-in, so
  // we register the converted palette under `default` and let it appear in the
  // Appearance grid / `/skin list` (#76579). Desktop's own boot default remains
  // `nous` via DEFAULT_SKIN_NAME; selecting `default` paints the classic gold.
  if (!BUILTIN_THEMES[name]) {
    const theme = skinToDesktopTheme(skin as HermesSkin)

    if (!theme) {
      return
    }

    // Prefer the CLI description for the classic default skin so the grid
    // shows "Classic Hermes — gold and kawaii" rather than a bare "Default".
    const description =
      name === 'default' && typeof (skin as HermesSkin).description === 'string'
        ? String((skin as HermesSkin).description).trim() || theme.description
        : theme.description
    const label =
      name === 'default'
        ? 'Classic Hermes'
        : theme.label
    const registered = { ...theme, description, label }

    const current = $backendThemes.get()

    if (JSON.stringify(current[name]) !== JSON.stringify(registered)) {
      $backendThemes.set({ ...current, [name]: registered })
    }
  }

  if (!apply) {
    // Connect-time seed: record without painting. A reconnect re-seed keeps an
    // earlier real apply's flag so repeat events can't override a manual switch.
    if (lastSynced?.name !== name || !lastSynced.applied) {
      lastSynced = { applied: false, name }
    }

    return
  }

  if (name !== lastSynced?.name || !lastSynced.applied) {
    lastSynced = { applied: true, name }
    $pendingSkinApply.set(name)
  }
}
