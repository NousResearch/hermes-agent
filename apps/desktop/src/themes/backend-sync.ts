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

import { storedString, storedStringRecord } from '@/lib/storage'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

import { BUILTIN_THEMES } from './presets'
import { skinToDesktopTheme } from './skin'
import type { DesktopTheme } from './types'
import { $userThemes, installUserTheme } from './user-themes'

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

// Skin keys — keep in sync with `themes/context.tsx` (read-only use here,
// avoiding a circular import: context.tsx imports this module).
const SKIN_LEGACY_KEY = 'hermes-desktop-theme-v2'
const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'

/**
 * Whether `name` is what the user persisted for the live profile. Boot paint
 * runs before the gateway registers backend themes, so a backend-sourced skin
 * (e.g. one from `display.skin`) that the user picked is stored but NOT
 * resolvable on the first frame — `normalizeSkin` falls back to the default
 * and the connect-time seed deliberately never repaints. Seeding an already
 * persisted skin is not stomping a manual choice; it's finishing it.
 */
function isPersistedSkin(name: string): boolean {
  if (name === 'default') {
    return false
  }

  const profile = normalizeProfileKey($activeGatewayProfile.get())
  const stored =
    profile === 'default'
      ? storedString(SKIN_LEGACY_KEY)
      : (storedStringRecord(PROFILE_SKINS_KEY)[profile] ?? storedString(SKIN_LEGACY_KEY))

  return stored === name
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

  // `default` is "no opinion" on the PALETTE — the desktop keeps its own default
  // (nous), so we never register a converted theme under `default`. It is still a
  // valid apply TARGET though: a runtime switch back to `default` must repaint the
  // desktop to its own default (setTheme normalizes `default` → nous). So we only
  // skip the registry step here and let it flow through the apply logic below.
  // Built-in names (mono/slate/…) already have a hand-tuned desktop palette — we
  // never shadow it, but the name is still a valid apply target.
  if (name !== 'default' && !BUILTIN_THEMES[name]) {
    const theme = skinToDesktopTheme(skin as HermesSkin)

    if (!theme) {
      return
    }

    const current = $backendThemes.get()

    if (JSON.stringify(current[name]) !== JSON.stringify(theme)) {
      $backendThemes.set({ ...current, [name]: theme })
    }

    // Persist backend skins as user themes so the NEXT boot's first paint can
    // resolve them synchronously — boot paint runs before the gateway connects
    // and registers backend themes, so without this a backend-sourced skin
    // (e.g. one from `display.skin`) paints the default on the connecting
    // screen and only repaints after `gateway.ready`. Storing the converted
    // theme in localStorage makes it resolve exactly like a built-in.
    const installed = $userThemes.get()[name]

    if (JSON.stringify(installed) !== JSON.stringify(theme)) {
      installUserTheme(theme)
    }
  }

  if (!apply) {
    // Connect-time seed: record without painting. A reconnect re-seed keeps an
    // earlier real apply's flag so repeat events can't override a manual switch.
    // Exception: the user's persisted skin IS this backend skin — it was picked
    // last session, so repaint it now that it's finally resolvable.
    if (isPersistedSkin(name)) {
      lastSynced = { applied: true, name }
      $pendingSkinApply.set(name)

      return
    }

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
