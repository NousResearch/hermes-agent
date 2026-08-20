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

import { readJson, writeJson } from '@/lib/storage'

import { BUILTIN_THEMES } from './presets'
import { skinToDesktopTheme } from './skin'
import type { DesktopTheme } from './types'
import { isValidTheme } from './validity'

// Converted backend skins are CACHED, not just held in memory. The desktop
// persists the active skin's NAME (`hermes-desktop-theme-v2`), and every window
// resolves that name against the theme registry on its very first paint — but a
// backend skin only entered the registry once ITS OWN gateway connection
// announced it. A freshly opened window (HUD, session pop-out, ⌘⇧N) therefore
// booted with the right name and no palette to resolve it to, fell back to
// `nous`, and painted the wrong theme until its socket caught up (which for the
// HUD is never: `gateway.ready` seeds WITHOUT applying, by design).
//
// Only the ACTIVE skin is persisted, because that is the only one this cache
// exists to resolve. The backend announces one skin — the active one — so
// keeping the others would be a store of entries nothing can refresh: a skin
// deleted from disk would linger in Appearance and `/skin` forever, offering a
// palette that no longer exists. One entry keeps the cache honest (it is at
// most one connect behind the backend, never authoritative over it) and bounds
// it, while the in-memory registry still accumulates normally within a session.
const BACKEND_THEMES_KEY = 'hermes-desktop-backend-themes-v1'

function readStored(): Record<string, DesktopTheme> {
  const parsed = readJson<unknown>(BACKEND_THEMES_KEY)

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return {}
  }

  const out: Record<string, DesktopTheme> = {}

  for (const [key, value] of Object.entries(parsed)) {
    // Never let a cached skin shadow a built-in name — same rule the live
    // ingest path applies below.
    if (!BUILTIN_THEMES[key] && isValidTheme(value)) {
      out[key] = value
    }
  }

  return out
}

/** Skins pushed by the backend, keyed by name. Merged by `listAllThemes`. */
export const $backendThemes = atom<Record<string, DesktopTheme>>(typeof window === 'undefined' ? {} : readStored())

/** Persist a single converted skin as the cold-boot cache, replacing any prior. */
const cacheActiveSkin = (name: string, theme: DesktopTheme) => writeJson(BACKEND_THEMES_KEY, { [name]: theme })

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

/** Test-only: the localStorage key the converted-skin cache lives under. */
export const __BACKEND_THEMES_KEY = BACKEND_THEMES_KEY

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

    // This skin is the backend's active one by definition — it only announces
    // the skin in force — so it is what a cold-booting window has to resolve.
    // Written on every announcement, including a re-seed carrying an in-place
    // palette edit, so the cache tracks the file rather than only the name.
    cacheActiveSkin(name, theme)
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
