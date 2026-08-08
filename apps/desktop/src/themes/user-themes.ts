/**
 * User-installed desktop themes (currently: converted VS Code themes).
 *
 * This is the extensibility seam. The theme context reads the *merged* registry
 * (built-ins + user themes) for `availableThemes` and for every skin lookup, so
 * an installed theme shows up everywhere a built-in does — the Cmd-K palette,
 * the Appearance settings grid, and `/skin` — with no per-surface wiring.
 *
 * Stored as a localStorage record so the boot-time paint (which runs before
 * React mounts) can resolve a user theme synchronously, same as built-ins.
 */

import { atom, computed } from 'nanostores'

import { registry } from '@/contrib/registry'

import { $backendThemes } from './backend-sync'
import { BUILTIN_THEMES } from './presets'
import type { DesktopTheme, DesktopThemeColors } from './types'

const USER_THEMES_KEY = 'hermes-desktop-user-themes-v1'

// Marketplace imports stamp their description "VS Code · <publisher.extension>"
// (see `convertVscodeColorTheme`). This is the one place that convention is read
// back out, so every install surface can tell what's already installed.
const MARKETPLACE_DESC_PREFIX = 'VS Code · '

// The minimal set of color keys a stored theme must carry to be usable. We keep
// this loose — `applyTheme` tolerates missing optionals via fallbacks — but a
// theme with no background/foreground/primary is junk and gets dropped.
const REQUIRED_COLOR_KEYS: ReadonlyArray<keyof DesktopThemeColors> = ['background', 'foreground', 'primary']

function isValidTheme(value: unknown): value is DesktopTheme {
  if (!value || typeof value !== 'object') {
    return false
  }

  const theme = value as Partial<DesktopTheme>

  if (typeof theme.name !== 'string' || typeof theme.label !== 'string' || !theme.colors) {
    return false
  }

  const colors = theme.colors as unknown as Record<string, unknown>

  return REQUIRED_COLOR_KEYS.every(key => typeof colors[key] === 'string')
}

function readStored(): Record<string, DesktopTheme> {
  try {
    const raw = window.localStorage.getItem(USER_THEMES_KEY)

    if (!raw) {
      return {}
    }

    const parsed: unknown = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    const out: Record<string, DesktopTheme> = {}

    for (const [key, value] of Object.entries(parsed)) {
      // Never let a stored theme shadow a built-in name.
      if (!BUILTIN_THEMES[key] && isValidTheme(value)) {
        out[key] = value
      }
    }

    return out
  } catch {
    return {}
  }
}

function persist(record: Record<string, DesktopTheme>) {
  try {
    window.localStorage.setItem(USER_THEMES_KEY, JSON.stringify(record))
  } catch {
    // Best-effort: a restricted storage context shouldn't break theming.
  }
}

/** Reactive map of installed user themes, keyed by slug. */
export const $userThemes = atom<Record<string, DesktopTheme>>(typeof window === 'undefined' ? {} : readStored())

/** Install (or replace) a user theme. Returns the stored theme. */
export function installUserTheme(theme: DesktopTheme): DesktopTheme {
  if (BUILTIN_THEMES[theme.name]) {
    throw new Error(`"${theme.name}" collides with a built-in theme.`)
  }

  if (!isValidTheme(theme)) {
    throw new Error('Theme is missing required colors.')
  }

  const next = { ...$userThemes.get(), [theme.name]: theme }
  $userThemes.set(next)
  persist(next)

  return theme
}

/** Remove a user theme by slug. No-op for unknown / built-in names. */
export function removeUserTheme(name: string): void {
  const current = $userThemes.get()

  if (!current[name]) {
    return
  }

  const next = { ...current }
  delete next[name]
  $userThemes.set(next)
  persist(next)
}

export const isUserTheme = (name: string): boolean => Boolean($userThemes.get()[name])

// ── Backend skin persistence cache ──────────────────────────────────────────
// Backend skins (~/.hermes/skins/*.yaml) are converted in the renderer and
// registered only after the gateway connects, but boot-time resolution runs
// before that — so a chosen backend skin normalizes to the default on every
// relaunch. Cache the converted record of every applied backend skin in
// localStorage so boot can resolve it synchronously. The LIVE $backendThemes
// conversion always wins once connected (it precedes this cache in
// resolveTheme), so recolors still repaint; the cache only bridges the
// pre-connect window and the boot-time normalizeSkin check.
const BACKEND_CACHE_KEY = 'hermes-desktop-backend-skin-cache-v1'

function readBackendCache(): Record<string, DesktopTheme> {
  try {
    const raw = window.localStorage.getItem(BACKEND_CACHE_KEY)

    if (!raw) {
      return {}
    }

    const parsed: unknown = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    const out: Record<string, DesktopTheme> = {}

    for (const [key, value] of Object.entries(parsed)) {
      if (!BUILTIN_THEMES[key] && isValidTheme(value)) {
        out[key] = value
      }
    }

    return out
  } catch {
    return {}
  }
}

/** Reactive map of persisted backend-skin conversions, keyed by name. */
export const $backendSkinCache = atom<Record<string, DesktopTheme>>(
  typeof window === 'undefined' ? {} : readBackendCache()
)

/** Persist a backend skin's converted record so a future boot can resolve it. */
export function cacheBackendSkin(theme: DesktopTheme): void {
  const next = { ...$backendSkinCache.get(), [theme.name]: theme }
  $backendSkinCache.set(next)

  try {
    window.localStorage.setItem(BACKEND_CACHE_KEY, JSON.stringify(next))
  } catch {
    // Best-effort: a restricted storage context shouldn't break theming.
  }
}

/**
 * Reconcile the backend-skin cache against the backend's authoritative list
 * of available skin names. Entries whose skin file no longer exists (deleted
 * from `$HERMES_HOME/skins/`) are dropped, so a retired skin stops ghosting in
 * the picker after the next connect. Keeps every surviving record intact —
 * this is a prune, never a clobber.
 */
export function pruneBackendSkinCache(availableNames: readonly string[]): void {
  const valid = new Set(availableNames)
  const current = $backendSkinCache.get()
  const next: Record<string, DesktopTheme> = {}

  for (const [name, theme] of Object.entries(current)) {
    if (valid.has(name)) {
      next[name] = theme
    }
  }

  if (Object.keys(next).length === Object.keys(current).length) {
    return
  }

  $backendSkinCache.set(next)

  try {
    window.localStorage.setItem(BACKEND_CACHE_KEY, JSON.stringify(next))
  } catch {
    // Best-effort: a restricted storage context shouldn't break theming.
  }
}

/** The Marketplace extension id an installed theme came from, or null. */
export function marketplaceIdOf(theme: DesktopTheme): string | null {
  return theme.description.startsWith(MARKETPLACE_DESC_PREFIX)
    ? theme.description.slice(MARKETPLACE_DESC_PREFIX.length)
    : null
}

/**
 * Reactive `extensionId → installed theme` map, so the install UIs can mark
 * Marketplace rows you already have (and re-activate them without re-downloading)
 * from one memoized source instead of re-deriving the set on every render.
 */
export const $marketplaceInstalls = computed($userThemes, themes => {
  const map = new Map<string, DesktopTheme>()

  for (const theme of Object.values(themes)) {
    const id = marketplaceIdOf(theme)

    if (id) {
      map.set(id, theme)
    }
  }

  return map
})

// ── Contributed themes — the `themes` registry area ─────────────────────────
// A data contribution IS a DesktopTheme. Same validity bar as an installed
// theme; built-in names can't be shadowed, and user-installed themes win over
// contributed ones of the same name (the user's explicit install is intent).

export const THEMES_AREA = 'themes'

export function contributedThemes(): DesktopTheme[] {
  const seen = new Set<string>()
  const out: DesktopTheme[] = []

  for (const c of registry.getArea(THEMES_AREA)) {
    const theme = c.data as DesktopTheme | undefined

    if (theme && isValidTheme(theme) && !BUILTIN_THEMES[theme.name] && !seen.has(theme.name)) {
      seen.add(theme.name)
      out.push(theme)
    }
  }

  return out
}

/** Resolve a theme by name across the merged set (built-in + user + backend + cache + contributed). */
export function resolveTheme(name: string): DesktopTheme | undefined {
  return (
    BUILTIN_THEMES[name] ??
    $userThemes.get()[name] ??
    $backendThemes.get()[name] ??
    $backendSkinCache.get()[name] ??
    contributedThemes().find(theme => theme.name === name)
  )
}

/**
 * Resolve a theme from boot-time sources ONLY (everything except the live
 * backend registry). The backend's skins aren't available until the gateway
 * seeds them, so this is what boot-time normalizeSkin/deriveTheme can see.
 * Used by the first-connect adoption check: a persisted name that only
 * resolves via $backendThemes was seeded JUST NOW — it was unresolvable at
 * boot (the classic "reverts to the default after relaunch" case), so the
 * adoption must still fire and cache the record.
 */
export function resolveBootTheme(name: string): DesktopTheme | undefined {
  return (
    BUILTIN_THEMES[name] ??
    $userThemes.get()[name] ??
    $backendSkinCache.get()[name] ??
    contributedThemes().find(theme => theme.name === name)
  )
}

/** Built-ins first (stable order), then contributed, then backend skins, then cached backend skins, then user installs. */
export function listAllThemes(): DesktopTheme[] {
  const user = $userThemes.get()
  const backend = $backendThemes.get()
  const cache = $backendSkinCache.get()
  const shadows = (theme: DesktopTheme) => user[theme.name] || backend[theme.name] || cache[theme.name]

  return [
    ...Object.values(BUILTIN_THEMES),
    ...contributedThemes().filter(theme => !shadows(theme)),
    ...Object.values(backend).filter(theme => !user[theme.name]),
    ...Object.values(cache).filter(theme => !user[theme.name] && !backend[theme.name]),
    ...Object.values(user)
  ]
}
