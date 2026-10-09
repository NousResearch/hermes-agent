/**
 * Desktop theme context.
 *
 * Applies the active theme as CSS custom properties on :root so every
 * Tailwind utility that references a color or font-family token picks up
 * the change automatically.
 *
 * Mode (light/dark/system) controls brightness; skin controls accent.
 * The two are persisted independently. Shift+X toggles light/dark.
 */

import { ensureContrast, mix, parseColor } from '@hermes/shared/color'
import { useStore } from '@nanostores/react'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react'

import { $registryVersion } from '@/contrib/registry'
import { matchesQuery, useMediaQuery } from '@/hooks/use-media-query'
import { translateNow } from '@/i18n'
import { persistString, persistStringRecord, storedString, storedStringRecord } from '@/lib/storage'
import { recordFeatureUse } from '@/store/desktop-metrics'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'
import { setAppearance } from '@/store/translucency'

import { $accentOverride } from './accent-override'
import {
  $backendCustomCSS,
  $backendThemes,
  $pendingSkinApply,
  localDisplaySkinName,
  localDisplaySkinProfile
} from './backend-sync'
import { $chatFontFamily, resolveChatFontFamily } from './chat-font'
import { harmonize, readableInk } from './color'
import { BUILTIN_THEME_LIST, DEFAULT_SKIN_NAME, DEFAULT_TYPOGRAPHY, nousTheme, RETIRED_SKINS } from './presets'
import {
  $profileAppearance,
  adoptedAppearance,
  type AppearanceField,
  isThemeMode,
  profileAppearanceOwner,
  type ProfileAppearancePatch,
  refreshProfileAppearance,
  saveProfileAppearance
} from './profile-appearance'
import { retintTheme } from './retint'
import type { DesktopTheme, DesktopThemeColors, DesktopThemeTypography } from './types'
import { $userThemes, listAllThemes, resolveTheme } from './user-themes'

// Legacy global skin (pre per-profile themes). Still the inheritance fallback
// for any profile without its own assignment, so single-profile users and old
// installs are unaffected.
const SKIN_KEY = 'hermes-desktop-theme-v2'
const MODE_KEY = 'hermes-desktop-mode-v1'
// Per-profile skin + light/dark mode assignments: { [profileKey]: value }. A
// profile inherits the global default until it's given its own appearance.
const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'
const PROFILE_MODES_KEY = 'hermes-desktop-profile-modes-v1'
// The user's most recent pick on any profile: what a profile with no pick of
// its own inherits (#101216). Display-only; never uploaded to a config.
const INHERITED_SKIN_KEY = 'hermes-desktop-inherited-theme-v1'
const INHERITED_MODE_KEY = 'hermes-desktop-inherited-mode-v1'
// Last active profile, recorded so the boot-time paint can pick that profile's
// theme before the gateway reports which profile actually launched.
const LAST_PROFILE_KEY = 'hermes-desktop-active-profile-v1'

export type ThemeMode = 'light' | 'dark' | 'system'

const INJECTED_FONT_URLS = new Set<string>()

const resolveMode = (mode: ThemeMode, systemDark = matchesQuery('(prefers-color-scheme: dark)')): 'light' | 'dark' =>
  mode === 'system' ? (systemDark ? 'dark' : 'light') : mode

const normalizeSkin = (name: string | null): string =>
  name && resolveTheme(name) && !RETIRED_SKINS.has(name) ? name : DEFAULT_SKIN_NAME

/**
 * A stored mode, or `system` when there isn't one.
 *
 * A fresh profile follows the OS. Defaulting to `light` meant someone whose
 * desktop is dark got a white window on first launch and had to go find the
 * setting — and with per-appearance translucency it also handed them light's
 * much heavier tint, tuned for a bright desktop they don't have.
 */
const normalizeMode = (value: string | null): ThemeMode => (isThemeMode(value) ? value : 'system')

// ─── Per-profile appearance persistence ─────────────────────────────────────
// Skin and mode are each stored per profile. "default" isn't a real profile —
// it *is* the legacy global slot, so it reads/writes the global directly. Named
// profiles get their own entry and fall back to that global until assigned, so
// unassigned profiles and pre-per-profile installs stay on the global value.
// This is the local cache of the profile's config.yaml appearance
// (./profile-appearance): the boot paint reads it before any fetch.
//
// A user pick also records the inherited look, so a Bot Mode gateway hop onto
// a never-themed bot shows what the user just picked (#101216). It is a
// separate slot because the legacy one IS the default profile's own pick,
// which syncs to its config.yaml. Adopting a config is not a pick.
// A pre-existing install whose per-profile picks agree promotes that value
// (write-on-read; idempotent; a no-op when they disagree).
const promoteUnanimousPick = (record: string, legacy: string, inherited: string): void => {
  if (storedString(inherited) != null || storedString(legacy) != null) {
    return
  }

  const unique = [...new Set(Object.values(storedStringRecord(record)).filter(Boolean))]

  if (unique.length === 1) {
    persistString(inherited, unique[0])
  }
}

const profilePref = <T extends string>(
  record: string,
  legacy: string,
  inherited: string,
  normalize: (v: string | null) => T
) => {
  /** The profile's OWN pick — never the global value an unassigned profile inherits. */
  const own = (profile: string): string | null =>
    profile === 'default' ? storedString(legacy) : (storedStringRecord(record)[profile] ?? null)

  const stored = (profile: string): string | null => {
    promoteUnanimousPick(record, legacy, inherited)

    return own(profile) ?? storedString(inherited) ?? storedString(legacy)
  }

  /** Write a raw pick, or drop the profile's own entry (`null`). */
  const put = (profile: string, value: null | string): void => {
    if (profile === 'default') {
      persistString(legacy, value)

      return
    }

    const { [profile]: _dropped, ...rest } = storedStringRecord(record)
    persistStringRecord(record, value === null ? rest : { ...rest, [profile]: value })
  }

  /** A user pick: the profile's own value and the look unassigned profiles inherit. */
  const pick = (profile: string, value: string): void => {
    put(profile, value)
    persistString(inherited, value)
  }

  return {
    /** The pick as written, un-normalized. */
    stored,
    own,
    put,
    pick,
    resolve: (profile: string): T => normalize(stored(profile)),
    assign: (profile: string, value: T): void => pick(profile, value)
  }
}

export const skinPref = profilePref(PROFILE_SKINS_KEY, SKIN_KEY, INHERITED_SKIN_KEY, normalizeSkin)
export const modePref = profilePref(PROFILE_MODES_KEY, MODE_KEY, INHERITED_MODE_KEY, normalizeMode)

// The bridge's local skin is only a fallback for the profile this window booted
// into. A desktop-side pick remains the source of truth, and switching to a
// different profile cannot borrow a skin from this machine's initial profile.
const readBootProfileKey = () => normalizeProfileKey(storedString(LAST_PROFILE_KEY))
const BOOT_PROFILE_KEY = typeof window === 'undefined' ? 'default' : (localDisplaySkinProfile ?? readBootProfileKey())

// Provider state keeps the raw pick so a name nothing resolves YET (a backend
// skin the gateway hasn't seeded on this launch) isn't flattened to the default
// for the rest of the session — it paints as soon as the registry can resolve it.
const storedSkin = (profile: string): string =>
  skinPref.stored(profile) ??
  (profile === BOOT_PROFILE_KEY ? (localDisplaySkinName ?? DEFAULT_SKIN_NAME) : DEFAULT_SKIN_NAME)

/** Everything a peer window could change that this one has to repaint for. */
const APPEARANCE_KEYS = new Set([
  SKIN_KEY,
  PROFILE_SKINS_KEY,
  INHERITED_SKIN_KEY,
  MODE_KEY,
  PROFILE_MODES_KEY,
  INHERITED_MODE_KEY
])

const rememberActiveProfileKey = (profile: string) => persistString(LAST_PROFILE_KEY, profile)

// The profile picks are assigned to (read fresh, so callbacks stay stable across switches).
const liveProfile = () => normalizeProfileKey($activeGatewayProfile.get())

const APPEARANCE_PREFS: Record<AppearanceField, Pick<ReturnType<typeof profilePref>, 'own' | 'pick' | 'put'>> = {
  theme: skinPref,
  theme_mode: modePref
}

// This window's newest pick per (owner, field): it paints at once, over the
// config, until its save settles.
const pendingPicks = new Map<string, { value: string }>()
const pickKey = (owner: string, field: AppearanceField) => `${owner}\0${field}`

/** A profile's value on the live connection: a pick in flight here, then its config.yaml, else undefined (the cache decides). */
function configured(profile: string, field: AppearanceField): string | undefined {
  const owner = profileAppearanceOwner(profile)

  return pendingPicks.get(pickKey(owner, field))?.value ?? (adoptedAppearance(owner, field) || undefined)
}

const skinOf = (profile: string): string => configured(profile, 'theme') ?? storedSkin(profile)

const modeOf = (profile: string): ThemeMode =>
  normalizeMode(configured(profile, 'theme_mode') ?? modePref.stored(profile))

/** The cache follows the live connection's config.yaml, so the next boot
 *  paints it before any fetch; a pick in flight here keeps its slot until it
 *  settles. Adopting a config is not a pick. */
function cacheAdopted(profile: string): void {
  const owner = profileAppearanceOwner(profile)

  for (const field of ['theme', 'theme_mode'] as const) {
    const value = adoptedAppearance(owner, field)

    if (value && !pendingPicks.has(pickKey(owner, field)) && APPEARANCE_PREFS[field].own(profile) !== value) {
      APPEARANCE_PREFS[field].put(profile, value)
    }
  }
}

/**
 * Paint a pick at once and write it to the profile's config.yaml. Once the save
 * lands the cache follows, unless a newer save already answered, and that cache
 * write is how peer windows learn of it. A failed save repaints what the config
 * last said and says so, like every config-backed setting; only the newest pick
 * repaints or reports. A bare renderer (tests, the design preview) has no
 * backend: the pick is only cached.
 */
function commitPick(profile: string, field: AppearanceField, value: string, repaint: () => void): void {
  const pref = APPEARANCE_PREFS[field]

  if (!window.hermesDesktop) {
    pref.pick(profile, value)
    repaint()

    return
  }

  const owner = profileAppearanceOwner(profile)
  const key = pickKey(owner, field)
  const pick = { value }

  const settle = () => {
    const newest = pendingPicks.get(key) === pick

    if (newest) {
      pendingPicks.delete(key)
      cacheAdopted(profile)
      repaint()
    }

    return newest
  }

  pendingPicks.set(key, pick)
  repaint()
  saveProfileAppearance(profile, { [field]: value })
    .then(() => {
      // Unless a newer save already answered, this is the profile's config now.
      if (adoptedAppearance(owner, field) === value) {
        pref.pick(profile, value)
      }

      settle()
    })
    .catch(error => {
      if (settle()) {
        notifyError(error, translateNow('settings.config.autosaveFailed'))
      }
    })
}

// Profiles whose config was already checked for a local pick to upload.
const seededProfiles = new Set<string>()

/**
 * Existing installs kept their picks only in this origin's localStorage. The
 * first config load that finds the profile's appearance unset uploads the
 * profile's OWN pick once, so the Webapp and other clients inherit it. The
 * global value an unassigned named profile inherits was never chosen for it,
 * so it stays local.
 */
function seedFromLocalPick(appearance: { profile: string; theme: string; mode: string }): void {
  const { profile } = appearance

  if (seededProfiles.has(profile)) {
    return
  }

  seededProfiles.add(profile)

  const skin = appearance.theme ? null : skinPref.own(profile)
  const mode = appearance.mode ? null : modePref.own(profile)

  const patch: ProfileAppearancePatch = {
    ...(skin && !RETIRED_SKINS.has(skin) ? { theme: skin } : {}),
    ...(isThemeMode(mode) ? { theme_mode: mode } : {})
  }

  if (Object.keys(patch).length) {
    void saveProfileAppearance(profile, patch).catch(() => undefined)
  }
}

// ─── Color math (for synthesised light variants of dark-only skins) ────────
// mix / ensureContrast live in @hermes/shared/color (shared with the TUI);
// readableInk in ./color pins the desktop's near-black ink.

function synthLightColors(seed: DesktopTheme): DesktopThemeColors {
  const accent = seed.colors.ring || seed.colors.primary
  const soft = mix('#ffffff', accent, 0.1)
  const softer = mix('#ffffff', accent, 0.06)
  const border = mix('#ececef', accent, 0.14)
  const midground = seed.colors.midground ?? accent

  return {
    background: '#ffffff',
    foreground: '#161616',
    card: '#ffffff',
    cardForeground: '#161616',
    muted: softer,
    mutedForeground: mix('#6b6b70', accent, 0.16),
    popover: '#ffffff',
    popoverForeground: '#161616',
    primary: accent,
    primaryForeground: readableInk(accent),
    secondary: soft,
    secondaryForeground: mix('#2a2a2a', accent, 0.34),
    accent: soft,
    accentForeground: mix('#2a2a2a', accent, 0.34),
    border,
    input: mix('#e2e2e6', accent, 0.18),
    ring: accent,
    midground,
    midgroundForeground: readableInk(midground),
    destructive: '#b94a3a',
    destructiveForeground: '#ffffff',
    sidebarBackground: mix('#fafafa', accent, 0.05),
    sidebarBorder: border,
    userBubble: soft,
    userBubbleBorder: border
  }
}

/** Returns the seed palette for a given skin + mode (no overrides applied). */
export function getBaseColors(skinName: string, mode: 'light' | 'dark'): DesktopThemeColors {
  const seed = resolveTheme(skinName) ?? nousTheme

  if (mode === 'dark') {
    return seed.darkColors ?? seed.colors
  }

  return seed.darkColors ? seed.colors : synthLightColors(seed)
}

function deriveTheme(skinName: string, mode: 'light' | 'dark'): DesktopTheme {
  const seed = resolveTheme(skinName) ?? nousTheme

  return {
    ...seed,
    name: `${skinName}-${mode}`,
    label: `${seed.label} ${mode === 'light' ? 'Light' : 'Dark'}`,
    description: `${seed.label} ${mode} palette`,
    colors: getBaseColors(skinName, mode),
    // A backend skin named `default`/`mono`/… keeps the desktop's own palette
    // (never shadowed — see ingestBackendSkin), but its customCSS is carried
    // separately in $backendCustomCSS. The seed's own customCSS (non-built-in
    // backend skins) wins when both exist.
    customCSS: seed.customCSS ?? $backendCustomCSS.get()[skinName]
  }
}

/**
 * Some palettes intentionally keep a bright background even when
 * `mode === 'dark'`, so we shouldn't apply the `.dark` class. Decide from
 * the actual background luminance.
 */
function renderedModeFor(colors: DesktopThemeColors, mode: 'light' | 'dark'): 'light' | 'dark' {
  const rgb = parseColor(colors.background)

  if (!rgb) {
    return mode
  }

  const [r, g, b] = rgb.map(v => v / 255)

  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 ? 'light' : 'dark'
}

// ─── CSS application ────────────────────────────────────────────────────────

// Per-mode mix knobs. Light/dark fallbacks live in styles.css `:root` /
// `:root.dark`; setting them inline keeps active-skin overrides surviving
// the boot-time paint.
// styles.css --theme-neutral-chrome — keep in sync.
const NEUTRAL_CHROME = { light: '#f3f3f3', dark: '#0d0d0e' } as const

// The one foreground --dt-primary-solid is built to carry. Fixed rather than
// measured: the surface is derived to suit IT, not the other way round.
// styles.css --dt-primary-solid-foreground fallback — keep in sync.
const PRIMARY_SOLID_FOREGROUND = '#fcfcfc'

const chromeBackground = (background: string, isDark: boolean) =>
  mix(background, NEUTRAL_CHROME[isDark ? 'dark' : 'light'], isDark ? 0.26 : 0.08)

const mixesFor = (isDark: boolean): Record<string, string> => ({
  '--theme-mix-chrome': isDark ? '74%' : '92%',
  '--theme-mix-sidebar': '100%',
  '--theme-mix-card': isDark ? '38%' : '22%',
  '--theme-mix-elevated': isDark ? '46%' : '28%',
  '--theme-mix-bubble': isDark ? '46%' : '0%'
})

const TYPOGRAPHY_KNOB_VARS = {
  baseSize: '--dt-base-size',
  lineHeight: '--dt-line-height',
  letterSpacing: '--dt-letter-spacing'
} as const

// Optional typography knobs. They are the ONLY vars applyTheme may paint
// inline conditionally: styles.css declares the same fallbacks on :root, so
// a theme that stops providing one must drop the inline value — otherwise
// the previous skin's size/leading/tracking sticks across a switch (#41766).
function applyTypographyKnobs(root: HTMLElement, typo: Partial<DesktopThemeTypography>) {
  for (const [key, cssVar] of Object.entries(TYPOGRAPHY_KNOB_VARS) as [keyof typeof TYPOGRAPHY_KNOB_VARS, string][]) {
    const value = typo[key]

    if (value) {
      root.style.setProperty(cssVar, value)
    } else {
      root.style.removeProperty(cssVar)
    }
  }
}

function applyTheme(theme: DesktopTheme, mode: 'light' | 'dark', chatFontFamily = $chatFontFamily.get()) {
  if (typeof document === 'undefined') {
    return
  }

  const root = document.documentElement
  const c = theme.colors
  const typo = { ...DEFAULT_TYPOGRAPHY, ...nousTheme.typography, ...theme.typography }
  const rendered = renderedModeFor(c, mode)
  const isDark = rendered === 'dark'
  const midground = c.midground ?? c.ring
  const skinName = theme.name.endsWith(`-${mode}`) ? theme.name.slice(0, -mode.length - 1) : theme.name

  root.style.setProperty('color-scheme', rendered)
  root.dataset.hermesTheme = skinName
  root.dataset.hermesMode = rendered
  root.classList.toggle('dark', isDark)

  // Translucency is tuned per appearance, and "appearance" means the palette
  // actually painted — a skin that keeps a bright surface in "dark" wants
  // light's tint. Publishing from here covers the boot paint too, so the very
  // first resolved state main is told about is already the right one.
  setAppearance(rendered)

  // Brand seeds feed every glass + shadcn token via `color-mix()` in styles.css.
  const seeds: Record<string, string> = {
    '--theme-foreground': c.foreground,
    '--theme-primary': c.primary,
    '--theme-secondary': c.secondary,
    '--theme-accent-soft': c.accent,
    '--theme-midground': midground,
    '--theme-warm': c.primary,
    '--theme-background-seed': c.background,
    '--theme-sidebar-seed': c.sidebarBackground ?? c.background,
    '--theme-card-seed': c.card,
    '--theme-elevated-seed': c.popover,
    '--theme-bubble-seed': c.userBubble ?? c.popover
  }

  // shadcn/Tailwind tokens that aren't derived from the seed chain.
  const palette: Record<string, string> = {
    '--dt-primary-foreground': c.primaryForeground,
    '--dt-secondary-foreground': c.secondaryForeground,
    '--dt-accent-foreground': c.accentForeground,
    '--dt-border': c.border,
    '--dt-input': c.input,
    '--dt-ring': c.ring,
    '--dt-muted': c.muted,
    '--dt-midground-foreground': c.midgroundForeground ?? readableInk(midground),
    // A LOUD fill of the brand colour, for the rare surface that has to read as
    // the app speaking rather than as chrome. `primary` alone can't do that job:
    // a pale accent (imported VS Code themes love a pastel pink) is a perfectly
    // valid primary, and the honest `primaryForeground` for it is near-black —
    // so the "loud" surface comes out a pastel card with dark text on it,
    // whispering. Deepening the hue until the LIGHT foreground clears AA keeps
    // one look across every theme: no-ops on an accent that is already deep,
    // and only ever darkens, so the hue survives.
    '--dt-primary-solid': ensureContrast(c.primary, PRIMARY_SOLID_FOREGROUND, 4.5),
    '--dt-primary-solid-foreground': PRIMARY_SOLID_FOREGROUND,
    '--dt-composer-ring': c.composerRing ?? midground,
    '--dt-destructive': c.destructive,
    '--dt-destructive-foreground': c.destructiveForeground,
    '--dt-sidebar-border': c.sidebarBorder ?? c.border,
    '--dt-user-bubble-border': c.userBubbleBorder ?? c.border,
    // Semantic success, bent toward the accent so it settles into the palette
    // instead of clashing with it. A green accent barely moves it (see
    // `harmonize`); a blue one turns the sidebar's finished dots teal rather
    // than leaving eight emerald spots fighting the theme.
    '--ui-success': harmonize('#10b981', midground, 0.25),
    '--dt-font-sans': resolveChatFontFamily(chatFontFamily, typo.fontSans),
    '--dt-font-mono': typo.fontMono,
    '--noise-opacity-mul': isDark ? 'calc(0.04 / 0.21)' : 'calc(0.34 / 0.21)'
  }

  for (const [k, v] of Object.entries({ ...seeds, ...mixesFor(isDark), ...palette })) {
    root.style.setProperty(k, v)
  }

  applyTypographyKnobs(root, typo)

  const chromeBg = chromeBackground(c.background, isDark)

  window.hermesDesktop?.setTitleBarTheme?.({
    background: chromeBg,
    foreground: c.foreground
  })

  // Raw (non-JSON) keys read by the inline pre-paint script in index.html —
  // they let a brand-new window paint the themed background on its very first
  // frame, before this module has even loaded.
  try {
    window.localStorage.setItem('hermes-boot-background', chromeBg)
    window.localStorage.setItem('hermes-boot-color-scheme', rendered)
  } catch {
    // Storage may be unavailable (private mode / quota); the inline script
    // falls back to prefers-color-scheme.
  }

  if (typo.fontUrl && !INJECTED_FONT_URLS.has(typo.fontUrl)) {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = typo.fontUrl
    link.dataset.hermesThemeFont = 'true'
    document.head.appendChild(link)
    INJECTED_FONT_URLS.add(typo.fontUrl)
  }

  // Inject / clear customCSS from the skin (mirrors web/src/themes/context.tsx).
  // A theme carries the optional customCSS field; we inject/remove a single
  // <style> tag to keep the DOM clean and avoid stale rules on switch.
  const cssId = 'hermes-desktop-custom-css'
  let cssEl = document.getElementById(cssId) as HTMLStyleElement | null
  const customCSS = theme.customCSS?.trim()

  if (!customCSS) {
    if (cssEl) {
      cssEl.remove()
    }
  } else {
    if (!cssEl) {
      cssEl = document.createElement('style')
      cssEl.id = cssId
      cssEl.dataset.hermesSkinCSS = 'true'
      document.head.appendChild(cssEl)
    }

    cssEl.textContent = customCSS
  }
}

// Pin Electron's nativeTheme to the app's mode so the NATIVE window chrome
// (macOS vibrancy material, titlebar, pre-paint background) matches the app
// theme instead of the OS appearance. An explicit light/dark pick is forced;
// 'system' stays 'system' so prefers-color-scheme keeps tracking the OS.
const syncNativeTheme = (pref: ThemeMode, rendered: 'light' | 'dark') =>
  window.hermesDesktop?.setNativeTheme?.(pref === 'system' ? 'system' : rendered)

// Boot-time paint to avoid a flash before <ThemeProvider> mounts. Use the last
// active profile's appearance so a non-default profile relaunch paints its own
// skin + light/dark mode.
if (typeof window !== 'undefined') {
  const profile = BOOT_PROFILE_KEY
  const pref = modePref.resolve(profile)
  const resolved = resolveMode(pref)
  const theme = deriveTheme(normalizeSkin(storedSkin(profile)), resolved)
  applyTheme(theme, resolved)
  syncNativeTheme(pref, renderedModeFor(theme.colors, resolved))
}

// ─── Context ────────────────────────────────────────────────────────────────

interface ThemeContextValue {
  theme: DesktopTheme
  themeName: string
  mode: ThemeMode
  /** The light/dark switch the user picked. */
  resolvedMode: 'light' | 'dark'
  /**
   * The mode actually painted, derived from the active background's luminance.
   * Differs from `resolvedMode` for skins that keep a bright surface in "dark"
   * (or vice-versa). Surface-bound UI (e.g. the terminal palette) should key off
   * this so it matches what's on screen instead of inverting.
   */
  renderedMode: 'light' | 'dark'
  availableThemes: Array<{ name: string; label: string; description: string }>
  setTheme: (name: string) => void
  setMode: (mode: ThemeMode) => void
  /**
   * Paint a theme with an explicit light/dark, without persistence. This is
   * the highlight preview for the palette. A commit (`setTheme`) or
   * `clearThemePreview` repaints the committed appearance.
   */
  previewTheme: (name: string, mode: 'light' | 'dark') => void
  clearThemePreview: () => void
}

const SKIN_LIST = BUILTIN_THEME_LIST.map(({ name, label, description }) => ({ name, label, description }))

const ThemeContext = createContext<ThemeContextValue>({
  theme: nousTheme,
  themeName: DEFAULT_SKIN_NAME,
  mode: 'light',
  resolvedMode: 'light',
  renderedMode: 'light',
  availableThemes: SKIN_LIST,
  setTheme: () => {},
  setMode: () => {},
  previewTheme: () => {},
  clearThemePreview: () => {}
})

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Skin + mode are assigned per profile; the active profile drives which
  // appearance shows. Single-profile users only ever see "default", so their
  // behavior is unchanged.
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const connection = useStore($connection)
  // Before a gateway descriptor exists, the bridge is the only authoritative
  // profile for this window. Once one arrives, follow the live route as usual.
  const profileKey = normalizeProfileKey(connection?.profile ?? (connection ? activeGatewayProfile : BOOT_PROFILE_KEY))

  // Built-ins + user-installed + registry-contributed themes. Reactive so an
  // import or a plugin registration shows up live in the palette, settings
  // grid, and `/skin` without a reload.
  const userThemes = useStore($userThemes)
  const backendThemes = useStore($backendThemes)
  const backendCustomCSS = useStore($backendCustomCSS)
  const registryVersion = useStore($registryVersion)

  const availableThemes = useMemo(
    () =>
      listAllThemes().map(({ name, label, description }) => ({
        name,
        label,
        description
      })),
    // userThemes + backendThemes + registryVersion ARE listAllThemes' reactivity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [userThemes, backendThemes, registryVersion]
  )

  const [themeName, setThemeNameState] = useState(() =>
    typeof window === 'undefined' ? DEFAULT_SKIN_NAME : storedSkin(BOOT_PROFILE_KEY)
  )

  const [mode, setModeState] = useState<ThemeMode>(() =>
    typeof window === 'undefined' ? 'system' : modePref.resolve(BOOT_PROFILE_KEY)
  )

  // Remember the profile for the next boot's first paint.
  useEffect(() => rememberActiveProfileKey(profileKey), [profileKey])

  const repaint = useCallback((profile: string = liveProfile()) => {
    setThemeNameState(skinOf(profile))
    setModeState(modeOf(profile))
  }, [])

  // Follow profile switches and adopt the profile's config.yaml appearance:
  // it is the authority, so the cache follows it and it paints under any pick
  // still in flight here. An unset value leaves the local pick painted (and
  // may seed the config from it once).
  const configAppearance = useStore($profileAppearance)
  const appearanceOwner = profileAppearanceOwner(profileKey)
  const ownAppearance = configAppearance?.owner === appearanceOwner ? configAppearance : null

  useEffect(() => {
    cacheAdopted(profileKey)
    repaint(profileKey)

    if (ownAppearance) {
      seedFromLocalPick(ownAppearance)
    }
  }, [appearanceOwner, ownAppearance, profileKey, repaint])

  // Appearance is per-profile localStorage, and every desktop window is another
  // renderer on the same origin — so a switch made in the HUD (or any peer
  // window) only ever repainted the window it was made in. `storage` fires in
  // the OTHER windows, which is exactly the set that needs to catch up. A peer
  // caches a config value only once its save landed or it adopted a config, so
  // a re-read here orders it by revision; until then the cache paints only what
  // this window's config leaves unset.
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.storageArea && event.storageArea !== window.localStorage) {
        return
      }

      if (event.key === null || APPEARANCE_KEYS.has(event.key)) {
        repaint()
        void refreshProfileAppearance()
      }
    }

    window.addEventListener('storage', onStorage)

    return () => window.removeEventListener('storage', onStorage)
  }, [repaint])

  const systemDark = useMediaQuery('(prefers-color-scheme: dark)')
  const resolvedMode = resolveMode(mode, systemDark)

  // Transient highlight preview (palette theme picker). It is never
  // persisted. A commit or an explicit clear returns the paint to the
  // committed appearance.
  const [preview, setPreview] = useState<{ name: string; mode: 'light' | 'dark' } | null>(null)

  // The committed skin, resolved against the CURRENT registry — so a stored
  // backend skin that failed to resolve at boot paints once the gateway seeds it.
  const committedName = useMemo(
    () => normalizeSkin(themeName),
    // normalizeSkin resolves through the merged registry; the stores are its reactivity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [themeName, userThemes, backendThemes, registryVersion]
  )

  const paintedName = preview ? preview.name : committedName
  const paintedMode = preview ? preview.mode : resolvedMode

  const activeTheme = useMemo(
    () => deriveTheme(paintedName, paintedMode),
    // deriveTheme resolves its seed through the merged registry, so the theme
    // stores are its reactivity too — an in-place palette edit of the ACTIVE
    // skin (live theme authoring) must repaint, not just a name switch. The
    // backend CSS store matters the same way for built-in-named user skins.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [paintedName, paintedMode, userThemes, backendThemes, backendCustomCSS, registryVersion]
  )

  // Dev-only accent retint. `null` (always, in production) returns the theme
  // untouched, and retintTheme is an identity when the seed already matches —
  // so the picker costs nothing until it's actually moved off the default.
  const accentOverride = useStore($accentOverride)

  const paintedTheme = useMemo(
    () => (accentOverride === null ? activeTheme : retintTheme(activeTheme, accentOverride)),
    [activeTheme, accentOverride]
  )

  // What actually gets painted (matches the `.dark` class applyTheme toggles).
  const renderedMode = useMemo(() => renderedModeFor(paintedTheme.colors, paintedMode), [paintedTheme, paintedMode])

  // The chat face rides on the theme paint: the config-backed family is layered
  // in front of the theme's own stack, so an empty value is exactly the theme.
  const chatFontFamily = useStore($chatFontFamily)

  useEffect(() => applyTheme(paintedTheme, paintedMode, chatFontFamily), [paintedTheme, paintedMode, chatFontFamily])

  // Keep the native window appearance pinned to the app theme (vibrancy
  // material, titlebar, new-window pre-paint background).
  useEffect(() => syncNativeTheme(mode, renderedMode), [mode, renderedMode])

  const setTheme = useCallback(
    (name: string) => {
      recordFeatureUse('skins')
      setPreview(null)
      commitPick(liveProfile(), 'theme', normalizeSkin(name), repaint)
    },
    [repaint]
  )

  const setMode = useCallback(
    (next: ThemeMode) => {
      recordFeatureUse('skins')
      setPreview(null)
      commitPick(liveProfile(), 'theme_mode', next, repaint)
    },
    [repaint]
  )

  const previewTheme = useCallback((name: string, previewMode: 'light' | 'dark') => {
    setPreview(resolveTheme(name) ? { name, mode: previewMode } : null)
  }, [])

  const clearThemePreview = useCallback(() => setPreview(null), [])

  // Drain a backend-driven skin switch (Hermes authoring/activating a skin from a
  // prompt, or `/skin` on another surface). setTheme persists it to the profile's
  // config like any manual pick — which is why lifecycle.ts only lets through a
  // skin.changed tagged for the active profile.
  const pendingSkin = useStore($pendingSkinApply)

  useEffect(() => {
    if (pendingSkin) {
      setTheme(pendingSkin)
      $pendingSkinApply.set(null)
    }
  }, [pendingSkin, setTheme])

  // The light/dark toggle (Shift+X by default) is owned by the keybind runtime
  // (`appearance.toggleMode`) so it shows up in the hotkey map and is rebindable.

  const value = useMemo<ThemeContextValue>(
    () => ({
      theme: paintedTheme,
      themeName: committedName,
      mode,
      resolvedMode,
      renderedMode,
      availableThemes,
      setTheme,
      setMode,
      previewTheme,
      clearThemePreview
    }),
    [
      paintedTheme,
      committedName,
      mode,
      resolvedMode,
      renderedMode,
      availableThemes,
      setTheme,
      setMode,
      previewTheme,
      clearThemePreview
    ]
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = (): ThemeContextValue => useContext(ThemeContext)
