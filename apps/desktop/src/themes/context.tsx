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

import { useStore } from '@nanostores/react'
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react'

import { $registryVersion } from '@/contrib/registry'
import { matchesQuery, useMediaQuery } from '@/hooks/use-media-query'
import { persistString, storedString, storedStringRecord } from '@/lib/storage'
import { setAppearance } from '@/store/translucency'

import { $accentOverride } from './accent-override'
import { $backendThemes, $pendingSkinApply } from './backend-sync'
import { ensureContrast, harmonize, hexToRgb, mix, readableOn } from './color'
import { BUILTIN_THEME_LIST, DEFAULT_SKIN_NAME, DEFAULT_TYPOGRAPHY, nousTheme } from './presets'
import { retintTheme } from './retint'
import type { DesktopTheme, DesktopThemeColors } from './types'
import { $userThemes, listAllThemes, resolveTheme } from './user-themes'

// Desktop-wide appearance. Profiles are independent agent environments, but
// the desktop shell is one application and must keep one consistent palette.
const SKIN_KEY = 'hermes-desktop-theme-v2'
const MODE_KEY = 'hermes-desktop-mode-v1'
// Legacy per-profile assignments are retained only as a one-time migration
// source. New writes always target the desktop-wide keys above.
const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'
const PROFILE_MODES_KEY = 'hermes-desktop-profile-modes-v1'
const LAST_PROFILE_KEY = 'hermes-desktop-active-profile-v1'
const GLOBAL_APPEARANCE_MIGRATION_KEY = 'team-hermes-global-appearance-v1'
// Skins that no longer exist. A profile still pointing at one falls back to
// DEFAULT_SKIN_NAME rather than painting a name nothing resolves.
const RETIRED_SKINS = new Set(['nous-light', 'default', 'gold'])

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
const normalizeMode = (value: string | null): ThemeMode =>
  value === 'light' || value === 'dark' || value === 'system' ? value : 'system'

// ─── Desktop-wide appearance persistence ────────────────────────────────────
// Older builds stored appearance per profile. On the first Team Hermes boot,
// prefer the last active profile's explicit choices so the appearance visible
// immediately before upgrading becomes the new global desktop appearance.
function migrateGlobalAppearance(): void {
  if (storedString(GLOBAL_APPEARANCE_MIGRATION_KEY) === '1') {
    return
  }

  const lastProfile = storedString(LAST_PROFILE_KEY) || 'default'
  const legacySkin = storedStringRecord(PROFILE_SKINS_KEY)[lastProfile]
  const legacyMode = storedStringRecord(PROFILE_MODES_KEY)[lastProfile]

  if (legacySkin) {
    persistString(SKIN_KEY, legacySkin)
  }

  if (legacyMode) {
    persistString(MODE_KEY, legacyMode)
  }

  persistString(GLOBAL_APPEARANCE_MIGRATION_KEY, '1')
}

const desktopPref = <T extends string>(key: string, normalize: (v: string | null) => T) => ({
  resolve: (_profile = 'desktop'): T => {
    migrateGlobalAppearance()

    return normalize(storedString(key))
  },
  assign: (_profile: string, value: T): void => {
    migrateGlobalAppearance()
    persistString(key, value)
  }
})

// Keep the profile-shaped API for profile bundle compatibility. The profile
// parameter is intentionally ignored: importing or changing appearance updates
// the one desktop shell shared by every profile.
export const skinPref = desktopPref(SKIN_KEY, normalizeSkin)
export const modePref = desktopPref(MODE_KEY, normalizeMode)

/** Everything a peer window could change that this one has to repaint for. */
const APPEARANCE_KEYS = new Set([SKIN_KEY, MODE_KEY])

// ─── Color math (for synthesised light variants of dark-only skins) ────────
// hexToRgb / mix / readableOn live in ./color so the VS Code converter shares
// the exact same math.

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
    primaryForeground: readableOn(accent),
    secondary: soft,
    secondaryForeground: mix('#2a2a2a', accent, 0.34),
    accent: soft,
    accentForeground: mix('#2a2a2a', accent, 0.34),
    border,
    input: mix('#e2e2e6', accent, 0.18),
    ring: accent,
    midground,
    midgroundForeground: readableOn(midground),
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
    colors: getBaseColors(skinName, mode)
  }
}

/**
 * Some palettes intentionally keep a bright background even when
 * `mode === 'dark'`, so we shouldn't apply the `.dark` class. Decide from
 * the actual background luminance.
 */
function renderedModeFor(colors: DesktopThemeColors, mode: 'light' | 'dark'): 'light' | 'dark' {
  const rgb = hexToRgb(colors.background)

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

function applyTheme(theme: DesktopTheme, mode: 'light' | 'dark') {
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
    '--dt-midground-foreground': c.midgroundForeground ?? readableOn(midground),
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
    '--dt-font-sans': typo.fontSans,
    '--dt-font-mono': typo.fontMono,
    '--noise-opacity-mul': isDark ? 'calc(0.04 / 0.21)' : 'calc(0.34 / 0.21)'
  }

  for (const [k, v] of Object.entries({ ...seeds, ...mixesFor(isDark), ...palette })) {
    root.style.setProperty(k, v)
  }

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
}

// Pin Electron's nativeTheme to the app's mode so the NATIVE window chrome
// (macOS vibrancy material, titlebar, pre-paint background) matches the app
// theme instead of the OS appearance. An explicit light/dark pick is forced;
// 'system' stays 'system' so prefers-color-scheme keeps tracking the OS.
const syncNativeTheme = (pref: ThemeMode, rendered: 'light' | 'dark') =>
  window.hermesDesktop?.setNativeTheme?.(pref === 'system' ? 'system' : rendered)

// Boot-time paint to avoid a flash before <ThemeProvider> mounts.
if (typeof window !== 'undefined') {
  const pref = modePref.resolve()
  const resolved = resolveMode(pref)
  const theme = deriveTheme(skinPref.resolve(), resolved)
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
  // Built-ins + user-installed + registry-contributed themes. Reactive so an
  // import or a plugin registration shows up live in the palette, settings
  // grid, and `/skin` without a reload.
  const userThemes = useStore($userThemes)
  const backendThemes = useStore($backendThemes)
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
    typeof window === 'undefined' ? DEFAULT_SKIN_NAME : skinPref.resolve()
  )

  const [mode, setModeState] = useState<ThemeMode>(() =>
    typeof window === 'undefined' ? 'system' : modePref.resolve()
  )

  // Every desktop window is another renderer on the same origin. `storage`
  // fires in peer windows, keeping the main window, HUD, and popouts aligned.
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key && !APPEARANCE_KEYS.has(event.key)) {
        return
      }

      setThemeNameState(skinPref.resolve())
      setModeState(modePref.resolve())
    }

    window.addEventListener('storage', onStorage)

    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const systemDark = useMediaQuery('(prefers-color-scheme: dark)')
  const resolvedMode = resolveMode(mode, systemDark)

  // Transient highlight preview (palette theme picker). It is never
  // persisted. A commit or an explicit clear returns the paint to the
  // committed appearance.
  const [preview, setPreview] = useState<{ name: string; mode: 'light' | 'dark' } | null>(null)

  const paintedName = preview ? preview.name : themeName
  const paintedMode = preview ? preview.mode : resolvedMode

  const activeTheme = useMemo(
    () => deriveTheme(paintedName, paintedMode),
    // deriveTheme resolves its seed through the merged registry, so the theme
    // stores are its reactivity too — an in-place palette edit of the ACTIVE
    // skin (live theme authoring) must repaint, not just a name switch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [paintedName, paintedMode, userThemes, backendThemes, registryVersion]
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

  useEffect(() => applyTheme(paintedTheme, paintedMode), [paintedTheme, paintedMode])

  // Keep the native window appearance pinned to the app theme (vibrancy
  // material, titlebar, new-window pre-paint background).
  useEffect(() => syncNativeTheme(mode, renderedMode), [mode, renderedMode])

  const setTheme = useCallback((name: string) => {
    const next = normalizeSkin(name)
    setPreview(null)
    setThemeNameState(next)
    skinPref.assign('desktop', next)
  }, [])

  const setMode = useCallback((next: ThemeMode) => {
    setPreview(null)
    setModeState(next)
    modePref.assign('desktop', next)
  }, [])

  const previewTheme = useCallback((name: string, previewMode: 'light' | 'dark') => {
    setPreview(resolveTheme(name) ? { name, mode: previewMode } : null)
  }, [])

  const clearThemePreview = useCallback(() => setPreview(null), [])

  // Drain a backend-driven skin switch (Hermes authoring/activating a skin from a
  // prompt, or `/skin` on another surface). setTheme persists it globally, so
  // every profile and window adopts the same desktop appearance.
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
      themeName,
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
      themeName,
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
