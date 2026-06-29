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

import { matchesQuery, useMediaQuery } from '@/hooks/use-media-query'
import { persistString, persistStringRecord, storedString, storedStringRecord } from '@/lib/storage'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

import { hexToRgb, mix, readableOn } from './color'
import { BUILTIN_THEME_LIST, BUILTIN_THEMES, DEFAULT_SKIN_NAME, DEFAULT_TYPOGRAPHY, nousTheme, resolveAlias } from './presets'
import type { DesktopTheme, DesktopThemeColors } from './types'
import { $userThemes, resolveTheme } from './user-themes'

// Legacy global skin (pre per-profile themes). Still the inheritance fallback
// for any profile without its own assignment, so single-profile users and old
// installs are unaffected.
export const SKIN_KEY = 'hermes-desktop-theme-v2'
const MODE_KEY = 'hermes-desktop-mode-v1'
// Per-profile skin + light/dark mode assignments: { [profileKey]: value }. A
// profile inherits the global default until it's given its own appearance.
export const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'
const PROFILE_MODES_KEY = 'hermes-desktop-profile-modes-v1'
// Last active profile, recorded so the boot-time paint can pick that profile's
// theme before the gateway reports which profile actually launched.
const LAST_PROFILE_KEY = 'hermes-desktop-active-profile-v1'
const RETIRED_SKINS = new Set(['nous', 'nous-light', 'default', 'gold', 'hermes'])

export type ThemeMode = 'light' | 'dark' | 'system'

const INJECTED_FONT_URLS = new Set<string>()

const resolveMode = (mode: ThemeMode, systemDark = matchesQuery('(prefers-color-scheme: dark)')): 'light' | 'dark' =>
  mode === 'system' ? (systemDark ? 'dark' : 'light') : mode

export const normalizeSkin = (name: string | null): string => {
  const resolved = name ? resolveAlias(name) : DEFAULT_SKIN_NAME
  const canonical = resolveTheme(resolved) && !RETIRED_SKINS.has(resolved) ? resolved : DEFAULT_SKIN_NAME

  // Persist the canonical name back to localStorage when a migration applies
  // so legacy storage gets cleaned on first read, not just shadowed at runtime.
  // Writes both the legacy global slot (hermes-desktop-theme-v2) and any
  // per-profile entries (PROFILE_SKINS_KEY JSON map) that still hold a
  // retired/alias name. One-way safe: idempotent, never invents user data.
  if (name && name !== canonical) {
    try {
      persistString(SKIN_KEY, canonical)
      const profiles = storedStringRecord(PROFILE_SKINS_KEY)
      let mutated = false

      for (const [key, value] of Object.entries(profiles)) {
        const entryResolved = value ? resolveAlias(value) : DEFAULT_SKIN_NAME

        const entryCanonical =
          resolveTheme(entryResolved) && !RETIRED_SKINS.has(entryResolved) ? entryResolved : DEFAULT_SKIN_NAME

        if (value !== entryCanonical) {
          profiles[key] = entryCanonical
          mutated = true
        }
      }

      if (mutated) {
        persistStringRecord(PROFILE_SKINS_KEY, profiles)
      }
    } catch {
      // Storage may be unavailable (private mode / quota); migration is best-effort.
    }
  }

  return canonical
}

const normalizeMode = (value: string | null): ThemeMode =>
  value === 'light' || value === 'dark' || value === 'system' ? value : 'light'

// ─── Per-profile appearance persistence ─────────────────────────────────────
// Skin and mode are each stored per profile. "default" isn't a real profile —
// it *is* the legacy global slot, so it reads/writes the global directly. Named
// profiles get their own entry and fall back to that global until assigned, so
// unassigned profiles and pre-per-profile installs stay on the global value.
const profilePref = <T extends string>(record: string, legacy: string, normalize: (v: string | null) => T) => ({
  resolve: (profile: string): T => normalize(storedStringRecord(record)[profile] ?? storedString(legacy)),
  assign: (profile: string, value: T): void => {
    if (profile === 'default') {
      persistString(legacy, value)
    } else {
      persistStringRecord(record, { ...storedStringRecord(record), [profile]: value })
    }
  }
})

export const skinPref = profilePref(PROFILE_SKINS_KEY, SKIN_KEY, normalizeSkin)
export const modePref = profilePref(PROFILE_MODES_KEY, MODE_KEY, normalizeMode)

// Last active profile — lets the boot paint pick its appearance before the
// gateway reports which profile actually launched.
const readBootProfileKey = () => normalizeProfileKey(storedString(LAST_PROFILE_KEY))
const rememberActiveProfileKey = (profile: string) => persistString(LAST_PROFILE_KEY, profile)

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

export function deriveTheme(skinName: string, mode: 'light' | 'dark'): DesktopTheme {
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

const chromeBackground = (background: string, isDark: boolean) =>
  mix(background, NEUTRAL_CHROME[isDark ? 'dark' : 'light'], isDark ? 0.26 : 0.08)

const mixesFor = (isDark: boolean): Record<string, string> => ({
  '--theme-mix-chrome': isDark ? '74%' : '92%',
  '--theme-mix-sidebar': '100%',
  '--theme-mix-card': isDark ? '38%' : '22%',
  '--theme-mix-elevated': isDark ? '46%' : '28%',
  '--theme-mix-bubble': isDark ? '46%' : '0%'
})

export function applyTheme(theme: DesktopTheme, mode: 'light' | 'dark') {
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
    '--dt-composer-ring': c.composerRing ?? midground,
    '--dt-destructive': c.destructive,
    '--dt-destructive-foreground': c.destructiveForeground,
    '--dt-sidebar-border': c.sidebarBorder ?? c.border,
    '--dt-user-bubble-border': c.userBubbleBorder ?? c.border,
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

  // Theme-specific CSS overlays (e.g., Deluxe Dark animations, special effects)
  const overlayId = `hermes-theme-overlay-${skinName}`
  const existingOverlay = document.getElementById(overlayId)

  if (skinName === 'bubblegum') {
    if (!existingOverlay) {
      const style = document.createElement('style')
      style.id = overlayId
      style.textContent = `
/* ── Animations ────────────────────────────────────────────── */
@keyframes bubblegum-shimmer {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}
@keyframes bubblegum-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255,20,147,.55), 0 0 24px rgba(255,77,154,.35), 0 0 48px rgba(0,229,255,.15); }
  50% { box-shadow: 0 0 0 6px rgba(255,20,147,0), 0 0 36px rgba(255,77,154,.55), 0 0 64px rgba(0,229,255,.3); }
}
@keyframes bubblegum-float {
  0%, 100% { transform: translateY(0) rotate(0); }
  50% { transform: translateY(-2px) rotate(.5deg); }
}
@keyframes bubblegum-sparkle-drift {
  0% { background-position: 0% 0%, 0% 0%, 0% 0%; }
  100% { background-position: 100% 100%, -100% 50%, 50% -100%; }
}

/* ── Light Mode ─────────────────────────────────────────── */
[data-hermes-theme="bubblegum"][data-hermes-mode="light"] {
  background-color: #FFF0F8 !important;
  background-image:
    radial-gradient(1.5px 1.5px at 20% 30%, rgba(255,105,180,.6), transparent 50%),
    radial-gradient(1px 1px at 70% 80%, rgba(255,77,154,.5), transparent 50%),
    radial-gradient(1.5px 1.5px at 40% 70%, rgba(0,229,255,.4), transparent 50%),
    radial-gradient(1px 1px at 85% 20%, rgba(255,20,147,.5), transparent 50%),
    radial-gradient(2px 2px at 10% 90%, rgba(255,255,255,.3), transparent 50%),
    linear-gradient(135deg, #FFF0F8 0%, #FFE6F0 25%, #FFCCE0 50%, #FFE6F0 75%, #FFF0F8 100%) !important;
  background-size: 400px 400px, 350px 350px, 300px 300px, 280px 280px, 260px 260px, 400% 400% !important;
  animation: bubblegum-shimmer 18s ease infinite, bubblegum-sparkle-drift 60s linear infinite !important;
  color: #3A0F26 !important;
}

/* ── Dark Mode ──────────────────────────────────────────── */
[data-hermes-theme="bubblegum"][data-hermes-mode="dark"] {
  background-color: #2D0E1B !important;
  background-image:
    radial-gradient(1.5px 1.5px at 20% 30%, rgba(255,105,180,.6), transparent 50%),
    radial-gradient(1px 1px at 70% 80%, rgba(255,77,154,.5), transparent 50%),
    radial-gradient(1.5px 1.5px at 40% 70%, rgba(0,229,255,.4), transparent 50%),
    radial-gradient(1px 1px at 85% 20%, rgba(255,20,147,.5), transparent 50%),
    radial-gradient(2px 2px at 10% 90%, rgba(255,255,255,.2), transparent 50%),
    radial-gradient(1.5px 1.5px at 50% 50%, rgba(255,105,180,.3), transparent 50%),
    linear-gradient(135deg, #2D0E1B 0%, #3D1228 25%, #4D1635 50%, #3D1228 75%, #2D0E1B 100%) !important;
  background-size: 400px 400px, 350px 350px, 300px 300px, 280px 280px, 260px 260px, 200px 200px, 400% 400% !important;
  background-attachment: fixed !important;
  animation: bubblegum-shimmer 18s ease infinite, bubblegum-sparkle-drift 60s linear infinite !important;
  color: #FFE4F1 !important;
}

/* ── Card & UI elements ─────────────────────────────────── */
[data-hermes-theme="bubblegum"] .rounded-md.border:not(button):not(input):not(textarea),
[data-hermes-theme="bubblegum"] .card {
  background: rgba(255, 248, 252, 0.7) !important;
  backdrop-filter: blur(14px) saturate(1.4) !important;
  border-color: rgba(255, 46, 131, 0.4) !important;
}
[data-hermes-theme="bubblegum"][data-hermes-mode="dark"] .rounded-md.border:not(button):not(input):not(textarea),
[data-hermes-theme="bubblegum"][data-hermes-mode="dark"] .card {
  background: rgba(61, 18, 40, 0.7) !important;
}

/* ── Buttons ────────────────────────────────────────────── */
[data-hermes-theme="bubblegum"] button[class*="bg-primary"],
[data-hermes-theme="bubblegum"] button[data-variant="default"]:not([disabled]) {
  background: linear-gradient(135deg, #FF2E83 0%, #FF66BB 50%, #00E5FF 100%) !important;
  background-size: 200% 100% !important;
  color: white !important;
  box-shadow: 0 4px 18px rgba(255, 20, 147, 0.45) !important;
  transition: all .25s ease !important;
}
[data-hermes-theme="bubblegum"] button[class*="bg-primary"]:hover,
[data-hermes-theme="bubblegum"] button[data-variant="default"]:not([disabled]):hover {
  background-position: 100% 50% !important;
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 8px 28px rgba(255, 20, 147, 0.65) !important;
}

/* ── Input focus ───────────────────────────────────────── */
[data-hermes-theme="bubblegum"] input:focus,
[data-hermes-theme="bubblegum"] textarea:focus {
  border-color: #FF2E83 !important;
  box-shadow: 0 0 0 3px rgba(255, 20, 147, 0.4) !important;
}

/* ── Text selection ─────────────────────────────────────── */
[data-hermes-theme="bubblegum"] ::selection {
  background: rgba(255, 20, 147, 0.4) !important;
  color: #FFFFFF !important;
}

/* ── Title gradient ───────────────────────────────────── */
[data-hermes-theme="bubblegum"] h1[class*="title"],
[data-hermes-theme="bubblegum"] [class*="brand"] {
  background: linear-gradient(135deg, #FF2E83 0%, #FF66BB 40%, #00E5FF 80%, #FF2E83 100%) !important;
  background-size: 200% 100% !important;
  -webkit-background-clip: text !important;
  background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  animation: bubblegum-shimmer 6s ease infinite !important;
}

/* ── Scrollbar ────────────────────────────────────────── */
[data-hermes-theme="bubblegum"] ::-webkit-scrollbar { width: 10px; height: 10px; }
[data-hermes-theme="bubblegum"] ::-webkit-scrollbar-track { background: rgba(255, 20, 147, 0.15) !important; border-radius: 8px; }
[data-hermes-theme="bubblegum"] ::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #FF2E83, #FF66BB, #00E5FF) !important;
  border-radius: 8px !important;
}
`
      document.head.appendChild(style)
    }
  } else {
    // Remove the bubblegum overlay when switching away from the canonical skin
    // so decorative effects (gradient, sparkles, pink scrollbar) don't linger.
    document.getElementById('hermes-theme-overlay-bubblegum')?.remove()
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
  const profile = readBootProfileKey()
  const pref = modePref.resolve(profile)
  const resolved = resolveMode(pref)
  const theme = deriveTheme(skinPref.resolve(profile), resolved)
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
  setMode: () => {}
})

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Skin + mode are assigned per profile; the active profile drives which
  // appearance shows. Single-profile users only ever see "default", so their
  // behavior is unchanged.
  const profileKey = normalizeProfileKey(useStore($activeGatewayProfile))

  // Built-ins + user-installed themes. Reactive so an import shows up live in
  // the palette, settings grid, and `/skin` without a reload.
  const userThemes = useStore($userThemes)

  const availableThemes = useMemo(
    () =>
      [...Object.values(BUILTIN_THEMES), ...Object.values(userThemes)].map(({ name, label, description }) => ({
        name,
        label,
        description
      })),
    [userThemes]
  )

  const [themeName, setThemeNameState] = useState(() =>
    typeof window === 'undefined' ? DEFAULT_SKIN_NAME : skinPref.resolve(readBootProfileKey())
  )

  const [mode, setModeState] = useState<ThemeMode>(() =>
    typeof window === 'undefined' ? 'light' : modePref.resolve(readBootProfileKey())
  )

  // Follow profile switches: paint the profile's assigned skin + mode and
  // remember it for the next boot's first paint.
  useEffect(() => {
    rememberActiveProfileKey(profileKey)
    setThemeNameState(skinPref.resolve(profileKey))
    setModeState(modePref.resolve(profileKey))
  }, [profileKey])

  const systemDark = useMediaQuery('(prefers-color-scheme: dark)')
  const resolvedMode = resolveMode(mode, systemDark)
  const activeTheme = useMemo(() => deriveTheme(themeName, resolvedMode), [themeName, resolvedMode])

  // What actually gets painted (matches the `.dark` class applyTheme toggles).
  const renderedMode = useMemo(() => renderedModeFor(activeTheme.colors, resolvedMode), [activeTheme, resolvedMode])

  useEffect(() => applyTheme(activeTheme, resolvedMode), [activeTheme, resolvedMode])

  // Keep the native window appearance pinned to the app theme (vibrancy
  // material, titlebar, new-window pre-paint background).
  useEffect(() => syncNativeTheme(mode, renderedMode), [mode, renderedMode])

  // Assign to whichever profile is live right now (read fresh so the callbacks
  // stay stable across profile switches).
  const liveProfile = () => normalizeProfileKey($activeGatewayProfile.get())

  const setTheme = useCallback((name: string) => {
    const next = normalizeSkin(name)
    setThemeNameState(next)
    skinPref.assign(liveProfile(), next)
  }, [])

  const setMode = useCallback((next: ThemeMode) => {
    setModeState(next)
    modePref.assign(liveProfile(), next)
  }, [])

  // The light/dark toggle (Shift+X by default) is owned by the keybind runtime
  // (`appearance.toggleMode`) so it shows up in the hotkey map and is rebindable.

  const value = useMemo<ThemeContextValue>(
    () => ({ theme: activeTheme, themeName, mode, resolvedMode, renderedMode, availableThemes, setTheme, setMode }),
    [activeTheme, themeName, mode, resolvedMode, renderedMode, availableThemes, setTheme, setMode]
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = (): ThemeContextValue => useContext(ThemeContext)

/** Sync the desktop skin with the active Hermes backend theme on connect. */
export function useSyncThemeFromBackend(backendThemeName: string | undefined, setTheme: (name: string) => void) {
  useEffect(() => {
    if (backendThemeName && BUILTIN_THEMES[backendThemeName]) {
      setTheme(backendThemeName)
    }
  }, [backendThemeName, setTheme])
}
