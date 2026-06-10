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

import { BUILTIN_THEME_LIST, BUILTIN_THEMES, DEFAULT_SKIN_NAME, DEFAULT_TYPOGRAPHY, nousTheme } from './presets'
import type { DesktopTheme, DesktopThemeColors } from './types'

// Legacy global skin (pre per-profile themes). Still the inheritance fallback
// for any profile without its own assignment, so single-profile users and old
// installs are unaffected.
const SKIN_KEY = 'hermes-desktop-theme-v2'
const MODE_KEY = 'hermes-desktop-mode-v1'
// Per-profile skin + light/dark mode assignments: { [profileKey]: value }. A
// profile inherits the global default until it's given its own appearance.
const PROFILE_SKINS_KEY = 'hermes-desktop-profile-themes-v1'
const PROFILE_MODES_KEY = 'hermes-desktop-profile-modes-v1'
// Last active profile, recorded so the boot-time paint can pick that profile's
// theme before the gateway reports which profile actually launched.
const LAST_PROFILE_KEY = 'hermes-desktop-active-profile-v1'
const RETIRED_SKINS = new Set(['nous-light', 'default', 'gold'])

export type ThemeMode = 'light' | 'dark' | 'system'

const INJECTED_FONT_URLS = new Set<string>()

const resolveMode = (mode: ThemeMode, systemDark = matchesQuery('(prefers-color-scheme: dark)')): 'light' | 'dark' =>
  mode === 'system' ? (systemDark ? 'dark' : 'light') : mode

const normalizeSkin = (name: string | null): string =>
  name && BUILTIN_THEMES[name] && !RETIRED_SKINS.has(name) ? name : DEFAULT_SKIN_NAME

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

function hexToRgb(hex: string): [number, number, number] | null {
  const clean = hex.trim().replace(/^#/, '')

  if (!/^[0-9a-f]{6}$/i.test(clean)) {
    return null
  }

  return [0, 2, 4].map(i => parseInt(clean.slice(i, i + 2), 16)) as [number, number, number]
}

const rgbToHex = ([r, g, b]: [number, number, number]) =>
  `#${[r, g, b].map(n => Math.round(n).toString(16).padStart(2, '0')).join('')}`

function mix(a: string, b: string, amount: number): string {
  const ar = hexToRgb(a)
  const br = hexToRgb(b)

  return ar && br
    ? rgbToHex([ar[0] + (br[0] - ar[0]) * amount, ar[1] + (br[1] - ar[1]) * amount, ar[2] + (br[2] - ar[2]) * amount])
    : a
}

function readableOn(hex: string): string {
  const rgb = hexToRgb(hex)

  if (!rgb) {
    return '#ffffff'
  }

  const [r, g, b] = rgb.map(v => {
    const c = v / 255

    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })

  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.58 ? '#161616' : '#ffffff'
}

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
  const seed = BUILTIN_THEMES[skinName] ?? nousTheme

  if (mode === 'dark') {
    return seed.darkColors ?? seed.colors
  }

  return seed.darkColors ? seed.colors : synthLightColors(seed)
}

function deriveTheme(skinName: string, mode: 'light' | 'dark'): DesktopTheme {
  const seed = BUILTIN_THEMES[skinName] ?? nousTheme

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

  window.hermesDesktop?.setTitleBarTheme?.({
    background: c.background,
    foreground: c.foreground
  })

  if (typo.fontUrl && !INJECTED_FONT_URLS.has(typo.fontUrl)) {
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = typo.fontUrl
    link.dataset.hermesThemeFont = 'true'
    document.head.appendChild(link)
    INJECTED_FONT_URLS.add(typo.fontUrl)
  }
}

// Boot-time paint to avoid a flash before <ThemeProvider> mounts. Use the last
// active profile's appearance so a non-default profile relaunch paints its own
// skin + light/dark mode.
if (typeof window !== 'undefined') {
  const profile = readBootProfileKey()
  const resolved = resolveMode(modePref.resolve(profile))
  applyTheme(deriveTheme(skinPref.resolve(profile), resolved), resolved)
}

// ─── Context ────────────────────────────────────────────────────────────────

interface ThemeContextValue {
  theme: DesktopTheme
  themeName: string
  mode: ThemeMode
  resolvedMode: 'light' | 'dark'
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
  availableThemes: SKIN_LIST,
  setTheme: () => {},
  setMode: () => {}
})

export function ThemeProvider({ children }: { children: ReactNode }) {
  // Skin + mode are assigned per profile; the active profile drives which
  // appearance shows. Single-profile users only ever see "default", so their
  // behavior is unchanged.
  const profileKey = normalizeProfileKey(useStore($activeGatewayProfile))

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

  useEffect(() => applyTheme(activeTheme, resolvedMode), [activeTheme, resolvedMode])

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
    () => ({ theme: activeTheme, themeName, mode, resolvedMode, availableThemes: SKIN_LIST, setTheme, setMode }),
    [activeTheme, themeName, mode, resolvedMode, setTheme, setMode]
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
