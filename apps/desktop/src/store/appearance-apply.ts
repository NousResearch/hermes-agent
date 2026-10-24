/**
 * What `set_theme` and its picker cards change: accent, light/dark mode and
 * layout preset, validated as a whole before anything is applied so an
 * unknown layout never leaves a new accent half-applied.
 *
 * Accent persists through `persistAccent` and mode through `requestMode`
 * (both drained by the ThemeProvider); a layout applies through the same
 * registry the layout picker reads, switching Simple/Advanced to the preset's
 * tier first because Simple only shows its two sidebar presets.
 */

import { isLayoutNode } from '@/components/pane-shell/tree/model'
import { layoutPresetTier, LAYOUTS_AREA } from '@/components/pane-shell/tree/presets'
import { registry } from '@/contrib/registry'
import { matchesQuery } from '@/hooks/use-media-query'
import { ACCENT_SWATCH_IDS, resolveAccentPick, type ResolvedAccent } from '@/themes/accents'
import type { ThemeMode } from '@/themes/context'
import { persistAccent } from '@/themes/persist-accent'
import { requestMode } from '@/themes/request'

import { $interfaceMode, setInterfaceMode } from './interface-mode'
import { applyDesktopLayoutPreset, openLayoutPickPanes } from './pane-focus'

export const THEME_MODES: readonly ThemeMode[] = ['light', 'dark', 'system']

export interface AppearanceValues {
  accent?: string
  layout?: string
  mode?: ThemeMode
}

/** What was applied, in the ids the agent reads: `accent` is a swatch id, `custom:#rrggbb`, or null for the theme's own. */
export interface AppliedAppearance {
  accent?: null | string
  layout?: string
  mode?: ThemeMode
}

export type AppearanceAnswer =
  { applied: AppliedAppearance; success: true } | { available: string[]; error: string; success: false }

const listedLayoutIds = (): string[] =>
  registry
    .getArea(LAYOUTS_AREA)
    .filter(entry => isLayoutNode(entry.data))
    .map(entry => entry.id)

// Mono is black on a light surface and white on a dark one: resolve against the surface the pick lands on.
function paintsDark(mode: ThemeMode | undefined): boolean {
  if (mode === 'system') {
    return matchesQuery('(prefers-color-scheme: dark)')
  }

  return mode === undefined ? document.documentElement.classList.contains('dark') : mode === 'dark'
}

// The first-run demo layout is borrowed (components/onboarding/first-run/intro.ts registers this): a
// committed layout ends it first, so a Simple/Advanced switch saves the user's own layout, not the demo.
let releaseBorrowedLayout = (): void => {}

export function onLayoutCommit(release: () => void): void {
  releaseBorrowedLayout = release
}

function applyLayout(id: string): void {
  releaseBorrowedLayout()

  // A user's saved layouts are Advanced, like the layout picker's Custom section.
  const mode = layoutPresetTier(id) === 'simple' ? 'simple' : 'advanced'

  if ($interfaceMode.get() !== mode) {
    setInterfaceMode(mode)
  }

  if (applyDesktopLayoutPreset(id)) {
    openLayoutPickPanes(id)
  }
}

export function applyAppearance(values: AppearanceValues): AppearanceAnswer {
  if (values.mode !== undefined && !THEME_MODES.includes(values.mode)) {
    return { available: [...THEME_MODES], error: `Unknown mode "${values.mode}".`, success: false }
  }

  let accent: ResolvedAccent | undefined

  if (values.accent !== undefined) {
    accent = resolveAccentPick(values.accent, paintsDark(values.mode))

    if (!accent) {
      return { available: [...ACCENT_SWATCH_IDS, 'theme'], error: `Unknown accent "${values.accent}".`, success: false }
    }
  }

  if (values.layout !== undefined && !listedLayoutIds().includes(values.layout)) {
    return { available: listedLayoutIds(), error: `Unknown layout "${values.layout}".`, success: false }
  }

  const applied: AppliedAppearance = {}

  if (values.mode !== undefined) {
    requestMode(values.mode)
    applied.mode = values.mode
  }

  if (accent) {
    persistAccent(accent.hex)
    applied.accent = accent.id
  }

  if (values.layout !== undefined) {
    applyLayout(values.layout)
    applied.layout = values.layout
  }

  return { applied, success: true }
}
