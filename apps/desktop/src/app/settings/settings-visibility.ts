import { type InterfaceMode, type InterfaceTier, shownInMode } from '@/store/interface-mode'

import { APPEARANCE_SUBPAGES } from './appearance-subpages'
import { CONFIG_SUBPAGES } from './config-subpages'
import { SECTIONS } from './constants'
import { OTHER_SUBPAGES } from './other-subpages'
import type { SettingsView } from './types'

export interface SettingsSubpage {
  id: string
  labelKey: string
  tier?: InterfaceTier
}

// Standalone (non-config) views that Simple mode hides; config sections carry
// their tier on SECTIONS. Hidden views stay routable in Advanced only — a
// Simple deep link lands on SIMPLE_FALLBACK_VIEW instead of a blank page.
const VIEW_TIERS: Partial<Record<SettingsView, InterfaceTier>> = {
  connections: 'advanced',
  gateway: 'advanced',
  keys: 'advanced',
  plugins: 'advanced',
  vault: 'advanced'
}

// Appearance hosts the interface-mode picker, so it is always reachable.
const SIMPLE_FALLBACK_VIEW: SettingsView = 'config:appearance'

function settingsViewTier(view: SettingsView): InterfaceTier | undefined {
  if (view.startsWith('config:')) {
    const id = view.slice('config:'.length)

    return SECTIONS.find(section => section.id === id)?.tier
  }

  return VIEW_TIERS[view]
}

export function settingsViewShown(view: SettingsView, mode: InterfaceMode): boolean {
  return shownInMode(mode)({ tier: settingsViewTier(view) })
}

// Technical rows on pages Simple keeps. Config fields match by key (or `prefix.`),
// hand-written rows by manifest setting id; the renderer and search both read this.
const SIMPLE_HIDDEN_SETTINGS = [
  'display.show_reasoning',
  'agent.image_input_mode',
  'voice.client_direct',
  'voice.gpt_live.',
  'appearance.model-pricing'
] as const

/** Whether a config field key or manifest setting id is shown in a mode. */
export function settingShownInMode(key: string, mode: InterfaceMode): boolean {
  return (
    mode !== 'simple' ||
    !SIMPLE_HIDDEN_SETTINGS.some(hidden => (hidden.endsWith('.') ? key.startsWith(hidden) : key === hidden))
  )
}

/** The view to render for a requested one: hidden views fall back to a visible page. */
export function resolveSettingsView(view: SettingsView, mode: InterfaceMode): SettingsView {
  return settingsViewShown(view, mode) ? view : SIMPLE_FALLBACK_VIEW
}

function allSettingsSubpages(view: SettingsView): readonly SettingsSubpage[] {
  if (view === 'config:appearance') {
    return APPEARANCE_SUBPAGES
  }

  if (view.startsWith('config:')) {
    return CONFIG_SUBPAGES[view.slice('config:'.length)] ?? []
  }

  return OTHER_SUBPAGES[view] ?? []
}

/** Ordered child pages of a view that the given mode shows. */
export function settingsSubpages(view: SettingsView, mode: InterfaceMode): readonly SettingsSubpage[] {
  return allSettingsSubpages(view).filter(shownInMode(mode))
}

/** Whether a view (and, when given, one of its child pages) is reachable in a mode. */
export function settingsPageShown(view: SettingsView, subpage: string | undefined, mode: InterfaceMode): boolean {
  if (!settingsViewShown(view, mode)) {
    return false
  }

  const page = subpage && allSettingsSubpages(view).find(candidate => candidate.id === subpage)

  return !page || shownInMode(mode)(page)
}
