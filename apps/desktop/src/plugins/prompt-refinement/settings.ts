import { atom, type ReadableAtom } from 'nanostores'

import {
  REFINEMENT_DEPTHS,
  REFINEMENT_STYLES,
  type RefinementDepth,
  type RefinementMode,
  type RefinementModelSource,
  type RefinementSettings,
  type RefinementStyle,
  type RefineRequest
} from './types'

/** The one request both modes send. Custom guidance rides along only with
 *  the Custom style, and context only when given, so an unrelated preference
 *  never changes what the model sees (or misses the reuse cache). */
export function requestFor(
  settings: RefinementSettings,
  text: string,
  attachmentLabels: string[],
  context?: string
): RefineRequest {
  return {
    text,
    attachmentLabels,
    style: settings.style,
    depth: settings.autoDepth ? 'auto' : settings.depth,
    ...(settings.style === 'custom' && settings.customStyle.trim() ? { customStyle: settings.customStyle.trim() } : {}),
    ...(context ? { context } : {})
  }
}

/** Persisted under the plugin's storage namespace. The key names its scope
 *  (global: one preference for every profile and window) and its schema
 *  version, so v1.1 can migrate `v1` → `v2` without guessing. */
export const SETTINGS_KEY = 'v1.global.settings'

export const DEFAULT_SETTINGS: RefinementSettings = {
  version: 1,
  // Ships Off: nothing changes for a send until the user picks a mode.
  mode: 'off',
  modelSource: 'dedicated',
  style: 'auto',
  depth: 'deepen',
  // The refiner picks the depth that fits each prompt until the user fixes one.
  autoDepth: true,
  customStyle: '',
  useChatContext: false,
  minWords: 4
}

/** Longest custom style guidance kept (it rides along with every refinement). */
export const MAX_CUSTOM_STYLE_CHARS = 500

const MODES: readonly RefinementMode[] = ['off', 'manual', 'auto']
const SOURCES: readonly RefinementModelSource[] = ['dedicated', 'chat']
const MAX_MIN_WORDS = 50

/** Validate a stored value field by field; anything unrecognized falls back to
 *  the default for that field (never to a more active mode). */
export function normalizeSettings(raw: unknown): RefinementSettings {
  const value = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const mode = MODES.includes(value.mode as RefinementMode) ? (value.mode as RefinementMode) : DEFAULT_SETTINGS.mode

  const modelSource = SOURCES.includes(value.modelSource as RefinementModelSource)
    ? (value.modelSource as RefinementModelSource)
    : DEFAULT_SETTINGS.modelSource

  const depth = REFINEMENT_DEPTHS.includes(value.depth as RefinementDepth)
    ? (value.depth as RefinementDepth)
    : DEFAULT_SETTINGS.depth

  const style = REFINEMENT_STYLES.includes(value.style as RefinementStyle)
    ? (value.style as RefinementStyle)
    : DEFAULT_SETTINGS.style

  const customStyle =
    typeof value.customStyle === 'string'
      ? value.customStyle.slice(0, MAX_CUSTOM_STYLE_CHARS)
      : DEFAULT_SETTINGS.customStyle

  const useChatContext =
    typeof value.useChatContext === 'boolean' ? value.useChatContext : DEFAULT_SETTINGS.useChatContext

  const autoDepth = typeof value.autoDepth === 'boolean' ? value.autoDepth : DEFAULT_SETTINGS.autoDepth

  const minWords =
    typeof value.minWords === 'number' && Number.isInteger(value.minWords)
      ? Math.min(MAX_MIN_WORDS, Math.max(1, value.minWords))
      : DEFAULT_SETTINGS.minWords

  return { version: 1, mode, modelSource, style, depth, autoDepth, customStyle, useChatContext, minWords }
}

export interface SettingsStorage {
  get: (key: string, fallback: unknown) => unknown
  set: (key: string, value: unknown) => void
}

export interface SettingsStore {
  $settings: ReadableAtom<RefinementSettings>
  get: () => RefinementSettings
  update: (patch: Partial<Omit<RefinementSettings, 'version'>>) => void
  /** Re-read storage (another window wrote it). */
  reload: () => void
}

export function createSettingsStore(storage: SettingsStorage): SettingsStore {
  const read = () => normalizeSettings(storage.get(SETTINGS_KEY, null))
  const $settings = atom<RefinementSettings>(read())

  return {
    $settings,
    get: () => $settings.get(),
    update: patch => {
      const next = normalizeSettings({ ...$settings.get(), ...patch })
      storage.set(SETTINGS_KEY, next)
      $settings.set(next)
    },
    reload: () => {
      const next = read()
      const current = $settings.get()

      if (JSON.stringify(next) !== JSON.stringify(current)) {
        $settings.set(next)
      }
    }
  }
}
