import type { ReadableAtom } from 'nanostores'

import { TRANSLUCENCY_MAX, TRANSLUCENCY_MIN } from '@hermes/shared/translucency'

import { $backdrop, setBackdrop } from '@/store/backdrop'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $embedMode, type EmbedMode, setEmbedMode } from '@/store/embed-consent'
import { $interfaceMode, type InterfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { $introSplash, setIntroSplash } from '@/store/intro-splash'
import { $reasoningCollapsedByDefault, setReasoningCollapsedByDefault } from '@/store/reasoning-disclosure'
import { $sessionListDensity, type SessionListDensity, setSessionListDensity } from '@/store/session-list-density'
import { $hideThreadTimeline, setHideThreadTimeline } from '@/store/thread-timeline'
import { $hideCodeDiffs, setHideCodeDiffs, $toolViewMode, setToolViewMode, type ToolViewMode } from '@/store/tool-view'
import {
  $titlebarAppActionsSide,
  setTitlebarAppActionsSide,
  type TitlebarAppActionsSide
} from '@/store/titlebar-app-actions'
import { $userBubbleTransparency, setUserBubbleTransparency } from '@/store/user-bubble-transparency'
import { $tabStripDefault, setTabStripDefault, type TabStripDefault } from '@/store/tabstrip-prefs'

export interface DesktopSettingValues {
  'backdrop.v1': boolean
  'composerPopout.gesturesEnabled': boolean
  // Appearance-page follow-ups (#121896 item A): every key below is bound to
  // the same atom + setter the native Settings page uses, on the page's own
  // storage-key name. `toolView.technical` keeps the historical storage key
  // even though the store holds the 'product' | 'technical' enum — the
  // binding follows the store's shape, not the legacy boolean.
  'embed-mode': EmbedMode
  hideThreadTimeline: boolean
  'interfaceMode.v1': InterfaceMode
  'intro-splash.v1': boolean
  'reasoning.collapsedByDefault': boolean
  sessionListDensity: SessionListDensity
  tabStripDefault: TabStripDefault
  titlebarAppActions: TitlebarAppActionsSide
  'toolView.hideCodeDiffs': boolean
  'toolView.technical': ToolViewMode
  'user-bubble-transparency.v1': number
}

export type DesktopSettingKey = keyof DesktopSettingValues

interface SettingBinding<T> {
  accepts(value: unknown): value is T
  get(): T
  set(value: T): void
  subscribe(listener: (value: T) => void): () => void
}

const bindSetting = <T>(
  $value: ReadableAtom<T>,
  set: (value: T) => void,
  accepts: (value: unknown) => value is T
): SettingBinding<T> => ({
  accepts,
  get: () => $value.get(),
  set,
  subscribe: listener => $value.subscribe(value => listener(value))
})

const isBoolean = (value: unknown): value is boolean => typeof value === 'boolean'

const isEmbedMode = (value: unknown): value is EmbedMode =>
  value === 'ask' || value === 'always' || value === 'off'

const isInterfaceMode = (value: unknown): value is InterfaceMode => value === 'simple' || value === 'advanced'

const isSessionListDensity = (value: unknown): value is SessionListDensity =>
  value === 'compact' || value === 'comfortable' || value === 'detailed'

const isTabStripDefault = (value: unknown): value is TabStripDefault =>
  value === 'auto' || value === 'always' || value === 'never'

const isTitlebarAppActionsSide = (value: unknown): value is TitlebarAppActionsSide =>
  value === 'left' || value === 'right'

const isToolViewMode = (value: unknown): value is ToolViewMode => value === 'product' || value === 'technical'

// Same 0–100 band the Settings-page lever uses (apps/shared translucency).
// Unlike `setUserBubbleTransparency`, which clamps anything numeric, the
// gateway refuses out-of-band and non-numeric values outright — a plugin bug
// must not silently land on an endpoint.
const isBubbleTransparency = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= TRANSLUCENCY_MIN && value <= TRANSLUCENCY_MAX

const settingBindings = {
  'backdrop.v1': bindSetting($backdrop, setBackdrop, isBoolean),
  'composerPopout.gesturesEnabled': bindSetting(
    $composerPopoutGesturesEnabled,
    setComposerPopoutGesturesEnabled,
    isBoolean
  ),
  'embed-mode': bindSetting($embedMode, setEmbedMode, isEmbedMode),
  hideThreadTimeline: bindSetting($hideThreadTimeline, setHideThreadTimeline, isBoolean),
  // Bound through the exported setter, never the raw atom: a mode change also
  // re-scopes layout persistence (modeLayout.change) — the Settings-page
  // click is the whole side effect, for plugins too.
  'interfaceMode.v1': bindSetting($interfaceMode, setInterfaceMode, isInterfaceMode),
  'intro-splash.v1': bindSetting($introSplash, setIntroSplash, isBoolean),
  'reasoning.collapsedByDefault': bindSetting($reasoningCollapsedByDefault, setReasoningCollapsedByDefault, isBoolean),
  sessionListDensity: bindSetting($sessionListDensity, setSessionListDensity, isSessionListDensity),
  tabStripDefault: bindSetting($tabStripDefault, setTabStripDefault, isTabStripDefault),
  titlebarAppActions: bindSetting($titlebarAppActionsSide, setTitlebarAppActionsSide, isTitlebarAppActionsSide),
  'toolView.hideCodeDiffs': bindSetting($hideCodeDiffs, setHideCodeDiffs, isBoolean),
  // Enum, not the legacy boolean the storage key's name suggests (see above).
  // modeBound writes land as user intent: policy-shadowed in Simple mode,
  // revealed on first write — exactly how the Settings page behaves.
  'toolView.technical': bindSetting($toolViewMode, setToolViewMode, isToolViewMode),
  'user-bubble-transparency.v1': bindSetting($userBubbleTransparency, setUserBubbleTransparency, isBubbleTransparency)
} satisfies { [Key in DesktopSettingKey]: SettingBinding<DesktopSettingValues[Key]> }

const bindingsByKey = settingBindings as unknown as Record<string, SettingBinding<unknown>>

const bindingFor = (key: string): SettingBinding<unknown> => {
  // Own keys only: `toString`/`constructor` would otherwise resolve to
  // `Object.prototype` functions and TypeError instead of being refused.
  if (!Object.hasOwn(settingBindings, key)) {
    throw new Error(`Unsupported desktop setting: ${key}`)
  }

  return bindingsByKey[key]
}

export const desktopSettings = {
  get<Key extends DesktopSettingKey>(key: Key): DesktopSettingValues[Key] {
    return bindingFor(key).get() as DesktopSettingValues[Key]
  },

  set<Key extends DesktopSettingKey>(key: Key, value: DesktopSettingValues[Key]): void {
    const binding = bindingFor(key)

    if (!binding.accepts(value)) {
      throw new Error(`Invalid value for desktop setting: ${key}`)
    }

    binding.set(value)
  },

  subscribe<Key extends DesktopSettingKey>(key: Key, listener: (value: DesktopSettingValues[Key]) => void): () => void {
    return bindingFor(key).subscribe(value => listener(value as DesktopSettingValues[Key]))
  }
}
