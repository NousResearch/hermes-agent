import type { ReadableAtom } from 'nanostores'

import { $backdrop, setBackdrop } from '@/store/backdrop'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $introSplash, setIntroSplash } from '@/store/intro-splash'
import { $reasoningCollapsedByDefault, setReasoningCollapsedByDefault } from '@/store/reasoning-disclosure'
import { $sessionListDensity, type SessionListDensity, setSessionListDensity } from '@/store/session-list-density'
import { $tabStripDefault, setTabStripDefault, type TabStripDefault } from '@/store/tabstrip-prefs'

export interface DesktopSettingValues {
  'backdrop.v1': boolean
  'composerPopout.gesturesEnabled': boolean
  'intro-splash.v1': boolean
  'reasoning.collapsedByDefault': boolean
  sessionListDensity: SessionListDensity
  tabStripDefault: TabStripDefault
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

const isSessionListDensity = (value: unknown): value is SessionListDensity =>
  value === 'compact' || value === 'comfortable' || value === 'detailed'

const isTabStripDefault = (value: unknown): value is TabStripDefault =>
  value === 'auto' || value === 'always' || value === 'never'

const settingBindings = {
  'backdrop.v1': bindSetting($backdrop, setBackdrop, isBoolean),
  'composerPopout.gesturesEnabled': bindSetting(
    $composerPopoutGesturesEnabled,
    setComposerPopoutGesturesEnabled,
    isBoolean
  ),
  'intro-splash.v1': bindSetting($introSplash, setIntroSplash, isBoolean),
  'reasoning.collapsedByDefault': bindSetting($reasoningCollapsedByDefault, setReasoningCollapsedByDefault, isBoolean),
  sessionListDensity: bindSetting($sessionListDensity, setSessionListDensity, isSessionListDensity),
  tabStripDefault: bindSetting($tabStripDefault, setTabStripDefault, isTabStripDefault)
} satisfies { [Key in DesktopSettingKey]: SettingBinding<DesktopSettingValues[Key]> }

const bindingsByKey = settingBindings as unknown as Record<string, SettingBinding<unknown>>

const bindingFor = (key: string): SettingBinding<unknown> => {
  const binding = bindingsByKey[key]

  if (!binding) {
    throw new Error(`Unsupported desktop setting: ${key}`)
  }

  return binding
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
