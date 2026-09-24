import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { onPersistenceEvent } from '@/lib/storage'
import { host } from '@/sdk'
import { $backdrop, setBackdrop } from '@/store/backdrop'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $introSplash, setIntroSplash } from '@/store/intro-splash'
import { $reasoningCollapsedByDefault, setReasoningCollapsedByDefault } from '@/store/reasoning-disclosure'
import { $sessionListDensity, setSessionListDensity } from '@/store/session-list-density'
import { $tabStripDefault, setTabStripDefault } from '@/store/tabstrip-prefs'

const resetSettings = () => {
  setSessionListDensity('compact')
  setTabStripDefault('auto')
  setBackdrop(false)
  setIntroSplash(true)
  setReasoningCollapsedByDefault(false)
  setComposerPopoutGesturesEnabled(true)
}

describe('host.settings', () => {
  beforeEach(() => {
    resetSettings()
  })

  afterEach(resetSettings)

  it('reads and writes the allowlisted typed preferences through their owning stores', () => {
    host.settings.set('sessionListDensity', 'detailed')
    host.settings.set('tabStripDefault', 'always')
    host.settings.set('backdrop.v1', true)
    host.settings.set('intro-splash.v1', false)
    host.settings.set('reasoning.collapsedByDefault', true)
    host.settings.set('composerPopout.gesturesEnabled', false)

    expect(host.settings.get('sessionListDensity')).toBe('detailed')
    expect(host.settings.get('tabStripDefault')).toBe('always')
    expect(host.settings.get('backdrop.v1')).toBe(true)
    expect(host.settings.get('intro-splash.v1')).toBe(false)
    expect(host.settings.get('reasoning.collapsedByDefault')).toBe(true)
    expect(host.settings.get('composerPopout.gesturesEnabled')).toBe(false)

    expect($sessionListDensity.get()).toBe('detailed')
    expect($tabStripDefault.get()).toBe('always')
    expect($backdrop.get()).toBe(true)
    expect($introSplash.get()).toBe(false)
    expect($reasoningCollapsedByDefault.get()).toBe(true)
    expect($composerPopoutGesturesEnabled.get()).toBe(false)
  })

  it('preserves the stores existing persistence schema', () => {
    const writes: Array<[string, null | string]> = []

    const unsubscribe = onPersistenceEvent(event => {
      if (event.op !== 'read') {
        writes.push([event.key, event.value])
      }
    })

    host.settings.set('sessionListDensity', 'detailed')
    host.settings.set('tabStripDefault', 'never')
    host.settings.set('backdrop.v1', true)
    host.settings.set('intro-splash.v1', false)
    host.settings.set('reasoning.collapsedByDefault', true)
    host.settings.set('composerPopout.gesturesEnabled', false)

    unsubscribe()

    expect(writes).toEqual(
      expect.arrayContaining([
        ['hermes.desktop.sessionListDensity', 'detailed'],
        ['hermes.desktop.tabStripDefault', 'never'],
        ['hermes.desktop.backdrop.v1', 'true'],
        ['hermes.desktop.intro-splash.v1', 'false'],
        ['hermes.desktop.reasoning.collapsedByDefault', 'true'],
        ['hermes.desktop.composerPopout.gesturesEnabled', 'false']
      ])
    )
  })

  it('subscribes immediately and follows changes from the native settings surface', () => {
    const listener = vi.fn()
    const unsubscribe = host.settings.subscribe('backdrop.v1', listener)

    expect(listener).toHaveBeenLastCalledWith(false)

    setBackdrop(true)

    expect(listener).toHaveBeenLastCalledWith(true)
    expect(listener).toHaveBeenCalledTimes(2)

    unsubscribe()
    setBackdrop(false)

    expect(listener).toHaveBeenCalledTimes(2)
  })

  it('rejects keys and values outside the public allowlist', () => {
    expect(() => (host.settings.get as (key: string) => unknown)('pluginDecisions.v2')).toThrow(
      'Unsupported desktop setting: pluginDecisions.v2'
    )
    // Inherited keys are not settings: a plain-object lookup would hand back
    // `Function.prototype.toString` and TypeError on `.get()`.
    expect(() => (host.settings.get as (key: string) => unknown)('toString')).toThrow(
      'Unsupported desktop setting: toString'
    )
    expect(() => (host.settings.set as (key: string, value: unknown) => void)('constructor', true)).toThrow(
      'Unsupported desktop setting: constructor'
    )
    expect(() => (host.settings.set as (key: string, value: unknown) => void)('backdrop.v1', 'on')).toThrow(
      'Invalid value for desktop setting: backdrop.v1'
    )

    expect($backdrop.get()).toBe(false)
  })
})
