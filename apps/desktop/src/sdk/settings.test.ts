import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { onPersistenceEvent } from '@/lib/storage'
import { host } from '@/sdk'
import { $backdrop, setBackdrop } from '@/store/backdrop'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'
import { $embedMode, setEmbedMode } from '@/store/embed-consent'
import { $interfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { $introSplash, setIntroSplash } from '@/store/intro-splash'
import { $reasoningCollapsedByDefault, setReasoningCollapsedByDefault } from '@/store/reasoning-disclosure'
import { $sessionListDensity, setSessionListDensity } from '@/store/session-list-density'
import { $hideThreadTimeline, setHideThreadTimeline } from '@/store/thread-timeline'
import { $hideCodeDiffs, $toolViewMode, setHideCodeDiffs, setToolViewMode } from '@/store/tool-view'
import { $titlebarAppActionsSide, setTitlebarAppActionsSide } from '@/store/titlebar-app-actions'
import { $userBubbleTransparency, setUserBubbleTransparency } from '@/store/user-bubble-transparency'
import { $tabStripDefault, setTabStripDefault } from '@/store/tabstrip-prefs'

const resetSettings = () => {
  setSessionListDensity('compact')
  setTabStripDefault('auto')
  setBackdrop(false)
  setIntroSplash(true)
  setReasoningCollapsedByDefault(false)
  setComposerPopoutGesturesEnabled(true)
  setEmbedMode('ask')
  setHideThreadTimeline(false)
  setInterfaceMode('advanced')
  setTitlebarAppActionsSide('right')
  setToolViewMode('product')
  setHideCodeDiffs(false)
  setUserBubbleTransparency(0)
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

  it('reads and writes the appearance follow-up keys through their owning stores (#121896)', () => {
    host.settings.set('embed-mode', 'off')
    host.settings.set('hideThreadTimeline', true)
    host.settings.set('interfaceMode.v1', 'simple')
    host.settings.set('titlebarAppActions', 'left')
    host.settings.set('toolView.technical', 'technical')
    host.settings.set('toolView.hideCodeDiffs', true)
    host.settings.set('user-bubble-transparency.v1', 42)

    expect(host.settings.get('embed-mode')).toBe('off')
    expect(host.settings.get('hideThreadTimeline')).toBe(true)
    expect(host.settings.get('interfaceMode.v1')).toBe('simple')
    expect(host.settings.get('titlebarAppActions')).toBe('left')
    expect(host.settings.get('toolView.technical')).toBe('technical')
    expect(host.settings.get('toolView.hideCodeDiffs')).toBe(true)
    expect(host.settings.get('user-bubble-transparency.v1')).toBe(42)

    expect($embedMode.get()).toBe('off')
    expect($hideThreadTimeline.get()).toBe(true)
    expect($interfaceMode.get()).toBe('simple')
    expect($titlebarAppActionsSide.get()).toBe('left')
    expect($toolViewMode.get()).toBe('technical')
    expect($hideCodeDiffs.get()).toBe(true)
    expect($userBubbleTransparency.get()).toBe(42)
  })

  it('preserves the follow-up keys persistence schema', () => {
    const writes: Array<[string, null | string]> = []

    const unsubscribe = onPersistenceEvent(event => {
      if (event.op !== 'read') {
        writes.push([event.key, event.value])
      }
    })

    host.settings.set('embed-mode', 'always')
    host.settings.set('hideThreadTimeline', true)
    host.settings.set('titlebarAppActions', 'left')
    host.settings.set('toolView.technical', 'technical')
    host.settings.set('toolView.hideCodeDiffs', true)
    host.settings.set('user-bubble-transparency.v1', 42)
    // interfaceMode last: the tool-view atoms are modeBound, and writes under
    // Simple mode reveal in memory instead of persisting the preference — the
    // same as a Settings-page click under Simple. The mode switch itself
    // persists and re-scopes layout keys (extra writes below are its doing).
    host.settings.set('interfaceMode.v1', 'simple')

    unsubscribe()

    expect(writes).toEqual(
      expect.arrayContaining([
        ['hermes.desktop.embed-mode', 'always'],
        ['hermes.desktop.hideThreadTimeline', 'true'],
        ['hermes.desktop.interfaceMode.v1', 'simple'],
        ['hermes.desktop.titlebarAppActions', 'left'],
        ['hermes.desktop.toolView.technical', 'true'],
        ['hermes.desktop.toolView.hideCodeDiffs', 'true'],
        ['hermes.desktop.user-bubble-transparency.v1', '42']
      ])
    )
    // Advanced is the absence of a mode: the codec writes no key, mirroring the
    // Settings page. Reset back to the default must not leave a stale record.
    host.settings.set('interfaceMode.v1', 'advanced')
  })

  it('refuses follow-up values outside each store shape', () => {
    expect(() => host.settings.set('embed-mode' as never, 'sometimes' as never)).toThrow(
      'Invalid value for desktop setting: embed-mode'
    )
    expect(() => host.settings.set('interfaceMode.v1' as never, 'expert' as never)).toThrow(
      'Invalid value for desktop setting: interfaceMode.v1'
    )
    // The gateway band-checks instead of clamping: off-band numbers are a bug,
    // not a request for the endpoint.
    expect(() => host.settings.set('user-bubble-transparency.v1' as never, 101 as never)).toThrow(
      'Invalid value for desktop setting: user-bubble-transparency.v1'
    )
    expect(() => host.settings.set('user-bubble-transparency.v1' as never, '60' as never)).toThrow(
      'Invalid value for desktop setting: user-bubble-transparency.v1'
    )
    expect(() => host.settings.set('toolView.technical' as never, true as never)).toThrow(
      'Invalid value for desktop setting: toolView.technical'
    )

    expect($embedMode.get()).toBe('ask')
    expect($interfaceMode.get()).toBe('advanced')
    expect($userBubbleTransparency.get()).toBe(0)
    expect($toolViewMode.get()).toBe('product')
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
