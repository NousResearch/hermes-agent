import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { onPersistenceEvent } from '@/lib/storage'
import { host } from '@/sdk'
import { $backdrop, setBackdrop } from '@/store/backdrop'
import { $chatTextScale, setChatTextScale } from '@/store/chat-text-scale'
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
import { $showModelPricing, setShowModelPricing } from '@/store/model-pricing'
import { $reactionsEnabled, setReactionsEnabled } from '@/store/reactions-enabled'
import { $textDirection, setTextDirection } from '@/store/text-direction'
import { $tipsEnabled, setTipsEnabled } from '@/store/tips'
import { $toursEnabled, setToursEnabled } from '@/store/tours'
import { $vibeHeartsEnabled, setVibeHeartsEnabled } from '@/store/vibe-hearts-enabled'

const resetSettings = () => {
  setSessionListDensity('compact')
  setTabStripDefault('auto')
  setBackdrop(false)
  setIntroSplash(true)
  setReasoningCollapsedByDefault(false)
  setComposerPopoutGesturesEnabled(true)
  setChatTextScale(110)
  setEmbedMode('ask')
  setHideThreadTimeline(false)
  setInterfaceMode('advanced')
  setTitlebarAppActionsSide('right')
  setToolViewMode('product')
  setHideCodeDiffs(false)
  setUserBubbleTransparency(0)
  setTextDirection('auto')
  setReactionsEnabled(false)
  setToursEnabled(true)
  setVibeHeartsEnabled(true)
  setTipsEnabled(true)
  setShowModelPricing(false)
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
    host.settings.set('chatTextScale', 125)

    expect(host.settings.get('sessionListDensity')).toBe('detailed')
    expect(host.settings.get('tabStripDefault')).toBe('always')
    expect(host.settings.get('backdrop.v1')).toBe(true)
    expect(host.settings.get('intro-splash.v1')).toBe(false)
    expect(host.settings.get('reasoning.collapsedByDefault')).toBe(true)
    expect(host.settings.get('composerPopout.gesturesEnabled')).toBe(false)
    expect(host.settings.get('chatTextScale')).toBe(125)

    expect($sessionListDensity.get()).toBe('detailed')
    expect($tabStripDefault.get()).toBe('always')
    expect($backdrop.get()).toBe(true)
    expect($introSplash.get()).toBe(false)
    expect($reasoningCollapsedByDefault.get()).toBe(true)
    expect($composerPopoutGesturesEnabled.get()).toBe(false)
    expect($chatTextScale.get()).toBe(125)
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
    host.settings.set('chatTextScale', 150)

    unsubscribe()

    expect(writes).toEqual(
      expect.arrayContaining([
        ['hermes.desktop.sessionListDensity', 'detailed'],
        ['hermes.desktop.tabStripDefault', 'never'],
        ['hermes.desktop.backdrop.v1', 'true'],
        ['hermes.desktop.intro-splash.v1', 'false'],
        ['hermes.desktop.reasoning.collapsedByDefault', 'true'],
        ['hermes.desktop.composerPopout.gesturesEnabled', 'false'],
        ['hermes.desktop.chat-text-scale.v1', '150']
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

  it('reads and writes the remaining Appearance rows through their owning stores (#121896)', () => {
    host.settings.set('textDirection', 'rtl')
    host.settings.set('reactions.v1', true)
    host.settings.set('tours.v1', false)
    host.settings.set('vibeHearts.v1', false)
    host.settings.set('tips.rotation.v1', false)
    host.settings.set('model-pricing.v1', true)

    expect(host.settings.get('textDirection')).toBe('rtl')
    expect(host.settings.get('reactions.v1')).toBe(true)
    expect(host.settings.get('tours.v1')).toBe(false)
    expect(host.settings.get('vibeHearts.v1')).toBe(false)
    expect(host.settings.get('tips.rotation.v1')).toBe(false)
    expect(host.settings.get('model-pricing.v1')).toBe(true)

    expect($textDirection.get()).toBe('rtl')
    expect($reactionsEnabled.get()).toBe(true)
    expect($toursEnabled.get()).toBe(false)
    expect($vibeHeartsEnabled.get()).toBe(false)
    expect($tipsEnabled.get()).toBe(false)
    expect($showModelPricing.get()).toBe(true)
  })

  it('preserves the remaining rows persistence schema', () => {
    const writes: Array<[string, null | string]> = []

    const unsubscribe = onPersistenceEvent(event => {
      if (event.op !== 'read') {
        writes.push([event.key, event.value])
      }
    })

    host.settings.set('textDirection', 'rtl')
    host.settings.set('reactions.v1', true)
    host.settings.set('tours.v1', false)
    host.settings.set('vibeHearts.v1', false)
    host.settings.set('tips.rotation.v1', false)
    host.settings.set('model-pricing.v1', true)

    unsubscribe()

    expect(writes).toEqual(
      expect.arrayContaining([
        // Auto is stored as absence, so 'rtl' is the value that has to land.
        ['hermes.desktop.textDirection', 'rtl'],
        // Reactions and vibe hearts predate the gateway and store 'on'/'off',
        // not booleans — the binding keeps each store's own encoding.
        ['hermes.desktop.reactions.v1', 'on'],
        ['hermes.desktop.vibeHearts.v1', 'off'],
        ['hermes.desktop.tours.v1', 'false'],
        ['hermes.desktop.tips.rotation.v1', 'false'],
        ['hermes.desktop.model-pricing.v1', 'true']
      ])
    )
  })

  it('refuses remaining rows outside each store shape', () => {
    expect(() => host.settings.set('textDirection' as never, 'sideways' as never)).toThrow(
      'Invalid value for desktop setting: textDirection'
    )
    expect(() => host.settings.set('reactions.v1' as never, 'on' as never)).toThrow(
      'Invalid value for desktop setting: reactions.v1'
    )
    expect(() => host.settings.set('tours.v1' as never, 1 as never)).toThrow(
      'Invalid value for desktop setting: tours.v1'
    )
    expect(() => host.settings.set('model-pricing.v1' as never, null as never)).toThrow(
      'Invalid value for desktop setting: model-pricing.v1'
    )

    expect($textDirection.get()).toBe('auto')
    expect($reactionsEnabled.get()).toBe(false)
    expect($toursEnabled.get()).toBe(true)
    expect($showModelPricing.get()).toBe(false)
  })

  it('rounds in-band bubble values the way the Settings-page lever does', () => {
    // Documented, not a bug: the gateway band-checks, the store rounds
    // (`clampIntensity` → Math.round). Pin it so a later "tighten the guard"
    // change has to be a decision, not a drive-by.
    host.settings.set('user-bubble-transparency.v1', 42.6)
    expect(host.settings.get('user-bubble-transparency.v1')).toBe(43)

    host.settings.set('user-bubble-transparency.v1', 0.4)
    expect(host.settings.get('user-bubble-transparency.v1')).toBe(0)
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

  it('drives the chat text size the native Appearance control drives', () => {
    const listener = vi.fn()
    const unsubscribe = host.settings.subscribe('chatTextScale', listener)

    expect(listener).toHaveBeenLastCalledWith(110)

    // A native Appearance change reaches plugin subscribers...
    setChatTextScale(90)
    expect(listener).toHaveBeenLastCalledWith(90)

    // ...and a plugin write applies the same CSS variable the chat reads.
    host.settings.set('chatTextScale', 175)
    expect(listener).toHaveBeenLastCalledWith(175)
    expect(document.documentElement.style.getPropertyValue('--chat-text-scale')).toBe('1.75')

    unsubscribe()
  })

  it('accepts only the native chat text size presets', () => {
    const setUnchecked = host.settings.set as (key: string, value: unknown) => void

    // Off-preset numbers are refused, not silently snapped: the store would
    // normalize 112 to the 110 default and the plugin would never learn why.
    for (const bad of [112, 0, 1000, Number.NaN, '125', null]) {
      expect(() => setUnchecked('chatTextScale', bad)).toThrow('Invalid value for desktop setting: chatTextScale')
    }

    expect($chatTextScale.get()).toBe(110)
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
