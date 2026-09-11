import { afterEach, describe, expect, it } from 'vitest'

import { IS_MAC } from '@/lib/keybinds/combo'

import {
  $inlineLinkOpenModifierHeld,
  $requireModifierToOpenInlineLinks,
  INLINE_LINK_GATED_ATTR,
  INLINE_LINK_MODIFIER_HELD_ATTR,
  INLINE_LINK_REQUIRE_MODIFIER_ATTR,
  INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY,
  isInlineLinkOpenModifier,
  setRequireModifierToOpenInlineLinks,
  shouldOpenInlineLink
} from './inline-link-open'

afterEach(() => {
  setRequireModifierToOpenInlineLinks(false)
  localStorage.removeItem(INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY)
})

describe('inline link open preference', () => {
  it('defaults to single-click open', () => {
    expect($requireModifierToOpenInlineLinks.get()).toBe(false)
    expect(shouldOpenInlineLink({ ctrlKey: false, detail: 1, metaKey: false })).toBe(true)
  })

  it('persists through the shared boolean storage key', () => {
    setRequireModifierToOpenInlineLinks(true)

    expect(localStorage.getItem(INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY)).toBe('true')
    expect($requireModifierToOpenInlineLinks.get()).toBe(true)
  })

  it('uses ⌘ on Mac and Ctrl elsewhere as the open modifier', () => {
    expect(isInlineLinkOpenModifier({ ctrlKey: true, metaKey: false })).toBe(!IS_MAC)
    expect(isInlineLinkOpenModifier({ ctrlKey: false, metaKey: true })).toBe(IS_MAC)
    expect(isInlineLinkOpenModifier({ ctrlKey: false, metaKey: false })).toBe(false)
  })

  it('blocks a regular mouse click when the setting is on', () => {
    setRequireModifierToOpenInlineLinks(true)

    expect(shouldOpenInlineLink({ ctrlKey: false, detail: 1, metaKey: false })).toBe(false)
  })

  it('opens on the platform modifier when the setting is on', () => {
    setRequireModifierToOpenInlineLinks(true)

    expect(shouldOpenInlineLink({ ctrlKey: !IS_MAC, detail: 1, metaKey: IS_MAC })).toBe(true)
  })

  it('still opens a keyboard-generated click when the setting is on', () => {
    setRequireModifierToOpenInlineLinks(true)

    expect(shouldOpenInlineLink({ ctrlKey: false, detail: 0, metaKey: false })).toBe(true)
  })
})

function dispatchPlatformModifier(type: 'keydown' | 'keyup') {
  const event = IS_MAC
    ? new KeyboardEvent(type, { bubbles: true, key: 'Meta', metaKey: type === 'keydown' })
    : new KeyboardEvent(type, { bubbles: true, ctrlKey: type === 'keydown', key: 'Control' })

  window.dispatchEvent(event)
}

describe('inline link modifier cursor tracking', () => {
  afterEach(() => {
    setRequireModifierToOpenInlineLinks(false)
    localStorage.removeItem(INLINE_LINK_REQUIRE_MODIFIER_STORAGE_KEY)
  })

  it('does not flag the document or listen while the preference is off', () => {
    dispatchPlatformModifier('keydown')

    expect($inlineLinkOpenModifierHeld.get()).toBe(false)
    expect(document.documentElement.hasAttribute(INLINE_LINK_REQUIRE_MODIFIER_ATTR)).toBe(false)
    expect(document.documentElement.hasAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)).toBe(false)
  })

  it('flags the document when the preference turns on, without treating the modifier as held', () => {
    setRequireModifierToOpenInlineLinks(true)

    expect(document.documentElement.getAttribute(INLINE_LINK_REQUIRE_MODIFIER_ATTR)).toBe('true')
    expect(document.documentElement.hasAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)).toBe(false)
    expect($inlineLinkOpenModifierHeld.get()).toBe(false)
  })

  it('marks the modifier held on keydown and clears it on keyup', () => {
    setRequireModifierToOpenInlineLinks(true)

    dispatchPlatformModifier('keydown')

    expect($inlineLinkOpenModifierHeld.get()).toBe(true)
    expect(document.documentElement.getAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)).toBe('held')

    dispatchPlatformModifier('keyup')

    expect($inlineLinkOpenModifierHeld.get()).toBe(false)
    expect(document.documentElement.hasAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)).toBe(false)
  })

  it('clears a stuck held state on window blur', () => {
    setRequireModifierToOpenInlineLinks(true)
    dispatchPlatformModifier('keydown')

    expect($inlineLinkOpenModifierHeld.get()).toBe(true)

    window.dispatchEvent(new Event('blur'))

    expect($inlineLinkOpenModifierHeld.get()).toBe(false)
    expect(document.documentElement.hasAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)).toBe(false)
  })

  it('clears flags when the preference turns off', () => {
    setRequireModifierToOpenInlineLinks(true)
    dispatchPlatformModifier('keydown')
    setRequireModifierToOpenInlineLinks(false)

    expect($inlineLinkOpenModifierHeld.get()).toBe(false)
    expect(document.documentElement.hasAttribute(INLINE_LINK_REQUIRE_MODIFIER_ATTR)).toBe(false)
    expect(document.documentElement.hasAttribute(INLINE_LINK_MODIFIER_HELD_ATTR)).toBe(false)

    dispatchPlatformModifier('keydown')
    expect($inlineLinkOpenModifierHeld.get()).toBe(false)
  })

  it('exports a stable gated-link attribute name', () => {
    expect(INLINE_LINK_GATED_ATTR).toBe('data-inline-link-gated')
  })
})
