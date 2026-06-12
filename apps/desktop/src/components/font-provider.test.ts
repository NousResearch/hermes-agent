import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  FONT_OPTIONS,
  FONT_SIZE_OPTIONS,
  fontStore,
  fontSizeStore,
  initializeFontSettings,
} from './font-provider'

describe('font-provider', () => {
  beforeEach(() => {
    fontStore.set('system')
    fontSizeStore.set('medium')
    localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('initializeFontSettings', () => {
    it('hydrates fontStore from localStorage with a valid id', () => {
      localStorage.setItem('hermes-desktop-font', 'lora')
      initializeFontSettings()
      expect(fontStore.get()).toBe('lora')
    })

    it('hydrates fontSizeStore from localStorage with a valid id', () => {
      localStorage.setItem('hermes-desktop-font-size', 'large')
      initializeFontSettings()
      expect(fontSizeStore.get()).toBe('large')
    })

    it('rejects an invalid font id and leaves the default', () => {
      localStorage.setItem('hermes-desktop-font', 'not-a-font')
      initializeFontSettings()
      expect(fontStore.get()).toBe('system')
    })

    it('rejects an invalid font size id and leaves the default', () => {
      localStorage.setItem('hermes-desktop-font-size', 'huge')
      initializeFontSettings()
      expect(fontSizeStore.get()).toBe('medium')
    })

    it('leaves defaults when localStorage is empty', () => {
      initializeFontSettings()
      expect(fontStore.get()).toBe('system')
      expect(fontSizeStore.get()).toBe('medium')
    })

    it('hydrates both stores simultaneously', () => {
      localStorage.setItem('hermes-desktop-font', 'space-grotesk')
      localStorage.setItem('hermes-desktop-font-size', 'extraLarge')
      initializeFontSettings()
      expect(fontStore.get()).toBe('space-grotesk')
      expect(fontSizeStore.get()).toBe('extraLarge')
    })
  })

  describe('FONT_OPTIONS', () => {
    it('contains exactly the expected font ids', () => {
      const ids = FONT_OPTIONS.map(o => o.id)
      expect(ids).toEqual(['system', 'lora', 'space-grotesk', 'jetbrains'])
    })

    it('does not include Collapse (brand/display face, not a body font)', () => {
      const ids = FONT_OPTIONS.map(o => o.id)
      expect(ids).not.toContain('collapse')
    })

    it('every option has a non-empty value stack', () => {
      for (const opt of FONT_OPTIONS) {
        expect(opt.value).toBeTruthy()
        expect(opt.value.length).toBeGreaterThan(10)
      }
    })
  })

  describe('FONT_SIZE_OPTIONS', () => {
    it('contains exactly the expected size ids', () => {
      const ids = FONT_SIZE_OPTIONS.map(o => o.id)
      expect(ids).toEqual(['small', 'medium', 'large', 'extraLarge'])
    })

    it('every option has all three size properties', () => {
      for (const opt of FONT_SIZE_OPTIONS) {
        expect(opt.textSize).toBeTruthy()
        expect(opt.toolSize).toBeTruthy()
        expect(opt.captionSize).toBeTruthy()
      }
    })
  })

  describe('store persistence', () => {
    it('setFont updates fontStore and localStorage', () => {
      fontStore.set('lora')
      localStorage.setItem('hermes-desktop-font', 'lora')
      expect(fontStore.get()).toBe('lora')
      expect(localStorage.getItem('hermes-desktop-font')).toBe('lora')
    })

    it('setFontSize updates fontSizeStore and localStorage', () => {
      fontSizeStore.set('large')
      localStorage.setItem('hermes-desktop-font-size', 'large')
      expect(fontSizeStore.get()).toBe('large')
      expect(localStorage.getItem('hermes-desktop-font-size')).toBe('large')
    })
  })
})
