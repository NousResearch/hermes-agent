// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $interfaceStyle,
  applyInterfaceStyleToDocument,
  DEFAULT_INTERFACE_STYLE,
  installInterfaceStyleSync,
  INTERFACE_STYLE_STORAGE_KEY,
  normalizeInterfaceStyle,
  setInterfaceStyleEnabled,
  setInterfaceStylePart
} from './interface-style'

const ATTRIBUTES = [
  'data-hermes-interface-style',
  'data-hermes-interface-typography',
  'data-hermes-interface-controls',
  'data-hermes-interface-surfaces'
] as const

describe('GNOME interface style preference', () => {
  beforeEach(() => {
    localStorage.clear()
    $interfaceStyle.set({ ...DEFAULT_INTERFACE_STYLE })

    for (const attribute of ATTRIBUTES) {
      document.documentElement.removeAttribute(attribute)
    }
  })

  afterEach(() => localStorage.clear())

  it('keeps the original Hermes look as the clean-install default', () => {
    expect(DEFAULT_INTERFACE_STYLE).toEqual({
      controls: true,
      enabled: false,
      surfaces: true,
      typography: true
    })

    applyInterfaceStyleToDocument(DEFAULT_INTERFACE_STYLE)
    expect(ATTRIBUTES.every(attribute => !document.documentElement.hasAttribute(attribute))).toBe(true)
  })

  it('normalizes partial or malformed saved values without losing sub-option defaults', () => {
    expect(normalizeInterfaceStyle({ enabled: true, controls: false })).toEqual({
      controls: false,
      enabled: true,
      surfaces: true,
      typography: true
    })
    expect(normalizeInterfaceStyle('invalid')).toEqual(DEFAULT_INTERFACE_STYLE)
  })

  it('persists the master switch and each independently selectable part', () => {
    setInterfaceStyleEnabled(true)
    setInterfaceStylePart('surfaces', false)

    expect($interfaceStyle.get()).toEqual({
      controls: true,
      enabled: true,
      surfaces: false,
      typography: true
    })
    expect(JSON.parse(localStorage.getItem(INTERFACE_STYLE_STORAGE_KEY) ?? 'null')).toEqual($interfaceStyle.get())
  })

  it('publishes only enabled parts as root attributes and removes them when disabled', () => {
    const stop = installInterfaceStyleSync()

    try {
      setInterfaceStyleEnabled(true)
      setInterfaceStylePart('controls', false)

      expect(document.documentElement.hasAttribute('data-hermes-interface-style')).toBe(true)
      expect(document.documentElement.hasAttribute('data-hermes-interface-typography')).toBe(true)
      expect(document.documentElement.hasAttribute('data-hermes-interface-controls')).toBe(false)
      expect(document.documentElement.hasAttribute('data-hermes-interface-surfaces')).toBe(true)

      setInterfaceStyleEnabled(false)
      expect(ATTRIBUTES.every(attribute => !document.documentElement.hasAttribute(attribute))).toBe(true)
    } finally {
      stop()
    }
  })
})
