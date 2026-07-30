import { describe, expect, it } from 'vitest'

import {
  DIRECTIVE_CHIP_CLASS,
  directiveChipClass,
  SLASH_CHIP_BASE_CLASS
} from '@/components/assistant-ui/directive-text'
import { REFERENCE_STYLES, referenceStyle } from '@/components/assistant-ui/reference-kinds'

/**
 * A chip is inline text, not a badge: same size as the words around it, no
 * background or padding to step over, and a per-kind color + icon carrying the
 * "what kind of thing is this" signal.
 */
describe('chip typography', () => {
  for (const [name, cls] of [
    ['directive chip', DIRECTIVE_CHIP_CLASS],
    ['slash chip', SLASH_CHIP_BASE_CLASS]
  ] as const) {
    it(`${name} inherits the surrounding font size`, () => {
      expect(cls).not.toMatch(/\btext-\[0\.\d+em\]/)
    })

    it(`${name} renders as text, with no badge chrome`, () => {
      for (const chrome of ['bg-', 'rounded', 'px-', 'py-', 'border']) {
        expect(cls).not.toContain(chrome)
      }
    })

    it(`${name} sits on the text baseline without a nudge`, () => {
      // With no vertical padding there's nothing to cancel, so the pill needs
      // no magic em offset to stop riding low.
      expect(cls).toContain('align-baseline')
      expect(cls).not.toMatch(/align-\[-/)
    })
  }

  it('directive and slash chips are literally the same shape', () => {
    expect(SLASH_CHIP_BASE_CLASS).toBe(DIRECTIVE_CHIP_CLASS)
  })

  it('resolves to the same font size as its container', () => {
    const host = document.createElement('div')

    host.style.fontSize = '16px'
    host.innerHTML = `<span id="chip" class="${DIRECTIVE_CHIP_CLASS}">apps/desktop/</span>`
    document.body.append(host)

    const chip = host.querySelector('#chip') as HTMLElement

    expect(getComputedStyle(chip).fontSize).toBe(getComputedStyle(host).fontSize)

    host.remove()
  })
})

describe('the shared reference vocabulary', () => {
  it('gives every kind an icon, a color, and a label', () => {
    for (const [kind, style] of Object.entries(REFERENCE_STYLES)) {
      expect(style.codicon, `${kind} codicon`).toBeTruthy()
      expect(style.color, `${kind} color`).toBeTruthy()
      expect(style.label, `${kind} label`).toBeTruthy()

      // Emoji rows render the emoji itself instead of a glyph.
      if (kind !== 'emoji') {
        expect(style.paths.length, `${kind} paths`).toBeGreaterThan(0)
      }
    }
  })

  it('carries the kind colour into the chip class', () => {
    for (const kind of ['file', 'url', 'skill', 'command'] as const) {
      expect(directiveChipClass(kind)).toContain(referenceStyle(kind).color)
    }
  })

  it('falls back to a real style for an unknown kind', () => {
    const style = referenceStyle('something-new')

    expect(style).toBe(REFERENCE_STYLES.other)
    expect(style.codicon).toBeTruthy()
  })

  it('gives commands and skills distinct accents so a list reads at a glance', () => {
    expect(referenceStyle('skill').color).not.toBe(referenceStyle('command').color)
  })
})
