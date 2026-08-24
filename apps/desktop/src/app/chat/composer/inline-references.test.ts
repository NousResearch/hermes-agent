import { describe, expect, it } from 'vitest'

import { refAttrs, refAttrsHtml } from '@/components/assistant-ui/directive-text'
import { REFERENCE_STYLES, referenceKind, referenceStyle } from '@/components/assistant-ui/reference-kinds'

/**
 * There is ONE inline-reference system: `class="ref"` + `data-ref="<kind>"`.
 * A pasted link, an `@file:` chip, a `/skill`, a `@session:` the agent wrote —
 * all the same markup, styled by the `.ref` rules in styles.css.
 */
describe('the inline reference contract', () => {
  it('marks any element as a reference of a given kind', () => {
    expect(refAttrs('file')).toEqual({ className: 'ref', 'data-ref': 'file' })
    expect(refAttrsHtml('skill')).toBe('class="ref" data-ref="skill"')
  })

  it('an unkinded reference is a plain link, not a broken one', () => {
    // A bare external link has no kind — it keeps the default link colour
    // rather than being tagged with a wrong one.
    expect(refAttrs()).toEqual({ className: 'ref' })
    expect(refAttrsHtml()).toBe('class="ref"')
  })

  it('normalises an unknown kind instead of emitting it raw', () => {
    // A kind CSS has no rule for would silently render unstyled; coercing to
    // `other` keeps it inside the system.
    expect(refAttrs('wat')['data-ref']).toBe('other')
    expect(referenceKind('wat')).toBe('other')
  })

  it('ships no colour from TypeScript — the theme owns every accent', () => {
    // The whole point of keying on `data-ref`: a skin restyles all references
    // at once, and no hex or color-mix() is hardcoded in a component.
    for (const [kind, style] of Object.entries(REFERENCE_STYLES)) {
      expect(style, `${kind} must not carry a colour`).not.toHaveProperty('color')
    }

    expect(JSON.stringify(refAttrs('url'))).not.toMatch(/color|#[0-9a-f]{3}/i)
  })

  it('gives every kind a glyph and a label', () => {
    for (const [kind, style] of Object.entries(REFERENCE_STYLES)) {
      expect(style.codicon, `${kind} codicon`).toBeTruthy()
      expect(style.label, `${kind} label`).toBeTruthy()

      // Emoji rows render the emoji itself instead of a glyph.
      if (kind !== 'emoji') {
        expect(style.paths.length, `${kind} paths`).toBeGreaterThan(0)
      }
    }
  })

  it('keeps commands and skills visually distinct', () => {
    // Different data-ref values, so the stylesheet can accent them apart.
    expect(refAttrs('skill')['data-ref']).not.toBe(refAttrs('command')['data-ref'])
    expect(referenceStyle('skill').codicon).not.toBe(referenceStyle('command').codicon)
  })
})

describe('references are text, not badges', () => {
  it('carries no layout, padding, or background of its own', () => {
    // Everything visual lives in the stylesheet. If a component starts adding
    // its own chrome here, that's the drift this system exists to prevent.
    const { className } = refAttrs('file')

    expect(className).toBe('ref')

    for (const chrome of ['bg-', 'rounded', 'px-', 'py-', 'border', 'inline-flex', 'text-[']) {
      expect(className).not.toContain(chrome)
    }
  })
})
