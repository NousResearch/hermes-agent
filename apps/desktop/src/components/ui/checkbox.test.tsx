import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeAll, describe, expect, it } from 'vitest'

import { Checkbox } from './checkbox'

/**
 * The indicator stacks both glyphs and hides one with a utility class, so the
 * cascade — not the markup — decides what the user sees. codicon.css sets
 * `display: inline-block` on `.codicon[class*='codicon-']`, which outranks a
 * single-class utility and paints the check and the dash at the same time.
 * That specificity race is the real contract, so the test carries both
 * stylesheets rather than asserting on class strings. Tailwind's nested output
 * is flattened to the equivalent descendant selector because jsdom does not
 * implement CSS nesting.
 */
const CODICON_CSS = `.codicon[class*='codicon-'] { display: inline-block; }`

const TAILWIND_CSS = `
  .hidden\\! { display: none !important; }
  :where(.group)[data-state="checked"] .group-data-\\[state\\=checked\\]\\:block\\! {
    display: block !important;
  }
  :where(.group)[data-state="indeterminate"] .group-data-\\[state\\=indeterminate\\]\\:block\\! {
    display: block !important;
  }
`

function shownGlyphs(container: HTMLElement) {
  return [...container.querySelectorAll<HTMLElement>('.codicon')]
    .filter(glyph => getComputedStyle(glyph).display !== 'none')
    .map(glyph => (glyph.classList.contains('codicon-check') ? 'check' : 'dash'))
}

beforeAll(() => {

  const style = document.createElement('style')
  style.textContent = `${CODICON_CSS}\n${TAILWIND_CSS}`

  document.head.append(style)
})

afterEach(cleanup)

describe('Checkbox', () => {
  it('paints the check alone when checked', () => {
    const { container } = render(<Checkbox checked />)

    expect(shownGlyphs(container)).toEqual(['check'])
  })

  it('paints the dash alone when indeterminate', () => {
    const { container } = render(<Checkbox checked="indeterminate" />)

    expect(shownGlyphs(container)).toEqual(['dash'])
  })

  it('paints no glyph when unchecked', () => {
    const { container } = render(<Checkbox checked={false} />)

    expect(shownGlyphs(container)).toEqual([])
  })
})
