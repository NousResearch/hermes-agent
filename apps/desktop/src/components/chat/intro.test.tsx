import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { Intro } from './intro'

const HERO_SLOT = '[data-slot="ex-machina-cover-art"]'
const WORDMARK_SLOT = '[data-slot="aui_intro-wordmark"]'

const renderIntro = () => {
  const { container } = render(<Intro seed={0} />)
  const root = container.querySelector('[data-slot="aui_intro"]')

  if (!root) {
    throw new Error('intro root missing')
  }

  const hero = root.querySelector(HERO_SLOT)

  return {
    hero,
    heroImg: hero?.matches('img') ? hero : (hero?.querySelector('img') ?? null),
    root,
    wordmark: root.querySelector(WORDMARK_SLOT)
  }
}

describe('Intro Ex Machina cover art', () => {
  it('renders the supplied cover art directly', () => {
    const { heroImg } = renderIntro()

    // .webp, not the 1.9MB .jpg it replaced — the source format is the whole
    // reason this asset is shippable.
    expect(heroImg?.getAttribute('src')).toMatch(/ex-machina-cover-art\.webp/)
  })

  it('is decorative: hidden from a11y and carrying no accessible name', () => {
    const { heroImg } = renderIntro()

    expect(heroImg?.getAttribute('aria-hidden')).toBe('true')
    expect(heroImg?.getAttribute('alt')).toBe('')
    // An exposed `img` role here would make every empty state announce a
    // decoration before the actual prompt. Nothing in the intro may claim one.
    expect(screen.queryAllByRole('img')).toHaveLength(0)
  })

  it('decodes off the main thread and takes no priority hint', () => {
    const { heroImg } = renderIntro()

    expect(heroImg?.getAttribute('decoding')).toBe('async')
    // `fetchpriority=low` deferred the largest thing on the empty state.
    expect(heroImg?.getAttribute('fetchpriority')).toBeNull()
  })

  it('renders as a direct hero above the intro copy', () => {
    const { hero, root } = renderIntro()
    const copy = screen.getByText(/\S/, { selector: '[data-slot="aui_intro"] p:last-child' })

    expect(hero).not.toBeNull()
    expect(root.contains(hero)).toBe(true)
    expect(hero?.contains(copy)).toBe(false)
    expect(hero?.parentElement).toBe(copy.parentElement)
  })

  it('is aspect-preserving and unfiltered, sized by the stylesheet', () => {
    const { heroImg } = renderIntro()

    // Sizing and the per-skin reveal both live in styles.css
    // (.ex-machina-cover-art) and no stylesheet is applied under jsdom, so this
    // only pins the hooks those rules match on, and the filters.
    expect(heroImg?.className).toContain('ex-machina-cover-art')
    expect(heroImg?.className).toContain('object-contain')
    expect(heroImg?.className).not.toContain('mix-blend')
    expect(heroImg?.className).not.toContain('opacity-')
    // `max-w-none` would out-specify the stylesheet's `max-inline-size: 100%`
    // and let the art overflow a narrow window sideways.
    expect(heroImg?.className).not.toContain('max-w-none')
  })

  it('keeps the intro copy as the readable content', () => {
    renderIntro()

    const copy = screen.getByText(/\S/, { selector: '[data-slot="aui_intro"] p:last-child' })

    expect(copy.textContent?.trim().length).toBeGreaterThan(0)
  })
})

describe('Intro wordmark fallback', () => {
  it('ships alongside the cover art so the skin can swap them', () => {
    const { hero, wordmark } = renderIntro()

    // Both headers are in the markup unconditionally; only
    // `[data-hermes-theme='ex-machina']` reveals the art and visually hides this
    // one while preserving its accessible name, so every skin keeps a header.
    expect(hero).not.toBeNull()
    expect(wordmark).not.toBeNull()
    expect(wordmark?.textContent).toContain('HERMES AGENT')
  })

  it('carries the wordmark as its accessible name, once', () => {
    const { wordmark } = renderIntro()

    expect(wordmark?.getAttribute('aria-label')).toBe('HERMES AGENT')
    // The fit-text duplicate is the sizing sentinel, not a second announcement.
    expect(screen.getAllByLabelText('HERMES AGENT')).toHaveLength(1)
  })

  it('keeps the hooks the stylesheet sizes and gates it with', () => {
    const { wordmark } = renderIntro()

    expect(wordmark?.className).toContain('fit-text')
    expect(wordmark?.className).toContain('intro-wordmark')
    expect(wordmark?.getAttribute('style')).toContain('--fit-min')
  })
})
