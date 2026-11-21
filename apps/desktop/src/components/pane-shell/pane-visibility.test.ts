import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { PANE_HIDDEN_ATTR, queryAllVisible, queryVisible } from './pane-visibility'

// The live stylesheet, read directly: the painting-contract test below must
// validate the shipped rule, not a copy of it. Tests run with cwd =
// apps/desktop (vitest projects root).
const cssText = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8')

/**
 * Inactive tabs stay mounted with their layout box intact, so they answer
 * document-wide lookups exactly like the visible tab. These helpers are the one
 * place that difference is decided.
 */

const COMPOSER = '[data-slot="composer-root"]'

const tab = (id: string, hidden = false) => `
  <div ${hidden ? PANE_HIDDEN_ATTR : ''}>
    <section><div data-slot="composer-root" id="${id}"></div></section>
  </div>
`

afterEach(() => {
  document.body.innerHTML = ''
})

describe('pane visibility lookups', () => {
  it('resolves the foreground element even when a hidden tab matches first', () => {
    document.body.innerHTML = tab('background', true) + tab('foreground')

    expect(queryVisible(COMPOSER)?.id).toBe('foreground')
    expect(queryAllVisible(COMPOSER).map(el => el.id)).toEqual(['foreground'])
  })
})

describe('hidden pane painting contract', () => {
  // The pane-level rule from styles.css that keeps a keep-alive hidden pane
  // opaque to painting. Third-party widget scripts (Twitter's embed) render
  // their card with an inline `visibility: visible`, which CSS would otherwise
  // let paint through the pane's `visibility: hidden` over the active view
  // (#79833). The rule forces every descendant to follow the pane's
  // visibility. Extract it from the real stylesheet so this test validates the
  // shipped rule — if the rule disappears, the contract is broken.
  const paneBleedRule = (): string => {
    // Anchored on the rule's own comment + selector, so reordering adjacent
    // rules or reformatting unrelated CSS can't break the extraction.
    const match = cssText.match(/\/\* Keep-alive hidden panes[\s\S]*?\[data-pane-hidden\] \* \{[\s\S]*?\}/)

    expect(match).not.toBeNull()

    return match![0]
  }

  afterEach(() => {
    document.body.innerHTML = ''
    document.head.querySelector('style[data-pane-bleed-rule]')?.remove()
  })

  it('keeps a descendant with inline visibility:visible from painting through a hidden pane', () => {
    const style = document.createElement('style')

    style.dataset.paneBleedRule = 'true'
    style.textContent = paneBleedRule()
    document.head.appendChild(style)

    // What the reporter saw: the widget's rendered card (inline
    // visibility:visible) nested inside a keep-alive pane (visibility:hidden).
    document.body.innerHTML = `
      <div data-pane-hidden style="visibility: hidden">
        <div style="visibility: visible" id="leaky-card">card</div>
      </div>
    `

    expect(getComputedStyle(document.getElementById('leaky-card')!).visibility).toBe('hidden')
  })

  it('does not affect a visible pane or an inline override outside one', () => {
    document.body.innerHTML = `<div style="visibility: visible" id="live-card">card</div>`

    expect(getComputedStyle(document.getElementById('live-card')!).visibility).toBe('visible')
  })
})

describe('session switch transition suppression contract', () => {
  // Toggling `visibility: hidden` <-> `visible` across `[data-pane-hidden] *`
  // otherwise starts CSS transitions on transition-enabled descendants, firing
  // a `transitionrun` storm that compounds with synchronous scroll-restore
  // geometry reads into a 1-2s `Recalculate style` stall (#158689, #146852).
  const paneTransitionRule = (): string => {
    const match = cssText.match(
      /\/\* Suppress CSS transitions on hidden keep-alive panes[\s\S]*?\[data-session-switching\] \*::after \{[\s\S]*?\}/
    )

    expect(match).not.toBeNull()

    return match![0]
  }

  it('disables CSS transitions across hidden panes and switching surfaces (#158689)', () => {
    const rule = paneTransitionRule()

    expect(rule).toContain('[data-pane-hidden]')
    expect(rule).toContain('[data-pane-hidden] *')
    expect(rule).toContain('[data-session-switching]')
    expect(rule).toContain('[data-session-switching] *')
    expect(rule).toMatch(/transition:\s*none\s*!important/)
  })
})
