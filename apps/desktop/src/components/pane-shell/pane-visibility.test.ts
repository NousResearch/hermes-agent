import { afterEach, describe, expect, it } from 'vitest'

import { hiddenPaneProps, PANE_HIDDEN_ATTR, queryAllVisible, queryVisible } from './pane-visibility'

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

  it('answers normally when nothing is hidden', () => {
    document.body.innerHTML = tab('only')

    expect(queryVisible(COMPOSER)?.id).toBe('only')
  })

  it('marks a pane hidden only while it is inactive', () => {
    expect(hiddenPaneProps(true)).toEqual({ [PANE_HIDDEN_ATTR]: '' })
    expect(hiddenPaneProps(false)).toEqual({})
  })
})
