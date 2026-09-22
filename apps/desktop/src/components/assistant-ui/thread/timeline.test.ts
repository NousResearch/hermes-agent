import { afterEach, describe, expect, it, vi } from 'vitest'

import { activeTimelineIndex, cacheTimelineGeometry, ownViewport } from './timeline'

/**
 * Several chat surfaces are mounted at once — side by side in a split, and
 * stacked as kept-alive inactive tabs. A timeline scrolls its OWN thread.
 */

afterEach(() => {
  document.body.innerHTML = ''
})

const surface = (id: string, hidden = false) => `
  <div ${hidden ? 'data-pane-hidden' : ''}>
    <div data-session-anchor="${id}">
      <div data-slot="aui_thread-viewport" id="viewport-${id}"></div>
      <div data-slot="thread-timeline" id="timeline-${id}"></div>
    </div>
  </div>
`

describe('ownViewport', () => {
  it('resolves the viewport of the surface the timeline lives in', () => {
    document.body.innerHTML = surface('workspace') + surface('session-tile:b')

    expect(ownViewport(document.getElementById('timeline-session-tile:b'))?.id).toBe('viewport-session-tile:b')
    expect(ownViewport(document.getElementById('timeline-workspace'))?.id).toBe('viewport-workspace')
  })

  it('ignores a kept-alive tab that matches first', () => {
    document.body.innerHTML = surface('workspace', true) + surface('session-tile:b')

    expect(ownViewport(document.getElementById('timeline-session-tile:b'))?.id).toBe('viewport-session-tile:b')
  })

  it('falls back to the document when there is no surface around it', () => {
    document.body.innerHTML = '<div data-slot="aui_thread-viewport" id="viewport-lone"></div>'

    expect(ownViewport(null)?.id).toBe('viewport-lone')
  })
})

describe('timeline geometry cache', () => {
  it('uses cached turn offsets for each scroll frame', () => {
    document.body.innerHTML = `
      <div data-slot="aui_thread-viewport">
        <div data-slot="aui_turn-pair" data-top="20"><div data-message-id="first"></div></div>
        <div data-slot="aui_turn-pair" data-top="120"><div data-message-id="second"></div></div>
      </div>
    `
    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!

    const rect = vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (
      this: HTMLElement
    ) {
      const top = Number(this.getAttribute('data-top') ?? 0)

      return { width: 0, height: 0, top, left: 0, bottom: top, right: 0, x: 0, y: 0, toJSON: () => ({}) }
    })

    const geometry = cacheTimelineGeometry(
      viewport,
      new Map([
        ['first', 0],
        ['second', 1]
      ])
    )

    expect(activeTimelineIndex(geometry, 0)).toBe(0)
    expect(activeTimelineIndex(geometry, 130)).toBe(1)
    expect(rect).toHaveBeenCalledTimes(3)
  })

  it('refreshes cached offsets after a layout update', () => {
    document.body.innerHTML = `
      <div data-slot="aui_thread-viewport">
        <div data-slot="aui_turn-pair" data-top="20"><div data-message-id="first"></div></div>
        <div data-slot="aui_turn-pair" data-top="120"><div data-message-id="second"></div></div>
      </div>
    `
    const viewport = document.querySelector<HTMLElement>('[data-slot="aui_thread-viewport"]')!
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      const top = Number(this.getAttribute('data-top') ?? 0)

      return { width: 0, height: 0, top, left: 0, bottom: top, right: 0, x: 0, y: 0, toJSON: () => ({}) }
    })
    const turn = viewport.querySelectorAll<HTMLElement>('[data-slot="aui_turn-pair"]')[1]

    const indexes = new Map([
      ['first', 0],
      ['second', 1]
    ])

    const beforeLayoutChange = cacheTimelineGeometry(viewport, indexes)

    expect(activeTimelineIndex(beforeLayoutChange, 70)).toBe(0)
    turn.setAttribute('data-top', '50')
    expect(activeTimelineIndex(beforeLayoutChange, 70)).toBe(0)
    expect(activeTimelineIndex(cacheTimelineGeometry(viewport, indexes), 70)).toBe(1)
  })
})
