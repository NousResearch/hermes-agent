import { afterEach, describe, expect, it, vi } from 'vitest'

import { inFlowAncestor, ownViewport, promptJumpTop } from './timeline'

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

describe('inFlowAncestor', () => {
  it('returns the node itself when it is not sticky', () => {
    const node = document.createElement('div')
    document.body.appendChild(node)

    expect(inFlowAncestor(node)).toBe(node)
  })

  it('walks up past a sticky bubble to its in-flow turn container', () => {
    const turn = document.createElement('div')
    turn.style.position = 'static'
    const bubble = document.createElement('div')
    bubble.style.position = 'sticky'
    bubble.style.top = '4px'
    turn.appendChild(bubble)
    document.body.appendChild(turn)

    expect(inFlowAncestor(bubble)).toBe(turn)
  })

  it('walks through nested sticky ancestors until an in-flow element is found', () => {
    const turn = document.createElement('div')
    turn.style.position = 'static'
    const mid = document.createElement('div')
    mid.style.position = 'sticky'
    const bubble = document.createElement('div')
    bubble.style.position = 'sticky'
    mid.appendChild(bubble)
    turn.appendChild(mid)
    document.body.appendChild(turn)

    expect(inFlowAncestor(bubble)).toBe(turn)
  })
})

describe('promptJumpTop', () => {
  const setRect = (element: HTMLElement, top: number) => {
    element.getBoundingClientRect = vi.fn(() => ({ top } as DOMRect))
  }

  it('targets the prompt turn, offset 8px above the viewport top', () => {
    const viewport = document.createElement('div')
    setRect(viewport, 100)
    Object.defineProperty(viewport, 'scrollTop', { configurable: true, value: 400 })
    const turn = document.createElement('div')
    setRect(turn, 50)
    const bubble = document.createElement('div')
    bubble.style.position = 'sticky'
    turn.appendChild(bubble)
    viewport.appendChild(turn)

    expect(promptJumpTop(viewport, bubble)).toBe(400 + 50 - 100 - 8)
  })

  it('measures the in-flow ancestor of a sticky prompt, not its pinned rect', () => {
    const viewport = document.createElement('div')
    setRect(viewport, 100)
    Object.defineProperty(viewport, 'scrollTop', { configurable: true, value: 400 })
    const turn = document.createElement('div')
    turn.style.position = 'static'
    // The stuck bubble pins to the viewport top (~4px) even though its turn
    // sits 1500px below the viewport — the old code computed 400+104-100-8≈396
    // (a dead ~4px nudge), the fix lands on the turn's real position.
    const bubble = document.createElement('div')
    bubble.style.position = 'sticky'
    setRect(bubble, 104)
    setRect(turn, 1600)
    turn.appendChild(bubble)
    viewport.appendChild(turn)

    expect(promptJumpTop(viewport, bubble)).toBe(400 + 1600 - 100 - 8)
  })

  it('clamps the target at zero', () => {
    const viewport = document.createElement('div')
    setRect(viewport, 100)
    Object.defineProperty(viewport, 'scrollTop', { configurable: true, value: 0 })
    const turn = document.createElement('div')
    setRect(turn, 0)
    const bubble = document.createElement('div')
    bubble.style.position = 'sticky'
    turn.appendChild(bubble)
    viewport.appendChild(turn)

    expect(promptJumpTop(viewport, bubble)).toBe(0)
  })
})
