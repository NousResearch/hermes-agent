import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { captureHistoryScroll, restoreHistoryScroll } from './history-scroll'
import { scrollTimelineTarget } from './timeline-scroll'

const rect = (top: number, height: number) => ({
  top,
  bottom: top + height,
  height,
  left: 0,
  right: 800,
  width: 800,
  x: 0,
  y: top,
  toJSON: () => ({})
})

function stubCss(mode: string) {
  if (mode === 'absent') {
    vi.stubGlobal('CSS', undefined)
  } else if (mode === 'no-escape') {
    vi.stubGlobal('CSS', {})
  }
}

beforeEach(() => {
  vi.stubGlobal('matchMedia', () => ({ matches: false }))
})

afterEach(() => {
  document.body.innerHTML = ''
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('bounded history geometry', () => {
  it.each([
    { shift: 800, css: 'provided' },
    { shift: -800, css: 'provided' },
    { shift: 800, css: 'absent' },
    { shift: 800, css: 'no-escape' }
  ])('keeps an exact visible occurrence after a $shift pixel shift (CSS: $css)', ({ shift, css }) => {
    stubCss(css)
    const viewport = document.createElement('div')
    viewport.innerHTML =
      '<div data-slot="aui_message-group" data-history-anchor="old-group"><div data-history-anchor="text-99"></div><div data-history-anchor="text-99"></div></div>'
    document.body.append(viewport)
    viewport.scrollTop = 900
    vi.spyOn(viewport, 'getBoundingClientRect').mockImplementation(() => rect(0, 600))
    const group = viewport.firstElementChild as HTMLElement
    let displacement = 0
    vi.spyOn(group, 'getBoundingClientRect').mockImplementation(() =>
      rect(100 + displacement - viewport.scrollTop, 2000)
    )
    const children = Array.from(group.children) as HTMLElement[]
    // The shared bootstrap provides an identity escape stub, not a browser's
    // implementation. Quoted keys exercise our fallback when that API is absent.
    const key = css === 'provided' ? 'text-99' : 'text-[99]"quoted"'
    children.forEach(child => {
      child.dataset.historyAnchor = key
    })
    vi.spyOn(children[0], 'getBoundingClientRect').mockImplementation(() =>
      rect(200 + displacement - viewport.scrollTop, 100)
    )
    vi.spyOn(children[1], 'getBoundingClientRect').mockImplementation(() =>
      rect(1000 + displacement - viewport.scrollTop, 200)
    )
    const anchors = captureHistoryScroll(viewport)
    expect(anchors[0]).toEqual({ key, occurrence: 1, offset: 100 })
    group.dataset.historyAnchor = 'new-group' // the old folded assistant head was evicted
    displacement = shift
    expect(restoreHistoryScroll(viewport, anchors)).toBe(true)
    expect(children[1].getBoundingClientRect().top).toBe(100)
    expect(viewport.scrollTop).toBe(900 + shift)
  })

  it('does not measure descendants of offscreen content-visibility groups', () => {
    const viewport = document.createElement('div')
    viewport.innerHTML = '<div data-slot="aui_message-group"><div data-history-anchor="hidden"></div></div>'
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))
    vi.spyOn(viewport.firstElementChild!, 'getBoundingClientRect').mockReturnValue(rect(-2000, 500))
    const measure = vi.spyOn(viewport.firstElementChild!.firstElementChild!, 'getBoundingClientRect')
    expect(captureHistoryScroll(viewport)).toEqual([])
    expect(measure).not.toHaveBeenCalled()
  })

  it.each([
    { cancel: false, css: 'provided' },
    { cancel: true, css: 'provided' },
    { cancel: false, css: 'absent' },
    { cancel: false, css: 'no-escape' }
  ])('retargets the selected prompt through layout changes (cancel: $cancel, CSS: $css)', async ({ cancel, css }) => {
    stubCss(css)
    const frames = new Map<number, FrameRequestCallback>()
    let serial = 0
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frames.set(++serial, callback)

      return serial
    })
    vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
    vi.spyOn(performance, 'now').mockReturnValue(0)

    const tick = (time: number) => {
      const callbacks = [...frames.values()]
      frames.clear()

      for (const callback of callbacks) {
        callback(time)
      }
    }

    const viewport = document.createElement('div')
    viewport.innerHTML =
      '<div data-slot="aui_message-group"><div data-slot="aui_turn-pair"><div data-message-id="exact-row"></div></div></div>'

    const id = css === 'provided' ? 'exact-row' : 'exact-["row"]'
    viewport.querySelector('[data-message-id]')!.setAttribute('data-message-id', id)
    document.body.append(viewport)
    Object.defineProperty(viewport, 'scrollHeight', { value: 6000 })
    Object.defineProperty(viewport, 'clientHeight', { value: 600 })
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))
    const group = viewport.firstElementChild as HTMLElement
    group.style.contentVisibility = 'auto'
    let layoutTop = 1800
    vi.spyOn(group, 'getBoundingClientRect').mockImplementation(() => rect(layoutTop - viewport.scrollTop, 900))
    const controller = new AbortController()
    const pending = scrollTimelineTarget(viewport, id, controller.signal)

    try {
      tick(100)
      layoutTop = 2600 // earlier skipped turns acquired their real height

      if (cancel) {
        controller.abort()
      }

      const stoppedAt = viewport.scrollTop
      tick(200)
      tick(220)
      tick(240)
      expect(await pending).toBe(!cancel)
      expect(viewport.scrollTop).toBe(cancel ? stoppedAt : 2592)
      expect(group.style.contentVisibility).toBe('auto')
      expect(frames.size).toBe(0)
    } finally {
      controller.abort()
    }
  })
})
