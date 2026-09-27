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

beforeEach(() => {
  vi.stubGlobal('matchMedia', () => ({ matches: false }))
})

afterEach(() => {
  document.body.innerHTML = ''
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('bounded history geometry', () => {
  it.each([800, -800])('keeps an exact visible occurrence after a %+i pixel append/eviction shift', shift => {
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
    vi.spyOn(children[0], 'getBoundingClientRect').mockImplementation(() =>
      rect(200 + displacement - viewport.scrollTop, 100)
    )
    vi.spyOn(children[1], 'getBoundingClientRect').mockImplementation(() =>
      rect(1000 + displacement - viewport.scrollTop, 200)
    )
    const anchors = captureHistoryScroll(viewport)
    expect(anchors[0]).toEqual({ key: 'text-99', occurrence: 1, offset: 100 })
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

  it.each([false, true])('retargets the selected prompt as deferred layout changes (cancel=%s)', async cancel => {
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
    document.body.append(viewport)
    Object.defineProperty(viewport, 'scrollHeight', { value: 6000 })
    Object.defineProperty(viewport, 'clientHeight', { value: 600 })
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))
    const group = viewport.firstElementChild as HTMLElement
    group.style.contentVisibility = 'auto'
    let layoutTop = 1800
    vi.spyOn(group, 'getBoundingClientRect').mockImplementation(() => rect(layoutTop - viewport.scrollTop, 900))
    const controller = new AbortController()
    const pending = scrollTimelineTarget(viewport, 'exact-row', controller.signal)
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
  })
})
