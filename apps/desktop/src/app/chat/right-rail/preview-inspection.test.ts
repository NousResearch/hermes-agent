import { afterEach, beforeEach, describe, expect, it, type Mock, vi } from 'vitest'

import { actInPage } from '@/lib/preview-act/act-in-page'
import type { PreviewActHolder } from '@/lib/preview-act/types'
import { $rightRailActiveTabId } from '@/store/layout'
import { closeRightRail, openPreview } from '@/store/preview'

import { actOnActivePreview } from './preview-act'
import * as previewInput from './preview-input'
import { registerPreviewInput } from './preview-input'
import { registerPreviewScriptRunner } from './preview-script-runner'

const guest = window as typeof window & { __hermesActHolder?: PreviewActHolder }
const rect = { left: 594, top: 402, right: 614, bottom: 422, width: 20, height: 20, x: 594, y: 402, toJSON: () => ({}) }

describe('targeted elements inspection', () => {
  let cleanups: Array<() => void>
  let send: Mock<() => void>
  let focus: Mock<() => void>
  let runner: Mock<(code: string) => Promise<unknown>>
  let hit: Mock<() => Element | null>

  beforeEach(() => {
    closeRightRail()
    // Fixed local fixtures only; no business page or untrusted markup is loaded.
    document.body.innerHTML =
      '<section id="canvas"><svg class="handles"><circle/><circle/><circle data-testid="resize"/></svg></section><div id="toolbar">PRIVATE TEXT</div><input id="password" aria-label="PRIVATE LABEL" type="password" value="PRIVATE SECRET">'
    delete guest.__hermesActHolder
    vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue(rect)
    hit = vi.fn(() => document.querySelector('circle:last-child'))
    Object.defineProperty(document, 'elementFromPoint', { configurable: true, value: hit })
    openPreview({ kind: 'url', label: 'Browser', source: 'https://example.com', url: 'https://example.com' })
    const tab = $rightRailActiveTabId.get()!
    send = vi.fn()
    focus = vi.fn()
    // Execute the production serialized payload, as the guest would, to catch
    // accidental module-scope dependencies. All DOM/action inputs are fixtures.
    runner = vi.fn(async (code: string) => await new Function('return ' + code)())
    cleanups = [registerPreviewScriptRunner(tab, runner), registerPreviewInput(tab, { focus, send })]
  })

  afterEach(() => {
    cleanups.forEach(cleanup => cleanup())
    delete guest.__hermesActHolder
    delete (document as unknown as { elementFromPoint?: unknown }).elementFromPoint
    vi.restoreAllMocks()
    document.body.replaceChildren()
    closeRightRail()
  })

  it('inspects the SVG center and ancestry with no focus, input, scroll, watcher or holder write', async () => {
    const active = document.getElementById('password')!
    active.focus()
    const focused = vi.spyOn(HTMLElement.prototype, 'focus')
    const scroll = vi.fn()
    Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: scroll })
    const dispatch = vi.spyOn(EventTarget.prototype, 'dispatchEvent')
    const acquire = vi.spyOn(previewInput, 'activePreviewInput')
    const watcher = vi.fn()
    Object.defineProperty(window, '__hermesWatch_fn', { configurable: true, set: watcher })
    const mutations: MutationRecord[] = []
    const observer = new MutationObserver(records => mutations.push(...records))
    observer.observe(document.body, { attributes: true, childList: true, subtree: true, characterData: true })
    const html = document.body.innerHTML
    const result = await actOnActivePreview({ kind: 'elements', selector: 'svg circle:nth-of-type(3)' })

    expect(result.success).toBe(true)
    expect(result.inspection).toMatchObject({
      coordinateSpace: 'guest-viewport-css-pixels',
      candidateCount: 1,
      truncated: false,
      candidates: [
        {
          node: { tag: 'circle', testId: 'resize', nthOfType: 3 },
          rect: { left: 594, top: 402, width: 20, height: 20 },
          point: { x: 604, y: 412 },
          centerInViewport: true,
          hit: { node: { tag: 'circle' }, relationship: 'self' }
        }
      ]
    })
    expect(result.inspection?.candidates[0].ancestors.slice(0, 2)).toMatchObject([
      { tag: 'svg', class: 'handles' },
      { tag: 'section', id: 'canvas' }
    ])
    expect(hit).toHaveBeenCalledWith(604, 412)
    expect(runner).toHaveBeenCalledOnce()
    expect(document.body.innerHTML).toBe(html)
    expect(document.activeElement).toBe(active)
    expect(guest.__hermesActHolder).toBeUndefined()
    mutations.push(...observer.takeRecords())
    observer.disconnect()
    expect(mutations).toEqual([])
    delete (window as unknown as { __hermesWatch_fn?: unknown }).__hermesWatch_fn

    for (const spy of [send, focus, focused, scroll, dispatch, acquire, watcher]) {
      expect(spy).not.toHaveBeenCalled()
    }

    delete (Element.prototype as unknown as { scrollIntoView?: unknown }).scrollIntoView
  })

  it('distinguishes occlusion and reports pointer-events, visibility and display', async () => {
    const circle = document.querySelector('circle:last-child')! as SVGElement
    circle.style.pointerEvents = 'none'
    circle.style.visibility = 'hidden'
    circle.style.display = 'none'
    hit.mockReturnValue(document.getElementById('toolbar'))
    const result = await actOnActivePreview({ kind: 'elements', selector: '[data-testid="resize"]' })
    expect(result.inspection?.candidates[0]).toMatchObject({
      style: { pointerEvents: 'none', visibility: 'hidden', display: 'none' },
      hit: { node: { tag: 'div', id: 'toolbar' }, relationship: 'unrelated' }
    })
    expect(JSON.stringify(result)).not.toMatch(/PRIVATE|value|label|title|url/)
  })

  it.each(['descendant', 'ancestor', 'none'] as const)('reports %s hit relationships', async relationship => {
    const target = document.getElementById('canvas')!
    hit.mockReturnValue(
      relationship === 'descendant' ? target.firstElementChild : relationship === 'ancestor' ? document.body : null
    )
    const result = await actOnActivePreview({ kind: 'elements', selector: '#canvas' })
    expect(result.inspection?.candidates[0].hit.relationship).toBe(relationship)
  })

  it('counts all matches, caps entries, honors max and reports zero matches', async () => {
    document.body.innerHTML = '<svg>' + '<circle/>'.repeat(20) + '</svg>'
    const all = await actOnActivePreview({ kind: 'elements', selector: 'circle', max: 999 })
    expect(all.inspection).toMatchObject({ candidateCount: 20, truncated: true })
    expect(all.inspection?.candidates).toHaveLength(5)
    expect(
      (await actOnActivePreview({ kind: 'elements', selector: 'circle', max: 2 })).inspection?.candidates
    ).toHaveLength(2)
    expect(await actOnActivePreview({ kind: 'elements', selector: '.missing' })).toMatchObject({
      success: true,
      inspection: { candidateCount: 0, candidates: [], truncated: false }
    })
  })

  it.each(['[', ''])('returns a bounded clear error for invalid/empty selector %j', async selector => {
    const result = await actOnActivePreview({ kind: 'elements', selector })
    expect(result.success).toBe(false)
    expect(result.error).toMatch(/selector/i)
    expect(guest.__hermesActHolder).toBeUndefined()
    expect(hit).not.toHaveBeenCalled()
  })

  it('bounds identity strings, ancestry and serialized output without reading content or values', async () => {
    document.body.innerHTML =
      '<main>'.repeat(10) +
      (
        '<input type="password" value="SECRET" title="SECRET" aria-label="SECRET" id="' +
        '\\'.repeat(1000) +
        '" class="' +
        'x'.repeat(1000) +
        '" data-testid="' +
        'y'.repeat(1000) +
        '">'
      ).repeat(30) +
      '</main>'.repeat(10)

    // Maximize JSON escaping in both the target and hit ancestry; exercise
    // the serialized-size guard, not just the five-entry cap.
    for (const el of document.body.querySelectorAll('*')) {
      for (const attr of ['id', 'class', 'data-testid']) {
        el.setAttribute(attr, '\\'.repeat(1000))
      }
    }

    hit.mockReturnValue(document.querySelector('input'))
    const result = await actOnActivePreview({ kind: 'elements', selector: 'input' })
    expect(result.inspection?.candidateCount).toBe(30)
    expect(result.inspection!.candidates.length).toBeGreaterThan(0)
    expect(result.inspection!.candidates.length).toBeLessThan(5)

    for (const candidate of result.inspection!.candidates) {
      expect(candidate.ancestors.length).toBeLessThanOrEqual(4)

      for (const key of ['id', 'class', 'testId'] as const) {
        expect(candidate.node[key]!.length).toBeLessThanOrEqual(80)
      }
    }

    expect(JSON.stringify(result).length).toBeLessThanOrEqual(12000)
    expect(JSON.stringify(result)).not.toContain('SECRET')
  })

  it('does not hit-test offscreen or non-finite geometry', async () => {
    vi.mocked(Element.prototype.getBoundingClientRect).mockReturnValue({ ...rect, left: -100, right: -80 })
    expect(
      (await actOnActivePreview({ kind: 'elements', selector: 'circle' })).inspection?.candidates[0]
    ).toMatchObject({ centerInViewport: false, hit: { relationship: 'none' } })
    vi.mocked(Element.prototype.getBoundingClientRect).mockReturnValue({ ...rect, width: Infinity })
    expect(
      (await actOnActivePreview({ kind: 'elements', selector: 'circle' })).inspection?.candidates[0]
    ).toMatchObject({ rect: null, point: null })
    expect(hit).not.toHaveBeenCalled()
  })

  it('reads existing refs without rescanning, rebinding, or altering the inventory baseline', async () => {
    hit.mockReturnValue(document.body)
    const holder: PreviewActHolder = {}
    const inventory = actInPage(document, holder, { kind: 'elements' })
    const ref = inventory.elements!.find(entry => entry.selector === '#password')!.ref
    guest.__hermesActHolder = holder
    const before = { ...holder }
    const book = holder.book!.map(binding => ({ ...binding }))
    Object.freeze(holder)
    const result = await actOnActivePreview({ kind: 'elements', ref, selector: '[' })
    expect(result.inspection).toMatchObject({ candidateCount: 1, candidates: [{ node: { id: 'password' } }] })
    expect(holder).toEqual(before)
    expect(holder.book).toEqual(book)
    expect(result.elements).toBeUndefined()
    expect(result.delta).toBeUndefined()
    expect(JSON.stringify(result)).not.toContain('PRIVATE SECRET')
  })

  it('fails closed for unknown, removed and navigated refs without echoing ref text', async () => {
    hit.mockReturnValue(document.body)
    const holder: PreviewActHolder = {}
    const inventory = actInPage(document, holder, { kind: 'elements' })
    const ref = inventory.elements![0].ref
    guest.__hermesActHolder = holder
    expect((await actOnActivePreview({ kind: 'elements', ref: 'PRIVATE REF' })).error).toMatch(/Unknown/)
    holder.book![0].el.remove()
    expect((await actOnActivePreview({ kind: 'elements', ref })).error).toMatch(/removed/)
    holder.url = 'https://elsewhere.example'
    expect((await actOnActivePreview({ kind: 'elements', ref })).error).toMatch(/navigated/)
  })

  it('leaves plain elements on the existing inventory and watcher path', async () => {
    hit.mockReturnValue(document.body)
    const result = await actOnActivePreview({ kind: 'elements', full: true, max: 1 })
    expect(result.success).toBe(true)
    expect(result.elements).toHaveLength(1)
    expect(result.inspection).toBeUndefined()
    expect(guest.__hermesActHolder?.book).toHaveLength(1)
    expect(document.querySelector('[data-hermes-watch]') || guest.__hermesActHolder?.nodes).toBeTruthy()
    expect(send).not.toHaveBeenCalled()
  })

  it('works without a native input channel and never echoes rejected page errors', async () => {
    cleanups[1]()
    expect((await actOnActivePreview({ kind: 'elements', selector: 'circle' })).inspection?.candidateCount).toBe(3)
    runner.mockRejectedValueOnce(new Error('PRIVATE PAGE ERROR'))
    expect(await actOnActivePreview({ kind: 'elements', selector: 'circle' })).toEqual({
      success: false,
      error: 'Target inspection did not answer; no input was sent.'
    })
    expect(send).not.toHaveBeenCalled()
  })

  it('keeps empty refs fail-closed, even with a valid selector', async () => {
    for (const ref of ['', '   ', 'unknown']) {
      const result = await actOnActivePreview({ kind: 'elements', ref, selector: 'circle' })
      expect(result.success).toBe(false)
      expect(result.inspection).toBeUndefined()
    }

    expect(hit).not.toHaveBeenCalled()
  })

  it('rounds centers like locate and excludes the right and bottom viewport edges', async () => {
    const circle = document.querySelector('circle')!

    for (const [left, top, inside] of [
      [0.25, 1.5, true],
      [window.innerWidth - 10, 0, false],
      [0, window.innerHeight - 10, false]
    ] as const) {
      vi.mocked(Element.prototype.getBoundingClientRect).mockReturnValue({ ...rect, left, top })
      const inspection = await actOnActivePreview({ kind: 'elements', selector: 'circle', max: 1 })
      const location = actInPage(document, {}, { kind: 'locate', selector: 'circle' })
      expect(inspection.inspection?.candidates[0].point).toEqual(location.point)
      expect(inspection.inspection?.candidates[0].centerInViewport).toBe(inside)
    }

    expect(circle.isConnected).toBe(true)
  })

  it.each([0, -1, Infinity, NaN])('marks unusable dimensions %s as null', async width => {
    vi.mocked(Element.prototype.getBoundingClientRect).mockReturnValue({ ...rect, width })
    const result = await actOnActivePreview({ kind: 'elements', selector: 'circle', max: 1 })
    expect(result.inspection?.candidates[0]).toMatchObject({ rect: null, point: null, centerInViewport: false })
    expect(hit).not.toHaveBeenCalled()
  })

  it('bounds non-ASCII identity hints and escaped JSON without losing the total', async () => {
    document.body.innerHTML = '<div>'.repeat(10) + '<button></button>'.repeat(8) + '</div>'.repeat(10)

    for (const el of document.body.querySelectorAll('*')) {
      for (const attr of ['id', 'class', 'data-testid']) {el.setAttribute(attr, '雪😀\u0000"\\'.repeat(100))}
    }

    hit.mockReturnValue(document.querySelector('button'))
    const result = await actOnActivePreview({ kind: 'elements', selector: 'button' })
    const json = JSON.stringify(result)
    expect(json.length).toBeLessThanOrEqual(12_000)
    expect(JSON.parse(json)).toEqual(result)
    expect(result.inspection).toMatchObject({ candidateCount: 8, truncated: true })

    for (const candidate of result.inspection!.candidates) {
      expect(candidate.hit.ancestors.length).toBeLessThanOrEqual(4)

      for (const node of [candidate.node, ...candidate.ancestors, candidate.hit.node!, ...candidate.hit.ancestors]) {
        for (const key of ['tag', 'id', 'class', 'testId'] as const) {expect(node[key]!.length).toBeLessThanOrEqual(80)}
      }
    }
  })

  it('does not execute an already-cancelled inspection', async () => {
    const controller = new AbortController()
    controller.abort()
    expect(await actOnActivePreview({ kind: 'elements', selector: 'circle' }, controller.signal)).toMatchObject({
      success: false
    })
    expect(runner).not.toHaveBeenCalled()
    expect(send).not.toHaveBeenCalled()
  })

  it('sanitizes malformed IPC and thrown DOM errors without logging page data', async () => {
    const logs = ['log', 'warn', 'error', 'debug'].map(method =>
      vi.spyOn(console, method as 'log').mockImplementation(() => {})
    )

    runner.mockResolvedValueOnce('SYNTHETIC_PRIVATE_PAGE_DATA {')
    const malformed = await actOnActivePreview({ kind: 'elements', selector: 'circle' })
    expect(malformed).toEqual({ success: false, error: 'Target inspection did not answer; no input was sent.' })
    vi.mocked(Element.prototype.getBoundingClientRect).mockImplementation(() => {
      throw new Error('SYNTHETIC_PRIVATE_EXCEPTION')
    })
    expect(await actOnActivePreview({ kind: 'elements', selector: 'circle' })).toEqual(malformed)

    for (const log of logs) {expect(log).not.toHaveBeenCalled()}
  })

  it('fails rather than claiming a navigation success when inspection is silent', async () => {
    runner.mockResolvedValueOnce(undefined)
    expect(await actOnActivePreview({ kind: 'elements', selector: 'circle' })).toEqual({
      success: false,
      error: 'Target inspection did not answer; no input was sent.'
    })
    expect(send).not.toHaveBeenCalled()
  })
})
