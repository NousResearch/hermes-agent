import { beforeEach, describe, expect, it } from 'vitest'

import {
  BANNER_ID,
  buildPickerProbeSource,
  OVERLAY_ID,
  PICKER_GLOBAL_KEY,
  type PickerResult
} from './element-picker'

/**
 * Runs the probe source inside jsdom by evaluating it like
 * webview.executeJavaScript would. Returns the promise the probe produces.
 */
function runProbe(bannerMessage = 'Annotate mode'): Promise<PickerResult> {
  const source = buildPickerProbeSource(bannerMessage)
  // The probe is an IIFE returning a Promise — evaluate in page context.
  // eslint-disable-next-line no-eval
  return eval(source) as Promise<PickerResult>
}

function fireMouseMove(target: Element) {
  // jsdom lacks real hit-testing; stub elementFromPoint for the duration.
  const event = new MouseEvent('mousemove', { bubbles: true, cancelable: true })
  document.dispatchEvent(event)
  return target
}

beforeEach(() => {
  document.body.innerHTML = ''
  document.documentElement.innerHTML = '<head></head><body></body>'
})

describe('element picker probe', () => {
  it('injects highlight + banner overlays into the page', async () => {
    runProbe()
    expect(document.getElementById(OVERLAY_ID)).not.toBeNull()
    expect(document.getElementById(BANNER_ID)).not.toBeNull()
    expect(document.getElementById(BANNER_ID)!.textContent).toContain('Annotate mode')
  })

  it('resolves cancelled when Escape is pressed', async () => {
    const promise = runProbe()
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))

    const result = await promise
    expect(result.kind).toBe('cancelled')

    // overlays removed
    expect(document.getElementById(OVERLAY_ID)).toBeNull()
    expect(document.getElementById(BANNER_ID)).toBeNull()
  })

  it('resolves with element descriptor when a click lands', async () => {
    document.body.innerHTML = `<div id="app"><button class="submit primary">登录</button></div>`
    const button = document.querySelector('button')!

    // jsdom has no layout; stub getBoundingClientRect + elementFromPoint
    button.getBoundingClientRect = () =>
      ({ x: 10, y: 20, width: 80, height: 32, top: 20, left: 10, right: 90, bottom: 52, toJSON: () => ({}) }) as DOMRect
    const originalEFP = document.elementFromPoint
    document.elementFromPoint = () => button

    const promise = runProbe()
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    const result = await promise
    expect(result.kind).toBe('picked')
    if (result.kind === 'picked') {
      expect(result.element.selector).toBe('#app > button.submit.primary')
      expect(result.element.tagName).toBe('BUTTON')
      expect(result.element.text).toBe('登录')
      expect(result.element.rect).toEqual({ x: 10, y: 20, width: 80, height: 32 })
    }

    document.elementFromPoint = originalEFP
  })

  it('click is prevented from reaching the page', async () => {
    document.body.innerHTML = `<button id="b">x</button>`
    const button = document.querySelector('button')!
    const originalEFP = document.elementFromPoint
    document.elementFromPoint = () => button

    let pageSawClick = false
    button.addEventListener('click', () => {
      pageSawClick = true
    })

    const promise = runProbe()
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
    await promise

    expect(pageSawClick).toBe(false)
    document.elementFromPoint = originalEFP
  })

  it('a second injection tears down the first probe', async () => {
    document.body.innerHTML = `<button id="b1">one</button><button id="b2">two</button>`
    const first = document.querySelector('#b1')!
    const originalEFP = document.elementFromPoint
    document.elementFromPoint = () => first

    const firstPromise = runProbe()
    const secondPromise = runProbe() // should not throw

    // first probe's overlays were replaced (ids are unique)
    expect(document.querySelectorAll(`#${OVERLAY_ID}`).length).toBe(1)

    document.elementFromPoint = originalEFP
    // cancel the live one so we don't leak listeners
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))
    await secondPromise
    // first promise may or may not resolve depending on teardown timing — just ensure no unhandled rejection
    firstPromise.catch(() => undefined)
  })

  it('registers a teardown handle on the window global', () => {
    runProbe()
    const handle = (window as unknown as Record<string, { teardown?: unknown }>)[PICKER_GLOBAL_KEY]
    expect(typeof handle?.teardown).toBe('function')
  })

  it('resolves with region descriptor when user drags a marquee', async () => {
    document.body.innerHTML = `<div id="app"><p>content</p></div>`

    const promise = runProbe()

    // simulate drag from (10,10) to (110,80)
    document.dispatchEvent(new MouseEvent('mousedown', { clientX: 10, clientY: 10, button: 0, bubbles: true, cancelable: true }))
    document.dispatchEvent(new MouseEvent('mousemove', { clientX: 60, clientY: 45, bubbles: true, cancelable: true }))
    document.dispatchEvent(new MouseEvent('mouseup', { clientX: 110, clientY: 80, bubbles: true, cancelable: true }))

    const result = await promise
    expect(result.kind).toBe('region')
    if (result.kind === 'region') {
      expect(result.region.rect).toEqual({ x: 10, y: 10, width: 100, height: 70 })
    }

    // the trailing click after a drag must NOT produce a second resolution
    // (promise already settled; if it misbehaves we'd get unhandled picks)
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
  })

  it('a tiny drag below threshold falls through to click picking', async () => {
    document.body.innerHTML = `<button id="b">x</button>`
    const button = document.querySelector('button')!
    button.getBoundingClientRect = () =>
      ({ x: 0, y: 0, width: 10, height: 10, top: 0, left: 0, right: 10, bottom: 10, toJSON: () => ({}) }) as DOMRect
    const originalEFP = document.elementFromPoint
    document.elementFromPoint = () => button

    const promise = runProbe()

    // 3px drag — below the 6px threshold
    document.dispatchEvent(new MouseEvent('mousedown', { clientX: 5, clientY: 5, button: 0, bubbles: true, cancelable: true }))
    document.dispatchEvent(new MouseEvent('mouseup', { clientX: 8, clientY: 8, bubbles: true, cancelable: true }))
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    const result = await promise
    expect(result.kind).toBe('picked')

    document.elementFromPoint = originalEFP
  })
})
