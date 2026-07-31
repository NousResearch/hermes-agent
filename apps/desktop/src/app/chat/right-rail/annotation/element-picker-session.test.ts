import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  buildAddBadgeCall,
  buildFlashCall,
  buildRemoveBadgeCall,
  buildSessionProbeSource,
  buildSetPickingCall,
  buildTeardownCall,
  parseSessionEvent,
  SESSION_CHANNEL,
  SESSION_GLOBAL_KEY,
  type SessionEvent
} from './element-picker-session'

/**
 * Runs the session probe source inside jsdom like webview.executeJavaScript
 * would. The probe returns true immediately and stays resident.
 */
function runProbe(bannerMessage = 'Annotate mode'): boolean {
  const source = buildSessionProbeSource(bannerMessage)
  // eslint-disable-next-line no-eval
  return eval(source) as boolean
}

function sessionApi(): {
  addBadge: (id: string, number: number, x: number, y: number) => void
  flash: (x: number, y: number, w: number, h: number) => void
  removeBadge: (id: string) => void
  setPicking: (on: boolean) => void
  teardown: () => void
} {
  return (window as unknown as Record<string, never>)[SESSION_GLOBAL_KEY] as ReturnType<typeof sessionApi>
}

function stubRect(el: Element, rect: { x: number; y: number; width: number; height: number }) {
  el.getBoundingClientRect = () =>
    ({ ...rect, top: rect.y, left: rect.x, right: rect.x + rect.width, bottom: rect.y + rect.height, toJSON: () => ({}) }) as DOMRect
}

function emittedEvents(logSpy: ReturnType<typeof vi.spyOn>): SessionEvent[] {
  return logSpy.mock.calls
    .map((call: unknown[]) => String(call[0]))
    .filter((message: string) => message.startsWith(SESSION_CHANNEL))
    .map((message: string) => parseSessionEvent(message)!)
}

beforeEach(() => {
  document.documentElement.innerHTML = '<head></head><body></body>'
  vi.restoreAllMocks()
})

describe('annotation session probe', () => {
  it('stays resident and returns true immediately', () => {
    expect(runProbe()).toBe(true)
    expect(sessionApi()).toBeDefined()
    expect(typeof sessionApi().teardown).toBe('function')
    expect(typeof sessionApi().flash).toBe('function')
  })

  it('emits a pick event on click without tearing down', () => {
    document.body.innerHTML = `<div id="app"><button class="submit primary">登录</button></div>`
    const button = document.querySelector('button')!
    stubRect(button, { x: 10, y: 20, width: 80, height: 32 })
    document.elementFromPoint = () => button

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    runProbe()
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    const events = emittedEvents(logSpy)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('pick')
    if (events[0].type === 'pick') {
      expect(events[0].kind).toBe('element')
      expect(typeof events[0].clickX).toBe('number')
      expect(typeof events[0].clickY).toBe('number')
      if (events[0].kind === 'element') {
        expect(events[0].target.selector).toBe('#app > button.submit.primary')
      }
    }

    // Resident: probe API still alive after the pick.
    expect(sessionApi()).toBeDefined()
  })

  it('pauses picking after a pick until the host resumes it', () => {
    document.body.innerHTML = `<button id="a">A</button><button id="b">B</button>`
    const a = document.querySelector('#a')!
    const b = document.querySelector('#b')!
    stubRect(a, { x: 0, y: 0, width: 10, height: 10 })
    stubRect(b, { x: 20, y: 0, width: 10, height: 10 })
    let current: Element = a
    document.elementFromPoint = () => current

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    runProbe()
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    // Second click while paused → no new event.
    current = b
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
    expect(emittedEvents(logSpy)).toHaveLength(1)

    // Resume → next click emits again.
    sessionApi().setPicking(true)
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
    expect(emittedEvents(logSpy)).toHaveLength(2)
  })

  it('pins a numbered badge at the last picked rect and removes it', () => {
    document.body.innerHTML = `<button id="t">T</button>`
    const target = document.querySelector('#t')!
    stubRect(target, { x: 40, y: 50, width: 60, height: 24 })
    document.elementFromPoint = () => target

    vi.spyOn(console, 'log').mockImplementation(() => undefined)
    runProbe()
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    sessionApi().addBadge('ann-1', 1, 40, 50)
    const badge = document.querySelector('[data-hermes-badge="ann-1"]') as HTMLElement
    expect(badge).not.toBeNull()
    expect(badge.textContent).toBe('1')
    expect(badge.style.left).toBe('30px') // 40 - 10 offset
    expect(badge.style.top).toBe('40px') // 50 - 10 offset

    sessionApi().removeBadge('ann-1')
    expect(document.querySelector('[data-hermes-badge="ann-1"]')).toBeNull()
  })

  it('emits badge-click instead of a pick when a badge is clicked', () => {
    document.body.innerHTML = `<button id="t">T</button>`
    const target = document.querySelector('#t')!
    stubRect(target, { x: 40, y: 50, width: 60, height: 24 })

    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    runProbe()
    document.elementFromPoint = () => target
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    sessionApi().setPicking(true)
    sessionApi().addBadge('ann-9', 9, 40, 50)
    const badge = document.querySelector('[data-hermes-badge="ann-9"]')!
    document.elementFromPoint = () => badge
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))

    const events = emittedEvents(logSpy)
    expect(events[events.length - 1]).toEqual({ type: 'badge-click', id: 'ann-9' })
  })

  it('emits cancel-request on Escape and keeps running', () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    runProbe()
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))

    const events = emittedEvents(logSpy)
    expect(events).toHaveLength(1)
    expect(events[0].type).toBe('cancel-request')
    expect(sessionApi()).toBeDefined()
  })

  it('teardown removes overlays, badges and the global API', () => {
    document.body.innerHTML = `<button id="t">T</button>`
    const target = document.querySelector('#t')!
    stubRect(target, { x: 0, y: 0, width: 10, height: 10 })
    document.elementFromPoint = () => target

    vi.spyOn(console, 'log').mockImplementation(() => undefined)
    runProbe()
    document.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }))
    sessionApi().addBadge('ann-1', 1, 10, 10)

    sessionApi().teardown()
    expect(document.querySelector('[data-hermes-badge]')).toBeNull()
    expect((window as unknown as Record<string, unknown>)[SESSION_GLOBAL_KEY]).toBeUndefined()
  })
})

describe('host-side call builders', () => {
  it('builds setPicking / addBadge / removeBadge / teardown calls', () => {
    expect(buildSetPickingCall(true)).toContain('.setPicking(true)')
    expect(buildSetPickingCall(false)).toContain('.setPicking(false)')
    expect(buildAddBadgeCall('ann-1', 3, 12, 34)).toContain('.addBadge("ann-1", 3, 12, 34)')
    expect(buildRemoveBadgeCall('ann-1')).toContain('.removeBadge("ann-1")')
    expect(buildTeardownCall()).toContain('.teardown()')
    expect(buildFlashCall({ height: 50, width: 100, x: 12, y: 34 })).toContain('.flash(12, 34, 100, 50)')
  })
})

describe('parseSessionEvent', () => {
  it('parses channel messages and ignores others', () => {
    const event = { type: 'cancel-request' as const }
    expect(parseSessionEvent(SESSION_CHANNEL + JSON.stringify(event))).toEqual(event)
    expect(parseSessionEvent('ordinary console line')).toBeNull()
    expect(parseSessionEvent(SESSION_CHANNEL + '{broken json')).toBeNull()
  })
})


  describe('rejects malformed payloads (page controls console)', () => {
    it('returns null for pick with missing clickX', () => {
      const event = parseSessionEvent('__HERMES_ANNOTATE__:{"type":"pick","kind":"element","target":{"rect":{"x":0,"y":0,"width":100,"height":100}}}')
      expect(event).toBeNull()
    })

    it('returns null for pick with invalid rect', () => {
      const event = parseSessionEvent('__HERMES_ANNOTATE__:{"type":"pick","kind":"element","clickX":10,"clickY":20,"target":{"rect":{"x":0,"y":"bad","width":100,"height":100}}}')
      expect(event).toBeNull()
    })

    it('returns null for pick with wrong kind', () => {
      const event = parseSessionEvent('__HERMES_ANNOTATE__:{"type":"pick","kind":"malicious","clickX":10,"clickY":20,"target":{"rect":{"x":0,"y":0,"width":100,"height":100}}}')
      expect(event).toBeNull()
    })

    it('returns null for badge-click with non-string id', () => {
      const event = parseSessionEvent('__HERMES_ANNOTATE__:{"type":"badge-click","id":123}')
      expect(event).toBeNull()
    })

    it('returns null for garbage payload', () => {
      const event = parseSessionEvent('__HERMES_ANNOTATE__:{"type":"pick"}')
      expect(event).toBeNull()
    })

    it('still parses a valid pick event', () => {
      const event = parseSessionEvent('__HERMES_ANNOTATE__:{"type":"pick","kind":"element","clickX":10,"clickY":20,"target":{"selector":"#btn","text":"click me","tagName":"BUTTON","rect":{"x":0,"y":0,"width":100,"height":100}}}')
      expect(event).not.toBeNull()
      if (event && event.type === 'pick' && event.kind === 'element') {
        expect(event.kind).toBe('element')
        expect(event.clickX).toBe(10)
        expect(event.clickY).toBe(20)
        expect(event.target.selector).toBe('#btn')
      }
    })
  })
