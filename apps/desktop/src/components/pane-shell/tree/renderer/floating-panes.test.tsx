/**
 * Live-behavior test for FloatingPanes: mounts the REAL component into a real
 * DOM and drives real pointer/resize events. This is the closest thing to
 * "running it" without an Electron main process — it exercises the rendered
 * element's computed geometry, not the pure geometry module (that's
 * floating-rect.test.ts).
 */

import { act } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { reactRoot } from '@/test/react-root'

import { FloatingPanes } from './floating-panes'

const mount = reactRoot()
let disposers: (() => void)[] = []

const card = () => document.querySelector<HTMLElement>('[data-floating-pane="hud"]')

const grab = () => card()!.querySelector('header')!

/** jsdom has no real pointer events; PointerEvent falls back to MouseEvent. */
function pointer(target: Element, type: string, x: number, y: number) {
  const event = new MouseEvent(type, { bubbles: true, clientX: x, clientY: y })

  Object.defineProperty(event, 'pointerId', { value: 1 })

  act(() => {
    target.dispatchEvent(event)
  })
}

function resizeWindow(width: number, height: number) {
  Object.defineProperty(window, 'innerWidth', { configurable: true, value: width, writable: true })
  Object.defineProperty(window, 'innerHeight', { configurable: true, value: height, writable: true })

  act(() => {
    window.dispatchEvent(new Event('resize'))
  })
}

function registerHud(data: Record<string, unknown>) {
  disposers.push(
    registry.register({
      area: 'panes',
      data,
      id: 'hud',
      render: () => <p data-testid="hud-body">live</p>,
      title: 'HUD'
    })
  )
}

describe('FloatingPanes (live DOM)', () => {
  beforeEach(() => {
    window.localStorage.clear()
    resizeWindow(1440, 900)
    // setPointerCapture / releasePointerCapture don't exist in jsdom.
    Element.prototype.setPointerCapture = vi.fn()
    Element.prototype.releasePointerCapture = vi.fn()
  })

  afterEach(() => {
    mount.unmount()
    disposers.forEach(dispose => dispose())
    disposers = []
  })

  it('mounts a fixed card in the anchored corner with the pane body inside', () => {
    registerHud({ anchor: 'top-right', height: '132px', placement: 'floating', width: '224px' })
    mount.render(<FloatingPanes />)

    const el = card()!

    expect(el).toBeTruthy()
    expect(el.className).toContain('fixed')
    // 1440 - 224 - 12 margin = 1204; titlebar 34 + 12 = 46.
    expect(el.style.left).toBe('1204px')
    expect(el.style.top).toBe('46px')
    expect(el.style.width).toBe('224px')
    expect(document.querySelector('[data-testid="hud-body"]')?.textContent).toBe('live')
  })

  it('renders nothing for a non-floating placement', () => {
    registerHud({ placement: 'right', width: '224px' })
    mount.render(<FloatingPanes />)

    expect(card()).toBeNull()
  })

  it('moves with a real pointer drag on the header', () => {
    registerHud({ anchor: 'top-left', height: '132px', placement: 'floating', width: '224px' })
    mount.render(<FloatingPanes />)

    expect(card()!.style.left).toBe('12px')

    pointer(grab(), 'pointerdown', 100, 100)
    pointer(grab(), 'pointermove', 260, 240)
    pointer(grab(), 'pointerup', 260, 240)

    expect(card()!.style.left).toBe('172px')
    expect(card()!.style.top).toBe('186px')
  })

  it('persists the dragged position across a remount', () => {
    registerHud({ anchor: 'top-left', height: '132px', placement: 'floating', width: '224px' })
    mount.render(<FloatingPanes />)

    pointer(grab(), 'pointerdown', 100, 100)
    pointer(grab(), 'pointermove', 300, 300)
    pointer(grab(), 'pointerup', 300, 300)

    const moved = card()!.style.left

    mount.unmount()
    mount.render(<FloatingPanes />)

    expect(card()!.style.left).toBe(moved)
  })

  it('rides the right edge when the window shrinks', () => {
    registerHud({ anchor: 'top-right', height: '132px', placement: 'floating', width: '224px' })
    mount.render(<FloatingPanes />)

    expect(card()!.style.left).toBe('1204px')

    resizeWindow(1000, 700)

    // Tracks the edge: 1000 - 224 - 12 = 764.
    expect(card()!.style.left).toBe('764px')
  })

  it('never lets a drag push the card under the titlebar', () => {
    registerHud({ anchor: 'top-left', height: '132px', placement: 'floating', width: '224px' })
    mount.render(<FloatingPanes />)

    pointer(grab(), 'pointerdown', 100, 100)
    pointer(grab(), 'pointermove', 100, -900)
    pointer(grab(), 'pointerup', 100, -900)

    expect(Number.parseFloat(card()!.style.top)).toBeGreaterThanOrEqual(34)
  })

  it('collapses to the header and drops the body, and does not drag from the button', () => {
    registerHud({ anchor: 'top-left', height: '132px', placement: 'floating', width: '224px' })
    mount.render(<FloatingPanes />)

    const before = card()!.style.left
    const toggle = card()!.querySelector('button')!
    const chevron = () => toggle.querySelector('i')!

    // Expanded: down chevron (fold). Collapsed: up chevron (restore).
    expect(chevron().className).toContain('codicon-chevron-down')

    // The button is inside the drag handle — [data-floating-no-drag] must
    // stop it starting a drag.
    pointer(toggle, 'pointerdown', 100, 100)
    pointer(grab(), 'pointermove', 400, 400)
    pointer(grab(), 'pointerup', 400, 400)

    expect(card()!.style.left).toBe(before)

    act(() => {
      toggle.click()
    })

    expect(document.querySelector('[data-testid="hud-body"]')).toBeNull()
    expect(card()!.style.height).toBe('')
    expect(chevron().className).toContain('codicon-chevron-up')
  })

  it('renders one card per floating contribution', () => {
    registerHud({ anchor: 'top-left', placement: 'floating', width: '224px' })
    disposers.push(
      registry.register({
        area: 'panes',
        data: { anchor: 'bottom-right', placement: 'floating', width: '200px' },
        id: 'hud2',
        render: () => null,
        title: 'HUD 2'
      })
    )

    mount.render(<FloatingPanes />)

    expect(document.querySelectorAll('[data-floating-pane]').length).toBe(2)
  })
})
