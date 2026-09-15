import { describe, expect, it, vi } from 'vitest'

import {
  consumePixelWheelCarry,
  createTuiWheelDispatcher,
  encodeInkWheelToPty,
  isPointInTerminalWheelRect,
  isTerminalWheelEventTarget,
  isTuiTitleForWheel,
  nextTuiWheelLatch,
  planTuiWheelWrite,
  shouldFreezeTerminalFitForWheel,
  shouldSendTuiWheelToPty,
  TERMINAL_WHEEL_FIT_FREEZE_MS,
  TERMINAL_WHEEL_PIXEL_PAGE,
  tuiXtermScrollback
} from './terminal-wheel'

describe('shouldFreezeTerminalFitForWheel', () => {
  it('freezes FitAddon for a short window after a wheel', () => {
    expect(shouldFreezeTerminalFitForWheel(1_000, 0)).toBe(false)
    expect(shouldFreezeTerminalFitForWheel(1_000, 900)).toBe(true)
    expect(shouldFreezeTerminalFitForWheel(1_000 + TERMINAL_WHEEL_FIT_FREEZE_MS, 1_000)).toBe(false)
  })
})

describe('shouldSendTuiWheelToPty', () => {
  it('leaves a normal shell to xterm', () => {
    expect(shouldSendTuiWheelToPty('normal', 'zsh')).toBe(false)
    expect(shouldSendTuiWheelToPty('normal', '', false)).toBe(false)
  })

  it('sends on the alternate screen or a Cursor title', () => {
    expect(shouldSendTuiWheelToPty('alternate', '')).toBe(true)
    expect(shouldSendTuiWheelToPty('normal', 'Cursor Agent')).toBe(true)
    expect(isTuiTitleForWheel('Hermes Email Agent')).toBe(true)
    expect(shouldSendTuiWheelToPty('alternate', '', true)).toBe(false)
  })

  it('keeps a resumed chat title on the Ink path after Cursor chrome or alt screen', () => {
    expect(shouldSendTuiWheelToPty('normal', 'Hermes Desktop Fixes')).toBe(true)
    expect(shouldSendTuiWheelToPty('normal', '/home/hermes')).toBe(false)
    expect(nextTuiWheelLatch(false, 'normal', '', 'Run Everything')).toBe(true)
    expect(nextTuiWheelLatch(true, 'normal', 'Hermes Desktop Fixes')).toBe(true)
    expect(nextTuiWheelLatch(true, 'normal', 'zsh')).toBe(false)
    expect(nextTuiWheelLatch(true, 'normal', 'zsh', '', false)).toBe(true)
    expect(shouldSendTuiWheelToPty('normal', 'zsh', false, true, false)).toBe(true)
  })

  it('sends for the live Hatch conversation title on the normal buffer', () => {
    expect(shouldSendTuiWheelToPty('normal', 'Hermes Desktop Fixes', false, false)).toBe(true)
    expect(isPointInTerminalWheelRect(25, 40, { left: 10, right: 80, top: 20, bottom: 90, width: 70, height: 70 })).toBe(
      true
    )
    expect(isPointInTerminalWheelRect(5, 40, { left: 10, right: 80, top: 20, bottom: 90, width: 70, height: 70 })).toBe(
      false
    )

    const host = document.createElement('div')
    host.setAttribute('data-persistent-terminal', '')
    const canvas = document.createElement('canvas')
    host.appendChild(canvas)
    expect(isTerminalWheelEventTarget(canvas)).toBe(true)
    expect(isTerminalWheelEventTarget(document.body)).toBe(false)
  })
})

describe('planTuiWheelWrite', () => {
  it('writes Page Up for the live Hatch Desktop Fixes pane', () => {
    const plan = planTuiWheelWrite({
      bufferType: 'normal',
      deltaY: -TERMINAL_WHEEL_PIXEL_PAGE,
      sessionId: 'h-cfff2c5c',
      title: 'Hermes Desktop Fixes'
    })

    expect(plan.handled).toBe(true)
    expect(plan.sequence).toBe('\x1b[5~')
    expect(plan.sequence).not.toContain('900')
    expect(plan.sequence).not.toBe('\x1b[A')
    expect(plan.sequence).not.toContain('64')
  })

  it('keeps sending after a leftover zsh title once Cursor has latched', () => {
    const plan = planTuiWheelWrite({
      bufferType: 'normal',
      deltaMode: 1,
      deltaY: 1,
      latched: true,
      sessionId: 's1',
      title: 'zsh'
    })

    expect(plan.handled).toBe(true)
    expect(plan.sequence).toBe('\x1b[6~')
    expect(plan.nextLatched).toBe(true)
  })

  it('keeps a restored Cursor tab on Page Up even if the OSC title is zsh', () => {
    const plan = planTuiWheelWrite({
      bufferType: 'normal',
      deltaY: -TERMINAL_WHEEL_PIXEL_PAGE,
      sessionId: 'h-cfff2c5c',
      sticky: true,
      title: 'zsh'
    })

    expect(plan.handled).toBe(true)
    expect(plan.sequence).toBe('\x1b[5~')
  })

  it('dispatches a window wheel over the overlay as Page Up', () => {
    const host = document.createElement('div')
    host.setAttribute('data-persistent-terminal', '')
    document.body.appendChild(host)
    vi.spyOn(host, 'getBoundingClientRect').mockReturnValue({
      bottom: 100,
      height: 100,
      left: 0,
      right: 100,
      top: 0,
      width: 100,
      x: 0,
      y: 0,
      toJSON: () => ({})
    } as DOMRect)

    const writes: string[] = []
    let latched = false
    let carry = 0
    const wheel = createTuiWheelDispatcher({
      getBufferType: () => 'normal',
      getHost: () => host,
      getLatched: () => latched,
      getPixelCarry: () => carry,
      getSessionId: () => 'h-cfff2c5c',
      getTitle: () => 'Hermes Desktop Fixes',
      scrollToBottom: () => undefined,
      setLatched: value => {
        latched = value
      },
      setPixelCarry: value => {
        carry = value
      },
      write: (_id, data) => {
        writes.push(data)
      }
    })

    window.addEventListener('wheel', wheel.onWindowWheel, { capture: true, passive: false })
    const event = new WheelEvent('wheel', {
      bubbles: true,
      cancelable: true,
      clientX: 20,
      clientY: 20,
      deltaMode: 0,
      deltaY: -TERMINAL_WHEEL_PIXEL_PAGE
    })
    Object.defineProperty(event, 'target', { value: host })
    window.dispatchEvent(event)
    window.removeEventListener('wheel', wheel.onWindowWheel, true)
    host.remove()

    expect(writes).toEqual(['\x1b[5~'])
    expect(event.defaultPrevented).toBe(true)
  })
})

describe('encodeInkWheelToPty', () => {
  it('encodes Page Up/Down only after a real page of pixels', () => {
    expect(encodeInkWheelToPty(-4)).toBe('')
    expect(encodeInkWheelToPty(-TERMINAL_WHEEL_PIXEL_PAGE)).toBe('\x1b[5~')
    expect(encodeInkWheelToPty(TERMINAL_WHEEL_PIXEL_PAGE)).toBe('\x1b[6~')
    expect(encodeInkWheelToPty(0)).toBe('')
    expect(encodeInkWheelToPty(-TERMINAL_WHEEL_PIXEL_PAGE)).not.toContain('900')
    expect(encodeInkWheelToPty(-TERMINAL_WHEEL_PIXEL_PAGE)).not.toContain('\x1b[H')
    expect(encodeInkWheelToPty(-TERMINAL_WHEEL_PIXEL_PAGE)).not.toBe('\x15')
    expect(encodeInkWheelToPty(-TERMINAL_WHEEL_PIXEL_PAGE)).not.toBe('\x1b[A')
  })

  it('accumulates tiny pixel ticks into one page key', () => {
    let carry = 0
    let sequence = ''

    for (const tick of [4, 4]) {
      const next = consumePixelWheelCarry(carry + tick)
      carry = next.remaining
      sequence += next.sequence
    }

    expect(sequence).toBe('\x1b[6~')
    expect(carry).toBe(0)
  })
})

describe('tuiXtermScrollback', () => {
  it('clears xterm history on a TUI so leftover wheel cannot show old frames', () => {
    expect(tuiXtermScrollback(true)).toBe(0)
    expect(tuiXtermScrollback(false)).toBe(1000)
  })
})
