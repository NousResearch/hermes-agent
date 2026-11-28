import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $dropHint, $treeDragging } from '../store'

import { startDragSession } from './drag-session'

let handle: HTMLDivElement

function pointer(type: string, init: PointerEventInit = {}) {
  const event = new PointerEvent(type, {
    bubbles: true,
    clientX: 100,
    clientY: 100,
    pointerId: 1,
    pointerType: 'mouse',
    ...init
  })

  handle.dispatchEvent(event)
}

function start() {
  const spec = {
    ghost: { label: 'Session' },
    onCommit: vi.fn(),
    onEnd: vi.fn(),
    onEngage: vi.fn(),
    onTap: vi.fn(),
    resolveMove: vi.fn(() => ({ kind: 'group' as const, groupId: 'g', pos: 'center' as const }))
  }

  startDragSession(
    {
      button: 0,
      buttons: 1,
      clientX: 10,
      clientY: 10,
      currentTarget: handle,
      pointerId: 1
    } as unknown as ReactPointerEvent<HTMLElement>,
    spec
  )

  return spec
}

const frame = () => new Promise(resolve => requestAnimationFrame(resolve))

beforeEach(() => {
  handle = document.createElement('div')
  document.body.append(handle)
})

afterEach(() => {
  window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
  handle.remove()
  $dropHint.set(null)
  $treeDragging.set(null)
})

describe('drag session lifetime', () => {
  it('ignores capture loss bubbling from a child when capture transfers to the handle', async () => {
    const child = document.createElement('button')
    handle.append(child)
    const spec = start()

    pointer('pointermove', { buttons: 1 })
    await frame()
    child.dispatchEvent(new PointerEvent('lostpointercapture', { bubbles: true, pointerId: 1 }))

    expect(spec.onEnd).not.toHaveBeenCalled()
    pointer('pointerup')
    expect(spec.onCommit).toHaveBeenCalledTimes(1)
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
  })

  it('preserves a plain tap without engaging the drag', () => {
    const spec = start()

    pointer('pointerup', { clientX: 10, clientY: 10 })
    expect(spec.onTap).toHaveBeenCalledTimes(1)
    expect(spec.onEngage).not.toHaveBeenCalled()
    expect(spec.onCommit).not.toHaveBeenCalled()
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
  })

  it('preserves a real drag and tears down capture exactly once', async () => {
    document.body.style.cursor = 'crosshair'
    document.body.style.userSelect = 'text'
    const spec = start()
    handle.releasePointerCapture = () => pointer('lostpointercapture')

    pointer('pointermove', { buttons: 1 })
    await frame()
    pointer('pointerup')

    expect(spec.onCommit).toHaveBeenCalledTimes(1)
    expect(spec.onTap).not.toHaveBeenCalled()
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
    expect(document.body.style.cursor).toBe('crosshair')
    expect(document.body.style.userSelect).toBe('text')
    expect(document.body.classList.contains('guest-pointer-lock')).toBe(false)
    expect($dropHint.get()).toBeNull()
  })

  it('cancels pending frame work on blur before the drag engages', async () => {
    const spec = start()

    pointer('pointermove', { buttons: 1 })
    window.dispatchEvent(new Event('blur'))
    await frame()
    expect(spec.onEngage).not.toHaveBeenCalled()
    expect(spec.onCommit).not.toHaveBeenCalled()
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
  })

  it('preserves Escape cancellation and suppresses its release click', async () => {
    const spec = start()
    const onClick = vi.fn()
    handle.addEventListener('click', onClick)
    pointer('pointermove', { buttons: 1 })
    await frame()
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    pointer('pointerup')
    handle.dispatchEvent(new MouseEvent('click', { bubbles: true }))

    expect(spec.onCommit).not.toHaveBeenCalled()
    expect(spec.onTap).not.toHaveBeenCalled()
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
    expect(onClick).not.toHaveBeenCalled()
  })

  it.each(['hover', 'blur', 'lostpointercapture'])(
    'does not swallow a later ordinary click after %s cancellation',
    async type => {
      const spec = start()
      const onClick = vi.fn()
      handle.addEventListener('click', onClick)
      pointer('pointermove', { buttons: 1 })
      await frame()

      if (type === 'hover') {
        pointer('pointermove', { buttons: 0 })
      } else if (type === 'blur') {
        window.dispatchEvent(new Event('blur'))
      } else {
        pointer(type)
      }

      handle.dispatchEvent(new MouseEvent('click', { bubbles: true }))
      expect(onClick).toHaveBeenCalledTimes(1)
      expect(spec.onCommit).not.toHaveBeenCalled()
    }
  )

  it.each(['pointermove', 'pointerup', 'pointercancel'])('ignores %s from another pointer', async type => {
    const spec = start()

    pointer(type, { buttons: type === 'pointermove' ? 1 : 0, pointerId: 2 })
    await frame()
    expect(spec.onEnd).not.toHaveBeenCalled()
    expect(spec.onEngage).not.toHaveBeenCalled()
    expect(spec.onTap).not.toHaveBeenCalled()

    pointer('pointerup', { clientX: 10, clientY: 10 })
    expect(spec.onTap).toHaveBeenCalledTimes(1)
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
  })

  it.each(['blur', 'lostpointercapture'])('cancels on %s and tears down only once', async type => {
    const spec = start()

    // Simulate native releasePointerCapture emitting its event re-entrantly.
    handle.releasePointerCapture = () => pointer('lostpointercapture')
    pointer('pointermove', { buttons: 1 })
    await frame()
    expect(spec.onEngage).toHaveBeenCalledTimes(1)

    if (type === 'blur') {
      window.dispatchEvent(new Event('blur'))
    } else {
      pointer(type)
    }

    pointer('pointerup')
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
    expect(spec.onCommit).not.toHaveBeenCalled()
    expect(spec.onTap).not.toHaveBeenCalled()
    expect(document.body.classList.contains('guest-pointer-lock')).toBe(false)
    expect($dropHint.get()).toBeNull()
  })

  it('cancels a missed release instead of starting a drag on button-free hover', async () => {
    const spec = start()

    pointer('pointermove', { buttons: 0 })
    await frame()
    pointer('pointerup')

    expect(spec.onEngage).not.toHaveBeenCalled()
    expect(spec.onCommit).not.toHaveBeenCalled()
    expect(spec.onTap).not.toHaveBeenCalled()
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
  })
})
