import type { PointerEvent as ReactPointerEvent } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../store', async () => {
  const { atom } = await import('nanostores')

  return {
    $dropHint: atom(null),
    $treeDragging: atom(null),
    mergeTreeZones: vi.fn(),
    moveTreePanes: vi.fn(),
    reorderTreePanes: vi.fn()
  }
})
vi.mock('@/lib/reorder', () => ({ reorderCommitHaptic: vi.fn(), reorderStepHaptic: vi.fn() }))
import { $dropHint, $treeDragging } from '../store'

import { startDragSession } from './drag-session'

function pointer(type: string, buttons = 1) {
  const event = new MouseEvent(type, { bubbles: true, clientX: 20, clientY: 20, buttons })
  Object.defineProperty(event, 'pointerId', { value: 1 })

  return event
}

function begin() {
  const handle = document.createElement('button')
  document.body.append(handle)
  handle.setPointerCapture = vi.fn()
  handle.releasePointerCapture = vi.fn()

  const spec = {
    onEngage: vi.fn(() => $treeDragging.set('sessions')),
    resolveMove: vi.fn(() => ({ kind: 'group' as const, groupId: 'main', pos: 'center' as const })),
    onCommit: vi.fn(),
    onEnd: vi.fn(),
    onTap: vi.fn()
  }

  startDragSession(
    {
      button: 0,
      pointerId: 1,
      clientX: 0,
      clientY: 0,
      currentTarget: handle
    } as unknown as ReactPointerEvent<HTMLElement>,
    spec
  )

  return { handle, spec }
}

beforeEach(() => {
  vi.useFakeTimers()
  document.body.style.cursor = 'auto'
  document.body.style.userSelect = 'text'
})
afterEach(() => {
  window.dispatchEvent(pointer('pointercancel', 0))
  window.dispatchEvent(pointer('pointerup', 0))
  vi.runOnlyPendingTimers()
  vi.useRealTimers()
  document.body.replaceChildren()
  $dropHint.set(null)
  $treeDragging.set(null)
})
describe('interrupted drag sessions', () => {
  it.each(['blur', 'lostpointercapture', 'released-move'])(
    'cancels on %s without swallowing the next independent click',
    reason => {
      const { handle, spec } = begin()
      window.dispatchEvent(pointer('pointermove'))
      vi.advanceTimersByTime(32)
      expect($treeDragging.get()).toBe('sessions')
      expect(document.body.style.userSelect).toBe('none')

      if (reason === 'blur') {
        window.dispatchEvent(new Event('blur'))
      } else if (reason === 'lostpointercapture') {
        handle.dispatchEvent(pointer('lostpointercapture', 0))
      } else {
        window.dispatchEvent(pointer('pointermove', 0))
      }

      expect($treeDragging.get()).toBeNull()
      expect($dropHint.get()).toBeNull()
      expect(document.body.style.cursor).toBe('auto')
      expect(document.body.style.userSelect).toBe('text')
      expect(spec.onEnd).toHaveBeenCalledTimes(1)
      expect(spec.onCommit).not.toHaveBeenCalled()
      const click = vi.fn()
      handle.addEventListener('click', click)
      handle.click()
      expect(click).toHaveBeenCalledTimes(1)
      window.dispatchEvent(pointer('pointerup', 0))
      expect(spec.onEnd).toHaveBeenCalledTimes(1)
    }
  )
  it('preserves taps and normal drops while suppressing only the release click', () => {
    const tap = begin()
    window.dispatchEvent(pointer('pointerup', 0))
    expect(tap.spec.onTap).toHaveBeenCalledTimes(1)
    const { handle, spec } = begin()
    window.dispatchEvent(pointer('pointermove'))
    vi.advanceTimersByTime(32)
    window.dispatchEvent(pointer('pointerup', 0))
    expect(spec.onCommit).toHaveBeenCalledWith({ kind: 'group', groupId: 'main', pos: 'center' })
    expect(spec.onEnd).toHaveBeenCalledTimes(1)
    const click = vi.fn()
    handle.addEventListener('click', click)
    handle.click()
    expect(click).not.toHaveBeenCalled()
    vi.runOnlyPendingTimers()
    handle.click()
    expect(click).toHaveBeenCalledTimes(1)
  })
})
