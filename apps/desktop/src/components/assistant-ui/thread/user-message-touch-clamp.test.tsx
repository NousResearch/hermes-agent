import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $touchPointer } from '@/lib/touch-interaction'

import {
  assistantMessage,
  stubThreadEnvironment,
  stubThreadViewportSize,
  ThreadRuntime,
  userMessage
} from '../test-utils'

import { Thread } from '.'

const PROMPT = 'a long prompt with pasted image notes'

/** ResizeObservers whose deliveries the test drives, since jsdom has no layout. */
const observers = new Map<Element, Set<ResizeObserverCallback>>()

class DrivenResizeObserver {
  private readonly callback: ResizeObserverCallback

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback
  }

  disconnect() {
    for (const callbacks of observers.values()) {
      callbacks.delete(this.callback)
    }
  }

  observe(element: Element) {
    observers.set(element, (observers.get(element) ?? new Set()).add(this.callback))
  }

  unobserve(element: Element) {
    observers.get(element)?.delete(this.callback)
  }
}

stubThreadEnvironment()
vi.stubGlobal('ResizeObserver', DrivenResizeObserver)
stubThreadViewportSize()

/** Report a laid-out height for the innermost observed box holding the prompt. */
function layOutPrompt(blockSize: number) {
  const text = screen.getByText(PROMPT)
  const holders = [...observers.keys()].filter(element => observers.get(element)?.size && element.contains(text))
  const box = holders.find(element => !holders.some(other => other !== element && element.contains(other)))

  expect(box).toBeTruthy()

  const entry = { target: box, borderBoxSize: [{ blockSize, inlineSize: 600 }] } as unknown as ResizeObserverEntry

  act(() => {
    for (const callback of observers.get(box!) ?? []) {
      callback([entry], {} as ResizeObserver)
    }
  })

  return box!.parentElement!
}

function renderTouchThread() {
  render(
    <ThreadRuntime messages={[userMessage('user-1', PROMPT), assistantMessage()]}>
      <Thread />
    </ThreadRuntime>
  )
}

beforeEach(() => {
  window.document.documentElement.dataset.hermesDesktopHost = 'browser'
  $touchPointer.set(true)
})

afterEach(() => {
  cleanup()
  $touchPointer.set(false)
  delete window.document.documentElement.dataset.hermesDesktopHost
})

describe('touch user bubble clamp', () => {
  // A tablet has no hover and no tap-to-edit, and the bubble is sticky: an
  // unclamped long prompt pinned at the top of the transcript hid the reply.
  it('clamps a long prompt and lets an explicit toggle hold it open', async () => {
    renderTouchThread()
    await screen.findByText(PROMPT)

    const body = layOutPrompt(400)

    expect(body.classList.contains('sticky-human-clamp')).toBe(true)
    expect(body.dataset.clamped).toBe('true')

    const toggle = screen.getByRole('button', { name: 'Expand message' })

    expect(toggle.getAttribute('aria-expanded')).toBe('false')

    fireEvent.click(toggle)

    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(body.classList.contains('sticky-human-clamp')).toBe(false)

    fireEvent.click(toggle)

    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    expect(body.classList.contains('sticky-human-clamp')).toBe(true)
  })

  it('offers no toggle for a prompt that fits the clamp', async () => {
    renderTouchThread()
    await screen.findByText(PROMPT)

    const body = layOutPrompt(20)

    expect(body.dataset.clamped).toBeUndefined()
    expect(screen.queryByRole('button', { name: 'Expand message' })).toBeNull()
  })
})
