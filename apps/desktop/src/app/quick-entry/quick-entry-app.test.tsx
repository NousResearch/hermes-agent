/** Regression coverage for #77429: native picker options must paint a readable surface. */

import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { QuickEntryStatePush } from '@/store/quick-entry'

import { QuickEntryApp } from './quick-entry-app'

const initialHermesDesktop = window.hermesDesktop

describe('QuickEntryApp', () => {
  let pushState: ((payload: QuickEntryStatePush) => void) | undefined

  beforeEach(() => {
    pushState = undefined
    window.hermesDesktop = {
      quickEntry: {
        dismiss: vi.fn(),
        onShown: vi.fn(() => vi.fn()),
        onState: vi.fn(callback => {
          pushState = callback

          return vi.fn()
        }),
        onLateResult: vi.fn(() => vi.fn()),
        submit: vi.fn()
      }
    } as never
  })

  afterEach(() => {
    cleanup()
    window.hermesDesktop = initialHermesDesktop
    vi.restoreAllMocks()
  })

  it('paints every native target option with matching theme foreground and background tokens', () => {
    render(<QuickEntryApp />)

    act(() => {
      pushState?.({
        connected: true,
        sessions: [{ id: 'session-1', title: 'A recent session' }]
      })
    })

    const options = screen.getAllByRole('option') as HTMLOptionElement[]
    expect(options).toHaveLength(3)

    for (const option of options) {
      expect(option.style.color).toBe('var(--ui-text-primary, var(--foreground))')
      expect(option.style.backgroundColor).toBe('var(--ui-bg-elevated, var(--background))')
    }
  })

  it('declares the native drag band and takes every control out of it', () => {
    const { container } = render(<QuickEntryApp />)

    // A frameless window only moves where the page says so: the transparent
    // host and the card itself are the [-webkit-app-region:drag] band, which
    // is what lets the user pick the composer up and park it somewhere else.
    const hosts = Array.from(container.querySelectorAll('div')).slice(0, 2)

    expect(hosts).toHaveLength(2)

    for (const host of hosts) {
      expect(host.className).toContain('-webkit-app-region:drag')
    }

    // App-region hit-testing beats DOM order and z-index, so every interactive
    // child must opt back out or it silently becomes a window drag.
    const controls = Array.from(container.querySelectorAll('input, select, label'))

    expect(controls).toHaveLength(3)

    for (const control of controls) {
      expect(control.className).toContain('-webkit-app-region:no-drag')
      expect(control.className).not.toContain('-webkit-app-region:drag]')
    }
  })
})
