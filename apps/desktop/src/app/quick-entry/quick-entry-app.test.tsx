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
})
