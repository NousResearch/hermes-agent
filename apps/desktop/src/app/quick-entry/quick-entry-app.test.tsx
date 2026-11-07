/** Regression coverage for #77429: native picker options must paint a readable surface. */

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { QuickEntryStatePush } from '@/store/quick-entry'

import { QuickEntryApp } from './quick-entry-app'

const initialHermesDesktop = window.hermesDesktop

describe('QuickEntryApp', () => {
  let pushState: ((payload: QuickEntryStatePush) => void) | undefined
  let submit: ReturnType<typeof vi.fn>

  beforeEach(() => {
    pushState = undefined
    submit = vi.fn()
    window.hermesDesktop = {
      quickEntry: {
        dismiss: vi.fn(),
        onShown: vi.fn(() => vi.fn()),
        onState: vi.fn(callback => {
          pushState = callback

          return vi.fn()
        }),
        onLateResult: vi.fn(() => vi.fn()),
        submit
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

  it('paints a failed submit with the theme destructive token', async () => {
    submit.mockResolvedValueOnce({
      message: 'The gateway went away.',
      ok: false
    })

    render(<QuickEntryApp />)

    act(() => {
      pushState?.({ connected: true, sessions: [] })
    })

    const input = screen.getByRole('textbox', { name: 'Quick Entry' })
    fireEvent.change(input, { target: { value: 'ship it' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => {
      expect(screen.getByRole('alert').style.color).toBe('var(--dt-destructive)')
    })
  })
})
