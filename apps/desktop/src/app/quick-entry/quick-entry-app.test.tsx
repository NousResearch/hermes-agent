/** Regression coverage for #77429: native picker options must paint a readable surface. */

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
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
        setSettings: vi.fn(async (patch: Record<string, unknown>) => ({
          enabled: true,
          error: null,
          registered: true,
          shortcut: 'CommandOrControl+Shift+Space',
          ...patch
        })),
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

  it('remembers a "New session" pick as the default instead of resetting on the next summon', () => {
    render(<QuickEntryApp />)

    act(() => pushState?.({ connected: true, sessions: [] }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Target session' }), { target: { value: 'new' } })

    expect(window.hermesDesktop?.quickEntry?.setSettings).toHaveBeenCalledWith({ target: 'new' })
  })

  it('keeps a picked stored session as a one-off, not a new default', () => {
    render(<QuickEntryApp />)

    act(() => pushState?.({ connected: true, sessions: [{ id: 'session-1', title: 'A recent session' }] }))
    fireEvent.change(screen.getByRole('combobox', { name: 'Target session' }), { target: { value: 'session-1' } })

    expect(window.hermesDesktop?.quickEntry?.setSettings).not.toHaveBeenCalled()
  })

  it('a push without a default target keeps the default already adopted from main', () => {
    render(<QuickEntryApp />)

    act(() => pushState?.({ connected: true, defaultTarget: 'new', sessions: [] }))
    act(() => pushState?.({ connected: true, sessions: [{ id: 'session-1', title: 'A recent session' }] }))

    expect((screen.getByRole('combobox', { name: 'Target session' }) as HTMLSelectElement).value).toBe('new')
  })
})
