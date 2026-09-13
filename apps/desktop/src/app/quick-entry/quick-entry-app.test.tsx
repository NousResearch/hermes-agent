import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { QUICK_ENTRY_TARGET_STORAGE_KEY, QUICK_TARGET_NEW } from '@/store/quick-entry'

// The bridge API the window calls; the primary-renderer side is irrelevant here.
const quickEntryApi = {
  dismiss: vi.fn(),
  onShown: vi.fn(),
  onState: vi.fn(),
  onSubmit: vi.fn(),
  submit: vi.fn()
}

beforeAll(() => {
  ;(window as unknown as { hermesDesktop?: { quickEntry: typeof quickEntryApi } }).hermesDesktop = {
    quickEntry: quickEntryApi
  }
})

beforeEach(() => {
  localStorage.removeItem(QUICK_ENTRY_TARGET_STORAGE_KEY)
  vi.clearAllMocks()
  // Simulate the primary renderer pushing a live gateway so the target picker
  // is enabled; call the registered onState listener synchronously.
  quickEntryApi.onState.mockImplementation((cb: (payload: { connected: boolean; sessions: { id: string; title: string }[] }) => void) => {
    cb({ connected: true, sessions: [{ id: 's1', title: 'Fix the build' }] })
    return () => {}
  })
})

describe('QuickEntryApp target persistence integration', () => {
  it('writes the picker choice to localStorage when the target select changes', async () => {
    const { QuickEntryApp } = await import('./quick-entry-app')

    render(<QuickEntryApp />)

    // The target picker is enabled once the reducer is connected; drive the
    // select's onChange directly like the window does.
    const select = screen.getByLabelText('Target session')
    fireEvent.change(select, { target: { value: QUICK_TARGET_NEW } })

    // The event handler persists before dispatching.
    expect(localStorage.getItem(QUICK_ENTRY_TARGET_STORAGE_KEY)).toBe(QUICK_TARGET_NEW)
  })

  it('persists a specific session id choice', async () => {
    const { QuickEntryApp } = await import('./quick-entry-app')

    render(<QuickEntryApp />)

    const select = screen.getByLabelText('Target session')
    fireEvent.change(select, { target: { value: 's1' } })

    expect(localStorage.getItem(QUICK_ENTRY_TARGET_STORAGE_KEY)).toBe('s1')
  })
})
