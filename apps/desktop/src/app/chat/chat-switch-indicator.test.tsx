import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $profileSwitchFailure } from '@/store/profile-switch-state'

import { ChatSwitchIndicator, SWITCH_SLOW_MS, SWITCH_STALLED_MS } from './chat-switch-indicator'

// vi.mock is hoisted above the imports.
vi.mock('@/components/onboarding/first-run/assembly', () => ({ useOnboardingChatActive: () => false }))
vi.mock('@/store/profile', () => ({ $profileColors: atom({}) }))

let root: HTMLElement
const indicator = () => root.querySelector('[data-slot="chat-switch-indicator"]')!

describe('ChatSwitchIndicator', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.clearAllTimers()
    vi.useRealTimers()
    $profileSwitchFailure.set(null)
  })

  it('escalates by elapsed time: nothing, then a corner pill, then an inline strip', () => {
    root = render(<ChatSwitchIndicator failure={null} syncing={null} target="turqoise" />).container

    // A fast switch paints nothing at all.
    expect(indicator().getAttribute('data-switch-state')).toBe('pending')
    expect(screen.queryByText(/turqoise/)).toBeNull()

    act(() => vi.advanceTimersByTime(SWITCH_SLOW_MS))
    expect(indicator().getAttribute('data-switch-state')).toBe('slow')
    expect(screen.getByText('Switching to turqoise…')).toBeTruthy()
    expect(screen.queryByText(/will wait/)).toBeNull()

    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS - SWITCH_SLOW_MS))
    expect(indicator().getAttribute('data-switch-state')).toBe('stalled')
    expect(screen.getByText(/will wait and go to this agent/)).toBeTruthy()
  })

  it('restarts the clock for a new target and goes idle when the switch settles', () => {
    const rendered = render(<ChatSwitchIndicator failure={null} syncing={null} target="turqoise" />)
    root = rendered.container
    const { rerender } = rendered

    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))
    rerender(<ChatSwitchIndicator failure={null} syncing={null} target="amber" />)
    expect(indicator().getAttribute('data-switch-state')).toBe('pending')
    expect(screen.queryByText(/amber|turqoise/)).toBeNull()

    rerender(<ChatSwitchIndicator failure={null} syncing={null} target={null} />)
    expect(indicator().getAttribute('data-switch-state')).toBe('idle')
  })

  it('restarts the grace period when retrying the same failed target', () => {
    const rendered = render(<ChatSwitchIndicator failure={null} syncing={null} target="research" />)
    root = rendered.container
    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))

    rendered.rerender(
      <ChatSwitchIndicator failure={{ profile: 'research', retry: () => {} }} syncing={null} target={null} />
    )
    rendered.rerender(<ChatSwitchIndicator failure={null} syncing={null} target="research" />)

    expect(indicator().getAttribute('data-switch-state')).toBe('pending')
    expect(screen.queryByText(/Switching to research|will wait/)).toBeNull()
    act(() => vi.advanceTimersByTime(SWITCH_SLOW_MS))
    expect(indicator().getAttribute('data-switch-state')).toBe('slow')
    expect(screen.queryByText(/will wait/)).toBeNull()
  })

  it('renders the paint-first sync as the quiet state', () => {
    root = render(<ChatSwitchIndicator failure={null} syncing="shadow" target={null} />).container

    expect(indicator().getAttribute('data-switch-state')).toBe('syncing')
    expect(screen.getByText('Syncing shadow…')).toBeTruthy()
  })

  it('is a polite live region that never covers the conversation', () => {
    root = render(<ChatSwitchIndicator failure={null} syncing={null} target="turqoise" />).container
    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS))

    const region = indicator()

    expect(region.getAttribute('aria-live')).toBe('polite')
    expect(region.closest('[aria-hidden]')).toBeNull()
    expect(region.querySelector('[aria-hidden="true"] *')?.textContent ?? '').not.toContain('turqoise')

    for (const element of [region, ...region.querySelectorAll('*')]) {
      expect(element.className.toString()).not.toMatch(/\binset-0\b/)
    }
  })

  it('ends a failed switch in a terminal state with Retry and dismiss', () => {
    const retry = vi.fn()
    $profileSwitchFailure.set({ profile: 'turqoise', retry })
    root = render(
      <ChatSwitchIndicator failure={{ profile: 'turqoise', retry }} syncing={null} target={null} />
    ).container

    expect(indicator().getAttribute('data-switch-state')).toBe('failed')
    expect(screen.getByText('Couldn’t switch to turqoise')).toBeTruthy()
    // Terminal: no spinner keeps going, however long it stays up.
    act(() => vi.advanceTimersByTime(SWITCH_STALLED_MS * 3))
    expect(screen.queryByRole('status')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    expect(retry).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Close' }))
    expect($profileSwitchFailure.get()).toBeNull()
  })
})
