// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ThreadTimeline } from './timeline'

afterEach(() => cleanup())

// Stub AUI: useAuiState receives the selector and is called immediately inside
// the component. We feed a stable messages array so deriveTimelineEntries
// returns predictable entries. The hook below supports per-test overrides by
// letting the test set the active mock before render.
const useAuiStateMock = vi.fn()

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (s: unknown) => unknown) => useAuiStateMock(selector)
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))

function messagesWith(rows: Array<{ id: string; text: string }>) {
  return { thread: { messages: rows.map(r => ({ id: r.id, role: 'user', content: r.text })) } }
}

const fourPrompts = messagesWith([
  { id: 'u1', text: 'first prompt' },
  { id: 'u2', text: 'second prompt' },
  { id: 'u3', text: 'third prompt' },
  { id: 'u4', text: 'fourth prompt' }
])

const threePrompts = messagesWith([
  { id: 'u1', text: 'one' },
  { id: 'u2', text: 'two' },
  { id: 'u3', text: 'three' }
])

const whitespaceOnly = messagesWith([
  { id: 'u1', text: '   ' },
  { id: 'u2', text: '\n\n' },
  { id: 'u3', text: '\t' },
  { id: 'u4', text: 'real prompt' }
])

describe('ThreadTimeline (keyboard accessibility)', () => {
  it('renders a button trigger with the screen-reader-meaningful ARIA attributes', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const trigger = screen.getByRole('button', { name: /conversation timeline/i })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(trigger.getAttribute('aria-controls')).toBeTruthy()
  })

  it('opens the popover on Enter and closes on Escape (trigger has focus)', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const trigger = screen.getByRole('button', { name: /conversation timeline/i })

    fireEvent.keyDown(trigger, { key: 'Enter' })
    expect(trigger.getAttribute('aria-expanded')).toBe('true')

    fireEvent.keyDown(trigger, { key: 'Escape' })
    expect(trigger.getAttribute('aria-expanded')).toBe('false')
  })

  it('opens the popover on Space', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const trigger = screen.getByRole('button', { name: /conversation timeline/i })

    fireEvent.keyDown(trigger, { key: ' ' })
    expect(trigger.getAttribute('aria-expanded')).toBe('true')
  })

  it('uses a positional, preview-bearing accessible name on every popover row', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)
    fireEvent.keyDown(screen.getByRole('button', { name: /conversation timeline/i }), { key: 'Enter' })

    const group = screen.getByRole('group', { name: /past user prompts/i })
    expect(group).toBeTruthy()

    // 4 rows; each should read "Prompt N of 4: <preview>"
    expect(within(group).getByRole('button', { name: /prompt 1 of 4: first prompt/i })).toBeTruthy()
    expect(within(group).getByRole('button', { name: /prompt 2 of 4: second prompt/i })).toBeTruthy()
    expect(within(group).getByRole('button', { name: /prompt 3 of 4: third prompt/i })).toBeTruthy()
    expect(within(group).getByRole('button', { name: /prompt 4 of 4: fourth prompt/i })).toBeTruthy()
  })

  it('returns null when there are fewer than 4 user prompts (MIN_ENTRIES invariant)', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(threePrompts))
    const { container } = render(<ThreadTimeline />)

    expect(container.firstChild).toBeNull()
  })

  it('Escape on the popover container closes the popover and returns focus to the trigger', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)
    const trigger = screen.getByRole('button', { name: /conversation timeline/i })

    fireEvent.keyDown(trigger, { key: 'Enter' })
    expect(trigger.getAttribute('aria-expanded')).toBe('true')

    // Fire Escape on the popover container; the listener calls closeNow() and triggerRef.current?.focus()
    const group = screen.getByRole('group', { name: /past user prompts/i })
    fireEvent.keyDown(group, { key: 'Escape' })

    expect(trigger.getAttribute('aria-expanded')).toBe('false')
    expect(document.activeElement).toBe(trigger)
  })
})
