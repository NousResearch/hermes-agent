// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ThreadTimeline } from './timeline'

// jsdom does not implement CSS.escape; the production code calls it inside
// scrollToPrompt. Polyfill it with the spec-compliant escape for the only
// characters the test data uses. A real browser provides this natively.
if (typeof CSS === 'undefined' || typeof CSS.escape !== 'function') {
  // @ts-expect-error -- test-only polyfill
  globalThis.CSS = { escape: (value: string) => value.replace(/([^\w-])/g, '\\$1') }
}

afterEach(() => cleanup())

// Stub AUI: useAuiState receives the selector and is called immediately inside
// the component. We feed a stable messages array so deriveTimelineEntries
// returns predictable entries. The hook below supports per-test overrides by
// letting the test set the active mock before render.
const useAuiStateMock = vi.fn()

vi.mock('@assistant-ui/react', () => ({
  useAuiState: (selector: (s: unknown) => unknown) => useAuiStateMock(selector)
}))

// Mock the layout store's $timelineVisible atom. `useStore` (from
// @nanostores/react) reads via getSnapshot and re-renders on subscribe; this
// shape mirrors the project's existing @/store/layout mocks (see
// app/chat/sidebar/projects/project-menu.test.tsx) but with a mutable backing
// value so individual tests can flip visibility.
let timelineVisibleValue = true

const timelineVisibleMock = {
  get: () => timelineVisibleValue,
  listen: (fn: (v: boolean) => void) => {
    fn(timelineVisibleValue)

    return () => {}
  },
  subscribe: (fn: (v: boolean) => void) => {
    fn(timelineVisibleValue)

    return () => {}
  },
  set: (v: boolean) => {
    timelineVisibleValue = v
  }
}

vi.mock('@/store/layout', () => ({
  get $timelineVisible() {
    return timelineVisibleMock
  },
  toggleTimelineVisible: () => {
    timelineVisibleValue = !timelineVisibleValue
  }
}))

// Stub the haptic hook so the click→scroll→haptic path is observable per-test
// when we provide a viewport + message node. Each test that wants to observe
// the haptic can call `vi.mocked(triggerHaptic).mockClear()` first.
const { triggerHaptic } = vi.hoisted(() => ({ triggerHaptic: vi.fn() }))

vi.mock('@/lib/haptics', () => ({ triggerHaptic }))

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

/**
 * Mount a fake AUI thread viewport with one data-message-id per user prompt, so
 * scrollToPrompt's `if (!viewport || !node) return` guard is satisfied and the
 * call reaches triggerHaptic. Without this stub, jsdom has no viewport and
 * every tick click returns silently, making it impossible to assert the click
 * was processed.
 */
function mountFakeViewport(promptIds: string[]) {
  const viewport = document.createElement('div')

  viewport.setAttribute('data-slot', 'aui_thread-viewport')

  for (const id of promptIds) {
    const node = document.createElement('div')

    node.setAttribute('data-message-id', id)
    viewport.appendChild(node)
  }

  // getBoundingClientRect returns zeros in jsdom; that's fine — the math
  // reduces to scrollTop + 0 - 8, and the call reaches triggerHaptic.
  viewport.getBoundingClientRect = () => ({
    top: 0,
    bottom: 0,
    left: 0,
    right: 0,
    width: 0,
    height: 0,
    x: 0,
    y: 0,
    toJSON: () => ({})
  })

  document.body.appendChild(viewport)
}

beforeEach(() => {
  // Reset the visibility flag and clear any prior DOM stubs between tests.
  timelineVisibleValue = true
  triggerHaptic.mockClear()
  document.body.innerHTML = ''
})

describe('ThreadTimeline (keyboard accessibility)', () => {
  it('renders a group wrapper with the screen-reader-meaningful label', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    // The outer wrapper is now role="group" (not role="button"). It used to
    // also have aria-expanded/aria-controls, but those belong on a button;
    // dropping them was part of decoupling tick activation from the wrapper.
    const wrapper = screen.getByRole('group', { name: /conversation timeline/i })
    expect(wrapper).toBeTruthy()
  })

  it('Enter on the wrapper is a no-op (does not open the popover or shift focus)', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const wrapper = screen.getByRole('group', { name: /conversation timeline/i })

    // The popover container is always rendered (visibility is via CSS), so we
    // can find it. Before this fix, the wrapper had role="button" and a
    // keyDown handler that called toggle(); now it does nothing.
    const popover = screen.getByRole('group', { name: /past user prompts/i })
    const popoverClassBefore = popover.className

    fireEvent.keyDown(wrapper, { key: 'Enter' })
    fireEvent.keyDown(wrapper, { key: ' ' })
    fireEvent.keyDown(wrapper, { key: 'Escape' })

    // The popover's className should not have changed (no class includes
    // "pointer-events-auto opacity-100 translate-x-0" pre-toggle).
    expect(popover.className).toBe(popoverClassBefore)
  })

  it('uses a positional, "Jump to prompt N of M" accessible name on every tick', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const wrapper = screen.getByRole('group', { name: /conversation timeline/i })

    // 4 ticks, all reachable inside the wrapper as buttons. The "Jump to" prefix
    // tells a screen-reader user what activating the tick will do, vs. the
    // ambiguous "Prompt N of M" the popover rows use.
    expect(within(wrapper).getByRole('button', { name: /jump to prompt 1 of 4: first prompt/i })).toBeTruthy()
    expect(within(wrapper).getByRole('button', { name: /jump to prompt 2 of 4: second prompt/i })).toBeTruthy()
    expect(within(wrapper).getByRole('button', { name: /jump to prompt 3 of 4: third prompt/i })).toBeTruthy()
    expect(within(wrapper).getByRole('button', { name: /jump to prompt 4 of 4: fourth prompt/i })).toBeTruthy()
  })

  it('returns null when there are fewer than 4 user prompts (MIN_ENTRIES invariant)', () => {
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(threePrompts))
    const { container } = render(<ThreadTimeline />)

    expect(container.firstChild).toBeNull()
  })
})

describe('ThreadTimeline (visibility toggle)', () => {
  it('returns null when $timelineVisible is false, removing the rail from the Tab order', () => {
    timelineVisibleValue = false
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    const { container } = render(<ThreadTimeline />)

    expect(container.firstChild).toBeNull()
    // No "conversation timeline" group should be in the DOM, which is what
    // removes the wall of tick stops from the Tab order.
    expect(screen.queryByRole('group', { name: /conversation timeline/i })).toBeNull()
  })

  it('renders the rail again when $timelineVisible flips back to true', () => {
    timelineVisibleValue = false
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    const { rerender } = render(<ThreadTimeline />)

    expect(screen.queryByRole('group', { name: /conversation timeline/i })).toBeNull()

    timelineVisibleValue = true
    rerender(<ThreadTimeline />)

    expect(screen.getByRole('group', { name: /conversation timeline/i })).toBeTruthy()
  })
})

describe('ThreadTimeline (tick activation)', () => {
  it('clicking a tick stops propagation so the popover does not toggle and the jump fires exactly once', () => {
    mountFakeViewport(['u1', 'u2', 'u3', 'u4'])
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const wrapper = screen.getByRole('group', { name: /conversation timeline/i })
    const popover = screen.getByRole('group', { name: /past user prompts/i })
    const popoverClassBefore = popover.className

    const tick = within(wrapper).getByRole('button', { name: /jump to prompt 2 of 4: second prompt/i })
    fireEvent.click(tick)

    // The popover's className did not change — the click didn't bubble to a
    // popover toggle. (Before this fix, the wrapper had onClick={toggle} which
    // would have flipped the popover open.)
    expect(popover.className).toBe(popoverClassBefore)
    // The jump path ran once (triggerHaptic is the observable side-effect at
    // the end of scrollToPrompt).
    expect(triggerHaptic).toHaveBeenCalledTimes(1)
    expect(triggerHaptic).toHaveBeenCalledWith('selection')
  })

  it('clicking a tick that has no matching DOM node is a silent no-op (does not throw)', () => {
    // No fake viewport mounted; scrollToPrompt will return early at the
    // !viewport || !node guard. The click handler must not throw.
    useAuiStateMock.mockImplementation((selector: (s: unknown) => unknown) => selector(fourPrompts))
    render(<ThreadTimeline />)

    const wrapper = screen.getByRole('group', { name: /conversation timeline/i })
    const tick = within(wrapper).getByRole('button', { name: /jump to prompt 3 of 4: third prompt/i })

    expect(() => fireEvent.click(tick)).not.toThrow()
    expect(triggerHaptic).not.toHaveBeenCalled()
  })
})
