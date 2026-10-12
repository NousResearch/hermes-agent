import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { clearAllPrompts, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import {
  onScrollToBottomRequest,
  publishThreadMessagesBelow,
  resetThreadScroll,
  setThreadAtBottom
} from '@/store/thread-scroll'

import { ComposerSurfaceProvider } from './composer/scope'
import { ScrollToBottomButton } from './scroll-to-bottom-button'

function pendingApproval() {
  $activeSessionId.set('sess-1')
  setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: 'sess-1' })
}

afterEach(() => {
  cleanup()
  clearAllPrompts()

  for (const id of ['sess-1', 'sess-a', 'sess-b', 'tile-runtime', 'surface-a', 'surface-b']) {
    resetThreadScroll(id)
  }

  $activeSessionId.set(null)

  for (const stack of document.querySelectorAll('[data-approval-stack]')) {
    stack.remove()
  }
})

// `getByRole('button')` excludes aria-hidden nodes, so "queryByRole null" is the
// control's hidden (parked-at-bottom) state.
describe('ScrollToBottomButton', () => {
  it('isolates pre-runtime panes using their existing composer surface identity', () => {
    setThreadAtBottom(false, 'surface-a')
    publishThreadMessagesBelow(4, { paneVisible: true, sessionId: 'surface-a' })
    const first = vi.fn()
    const second = vi.fn()
    const stopFirst = onScrollToBottomRequest(first, 'surface-a')
    const stopSecond = onScrollToBottomRequest(second, 'surface-b')
    render(
      <>
        <ComposerSurfaceProvider value="surface-a">
          <ScrollToBottomButton sessionId={null} />
        </ComposerSurfaceProvider>
        <ComposerSurfaceProvider value="surface-b">
          <ScrollToBottomButton sessionId={null} />
        </ComposerSurfaceProvider>
      </>
    )

    expect(screen.getAllByRole('button')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'Scroll to bottom · 4 messages' }))
    expect(first).toHaveBeenCalledOnce()
    expect(second).not.toHaveBeenCalled()
    stopFirst()
    stopSecond()
  })

  it('does not light a sibling pane when only one visible session scrolls up', () => {
    setThreadAtBottom(false, 'sess-a')
    render(<ScrollToBottomButton sessionId="sess-b" />)

    expect(screen.queryByRole('button')).toBeNull()
  })

  it("keeps each visible pane's message count when its sibling publishes or unmounts", () => {
    setThreadAtBottom(false, 'sess-a')
    setThreadAtBottom(false, 'sess-b')
    publishThreadMessagesBelow(12, { paneVisible: true, sessionId: 'sess-a' })
    publishThreadMessagesBelow(3, { paneVisible: true, sessionId: 'sess-b' })
    render(
      <>
        <ScrollToBottomButton sessionId="sess-a" />
        <ScrollToBottomButton sessionId="sess-b" />
      </>
    )

    expect(screen.getByRole('button', { name: 'Scroll to bottom · 12 messages' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Scroll to bottom · 3 messages' })).toBeTruthy()
    act(() => resetThreadScroll('sess-b'))
    expect(screen.getByRole('button', { name: 'Scroll to bottom · 12 messages' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Scroll to bottom · 3 messages' })).toBeNull()
  })

  it('stays hidden while parked at the bottom', () => {
    render(<ScrollToBottomButton sessionId="sess-1" />)

    expect(screen.queryByRole('button')).toBeNull()
  })

  it('shows the messages below the viewport when scrolled up with no approval', () => {
    setThreadAtBottom(false, 'sess-1')
    publishThreadMessagesBelow(12, { paneVisible: true, sessionId: 'sess-1' })
    render(<ScrollToBottomButton sessionId="sess-1" />)

    expect(screen.getByRole('button', { name: 'Scroll to bottom · 12 messages' }).textContent).toBe('12 messages')
    expect(screen.queryByText('Approval needed')).toBeNull()
  })

  it('morphs into the approval pill when scrolled up with a pending approval', () => {
    pendingApproval()
    setThreadAtBottom(false, 'sess-1')
    render(<ScrollToBottomButton sessionId="sess-1" />)

    expect(screen.getByRole('button', { name: 'Approval needed' })).toBeTruthy()
    expect(screen.getByText('Approval needed')).toBeTruthy()
  })

  it('does not morph while a pending approval is still in view (at bottom)', () => {
    pendingApproval()
    render(<ScrollToBottomButton sessionId="sess-1" />)

    // Parked at bottom → control hidden, so it can't claim "approval needed".
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('labels uncounted content without zero and follows only its own approval', () => {
    pendingApproval()
    setThreadAtBottom(false, 'sess-1')
    setThreadAtBottom(false, 'tile-runtime')
    const view = render(<ScrollToBottomButton sessionId="tile-runtime" />)
    expect(screen.getByRole('button', { name: 'Scroll to bottom' }).textContent).toBe('Scroll to bottom')

    act(() => setApprovalRequest({ command: 'x', description: 'd', sessionId: 'tile-runtime', requestId: 'r1' }))
    expect(screen.getByRole('button', { name: 'Approval needed' })).toBeTruthy()
    act(() => clearApprovalRequest('tile-runtime', 'r1'))
    expect(screen.queryByText('Approval needed')).toBeNull()
    expect(screen.getByRole('button').textContent).toBe('Scroll to bottom')

    view.rerender(<ScrollToBottomButton sessionId="sess-1" />)
    expect(screen.getByRole('button', { name: 'Approval needed' })).toBeTruthy()
  })

  it('re-arms sticky-bottom on click', () => {
    const handler = vi.fn()
    const stop = onScrollToBottomRequest(handler, 'sess-1')
    setThreadAtBottom(false, 'sess-1')
    render(<ScrollToBottomButton sessionId="sess-1" />)

    fireEvent.click(screen.getByRole('button'))

    expect(handler).toHaveBeenCalledTimes(1)
    stop()
  })

  it('scrolls to the session approval stack instead of the bottom when one is pending', () => {
    pendingApproval()
    setThreadAtBottom(false, 'sess-1')
    const bottomHandler = vi.fn()
    const stopBottom = onScrollToBottomRequest(bottomHandler, 'sess-1')

    const stack = document.createElement('div')
    stack.setAttribute('data-approval-stack', '')
    stack.setAttribute('data-session-id', 'sess-1')
    const scrollIntoView = vi.fn()
    stack.scrollIntoView = scrollIntoView
    document.body.appendChild(stack)

    render(<ScrollToBottomButton sessionId="sess-1" />)
    fireEvent.click(screen.getByRole('button', { name: 'Approval needed' }))

    expect(scrollIntoView).toHaveBeenCalledWith({ block: 'nearest' })
    expect(bottomHandler).not.toHaveBeenCalled()

    stopBottom()
    stack.remove()
  })

  it('does not jump into a sibling session’s approval stack', () => {
    pendingApproval()
    setThreadAtBottom(false, 'sess-1')

    const otherStack = document.createElement('div')
    otherStack.setAttribute('data-approval-stack', '')
    otherStack.setAttribute('data-session-id', 'sess-other')
    const otherScrollIntoView = vi.fn()
    otherStack.scrollIntoView = otherScrollIntoView
    document.body.appendChild(otherStack)

    const bottomHandler = vi.fn()
    const stopBottom = onScrollToBottomRequest(bottomHandler, 'sess-1')

    render(<ScrollToBottomButton sessionId="sess-1" />)
    fireEvent.click(screen.getByRole('button', { name: 'Approval needed' }))

    expect(otherScrollIntoView).not.toHaveBeenCalled()
    // No stack tagged for this session exists, so it falls back to the bottom.
    expect(bottomHandler).toHaveBeenCalledTimes(1)

    stopBottom()
    otherStack.remove()
  })
})
