import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import { onScrollToBottomRequest, resetThreadScroll, setThreadAtBottom, setThreadJumpState } from '@/store/thread-scroll'

import { ScrollToBottomButton } from './scroll-to-bottom-button'

function pendingApproval() {
  $activeSessionId.set('sess-1')
  setApprovalRequest({ command: 'rm -rf /tmp/x', description: 'dangerous command', sessionId: 'sess-1' })
}

afterEach(() => {
  cleanup()
  clearAllPrompts()
  resetThreadScroll()
  $activeSessionId.set(null)
})

// `getByRole('button')` excludes aria-hidden nodes, so "queryByRole null" is the
// control's hidden (parked-at-bottom) state.
describe('ScrollToBottomButton', () => {
  it('stays hidden while parked at the bottom', () => {
    render(<ScrollToBottomButton />)

    expect(screen.queryByRole('button')).toBeNull()
  })

  it('is a plain jump-to-bottom control when scrolled up with no approval', () => {
    setThreadAtBottom(false)
    render(<ScrollToBottomButton />)

    expect(screen.getByRole('button', { name: 'Scroll to bottom' })).toBeTruthy()
    expect(screen.queryByText('Approval needed')).toBeNull()
  })

  it('morphs into the approval pill when scrolled up with a pending approval', () => {
    pendingApproval()
    setThreadAtBottom(false)
    render(<ScrollToBottomButton />)

    expect(screen.getByRole('button', { name: 'Approval needed' })).toBeTruthy()
    expect(screen.getByText('Approval needed')).toBeTruthy()
  })

  it('does not morph while a pending approval is still in view (at bottom)', () => {
    pendingApproval()
    render(<ScrollToBottomButton />)

    // Parked at bottom → control hidden, so it can't claim "approval needed".
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('keeps answer navigation label when an approval is pending but still in view', () => {
    pendingApproval()
    setThreadJumpState({ target: 'answer-start', visible: true })
    render(<ScrollToBottomButton />)

    expect(screen.getByRole('button', { name: 'Answer start' })).toBeTruthy()
    expect(screen.queryByText('Approval needed')).toBeNull()
  })

  it('re-arms sticky-bottom on click', () => {
    const handler = vi.fn()
    const stop = onScrollToBottomRequest(handler)
    setThreadAtBottom(false)
    render(<ScrollToBottomButton />)

    fireEvent.click(screen.getByRole('button'))

    expect(handler).toHaveBeenCalledTimes(1)
    stop()
  })
})
