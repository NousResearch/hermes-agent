// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import {
  $computerUseBySession,
  clearAllComputerUseStates,
  setComputerUseCompleted,
  setComputerUseDrafting,
  setComputerUseError,
  setComputerUseRunning
} from '@/store/computer-use'
import { $activeSessionId } from '@/store/session'

import { ComputerUseStatusPill } from './computer-use-pill'

describe('ComputerUseStatusPill', () => {
  const SID = 'test-session-1'

  beforeEach(() => {
    clearAllComputerUseStates()
    $activeSessionId.set(SID)
  })

  afterEach(() => {
    cleanup()
    clearAllComputerUseStates()
  })

  it('renders nothing without a computer_use call', () => {
    const { container } = render(<ComputerUseStatusPill />)
    expect(container.querySelector('[role="status"]')).toBeNull()
  })

  it('renders drafting state with status role', () => {
    act(() => {
      setComputerUseDrafting(SID)
    })

    render(<ComputerUseStatusPill />)
    const pill = screen.getByRole('status')
    expect(pill.getAttribute('data-computer-use-phase')).toBe('drafting')
    expect(screen.getByText('Computer Use')).not.toBeNull()
    expect(screen.getByText('Preparing…')).not.toBeNull()
  })

  it('renders running state with app and action', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'click', app: 'Google Chrome', element: 14 })
    })

    render(<ComputerUseStatusPill />)
    expect(screen.getByRole('status').getAttribute('data-computer-use-phase')).toBe('running')
    expect(screen.getByText('Google Chrome · Click #14')).not.toBeNull()
  })

  it('shows the length of typed text, not the text', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'type', app: 'Notepad', textLength: 8 })
    })

    render(<ComputerUseStatusPill />)
    expect(screen.getByText('Notepad · Type (8 characters)')).not.toBeNull()
  })

  it('renders completed state with checkmark indicator and duration', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'capture', app: 'Notepad', mode: 'som' })
      setComputerUseCompleted(SID, { durationSeconds: 0.8 })
    })

    render(<ComputerUseStatusPill />)
    expect(screen.getByRole('status').getAttribute('data-computer-use-phase')).toBe('completed')
    expect(screen.getByText('Notepad · Capture (SOM) (0.8s)')).not.toBeNull()
  })

  it('renders the error and dismisses it from a keyboard-reachable button', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'click', app: 'Explorer' })
      setComputerUseError(SID, 'Target window minimized')
    })

    render(<ComputerUseStatusPill />)
    expect(screen.getByRole('status').getAttribute('data-computer-use-phase')).toBe('error')
    expect(screen.getByText('Target window minimized')).not.toBeNull()

    const dismiss = screen.getByRole('button', { name: 'Dismiss' })
    expect(dismiss.getAttribute('type')).toBe('button')

    act(() => {
      fireEvent.click(dismiss)
    })
    expect($computerUseBySession.get()[SID]).toBeUndefined()
  })

  it('falls back to a localized failure label when the tool gave no reason', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'click' })
      setComputerUseError(SID)
    })

    render(<ComputerUseStatusPill />)
    expect(screen.getByText('Action failed')).not.toBeNull()
  })

  it('offers no dismiss button while the call is running', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'click', app: 'Explorer' })
    })

    render(<ComputerUseStatusPill />)
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('binds to specific sessionId when prop is provided', () => {
    const OTHER_SID = 'test-session-2'
    act(() => {
      setComputerUseRunning(OTHER_SID, { action: 'scroll', app: 'VS Code' })
    })

    const { rerender } = render(<ComputerUseStatusPill />)
    expect(screen.queryByRole('status')).toBeNull()

    rerender(<ComputerUseStatusPill sessionId={OTHER_SID} />)
    expect(screen.getByRole('status')).not.toBeNull()
    expect(screen.getByText('VS Code · Scroll')).not.toBeNull()
  })
})
