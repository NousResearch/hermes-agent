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

  it('renders nothing when idle', () => {
    const { container } = render(<ComputerUseStatusPill />)
    expect(container.querySelector('[role="status"]')).toBeNull()
  })

  it('renders drafting state with status role', () => {
    act(() => {
      setComputerUseDrafting(SID)
    })

    render(<ComputerUseStatusPill />)
    const pill = screen.getByRole('status')
    expect(pill).not.toBeNull()
    expect(pill.getAttribute('data-computer-use-phase')).toBe('drafting')
    expect(screen.getByText('Computer Use')).not.toBeNull()
    expect(screen.getByText('Preparing…')).not.toBeNull()
  })

  it('renders running state with app and action', () => {
    act(() => {
      setComputerUseRunning(SID, {
        action: 'click',
        app: 'Google Chrome',
        element: 14
      })
    })

    render(<ComputerUseStatusPill />)
    const pill = screen.getByRole('status')
    expect(pill).not.toBeNull()
    expect(pill.getAttribute('data-computer-use-phase')).toBe('running')
    expect(screen.getByText('Google Chrome · Click #14')).not.toBeNull()
  })

  it('renders completed state with checkmark indicator and duration', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'capture', app: 'Notepad', mode: 'som' })
      setComputerUseCompleted(SID, { durationSeconds: 0.8 })
    })

    render(<ComputerUseStatusPill />)
    const pill = screen.getByRole('status')
    expect(pill).not.toBeNull()
    expect(pill.getAttribute('data-computer-use-phase')).toBe('completed')
    expect(screen.getByText(/Notepad · SOM Capture/)).not.toBeNull()
    expect(screen.getByText(/\(0.8s\)/)).not.toBeNull()
  })

  it('renders error state with error message and allows click to dismiss', () => {
    act(() => {
      setComputerUseRunning(SID, { action: 'click', app: 'Explorer' })
      setComputerUseError(SID, 'Target window minimized')
    })

    render(<ComputerUseStatusPill />)
    const pill = screen.getByRole('status')
    expect(pill).not.toBeNull()
    expect(pill.getAttribute('data-computer-use-phase')).toBe('error')
    expect(screen.getByText('Target window minimized')).not.toBeNull()

    // Clicking error dismisses the state
    act(() => {
      fireEvent.click(pill)
    })
    expect($computerUseBySession.get()[SID]).toBeUndefined()
  })

  it('binds to specific sessionId when prop is provided', () => {
    const OTHER_SID = 'test-session-2'
    act(() => {
      setComputerUseRunning(OTHER_SID, { action: 'scroll', app: 'VS Code' })
    })

    // Active session is SID, so without prop nothing is rendered
    const { rerender } = render(<ComputerUseStatusPill />)
    expect(screen.queryByRole('status')).toBeNull()

    // With OTHER_SID prop, it renders OTHER_SID's state
    rerender(<ComputerUseStatusPill sessionId={OTHER_SID} />)
    const pill = screen.getByRole('status')
    expect(pill).not.toBeNull()
    expect(screen.getByText('VS Code · Scroll')).not.toBeNull()
  })
})
