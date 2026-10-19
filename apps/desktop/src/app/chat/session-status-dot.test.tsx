import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'
import { $backgroundStatusBySession } from '@/store/composer-status'
import { $powerUsage } from '@/store/power-usage'
import { $unreadFinishedSessionIds } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'

import { SessionStatusDot } from './session-status-dot'

afterEach(cleanup)

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        row: {
          backgroundRunning: 'Running in background',
          draftSession: 'Draft',
          finishedUnread: 'Finished',
          needsInput: 'Needs input',
          sessionRunning: 'Running',
          waitingForAnswer: 'Waiting for answer'
        }
      }
    }
  })
}))

const STORED = 's1'
const RUNTIME = 'rt1'

/** The dot a state paints, or null when the row is settled (no `role`). */
const dot = (container: HTMLElement) => container.querySelector('[role="status"]')

const isBreathing = (container: HTMLElement) => dot(container)?.className.includes('status-dot-breathe') ?? false

const working = () => publishSessionState(RUNTIME, { ...createClientSessionState(STORED), busy: true })

// The dotted ids are keyed by runtime id and bridged to the stored id the row
// knows through $sessionStates, so a background claim needs a published session
// to land on — same bridge the real gateway events go through.
const backgroundRunning = () => {
  publishSessionState(RUNTIME, createClientSessionState(STORED))
  $backgroundStatusBySession.set({
    [RUNTIME]: [{ id: 'p1', state: 'running', title: 'sleep 300', type: 'background' }]
  })
}

beforeEach(() => {
  clearAllSessionStates()
  $backgroundStatusBySession.set({})
  $unreadFinishedSessionIds.set([])
  $powerUsage.set('normal')
})

describe('SessionStatusDot activity cue', () => {
  // The arc is the shipped default: a live turn must not also animate the dot,
  // or every existing user gains motion they never asked for.
  it('leaves the dot still while a turn runs at normal power usage', () => {
    working()

    const { container } = render(<SessionStatusDot storedSessionId={STORED} />)

    expect(dot(container)).not.toBeNull()
    expect(isBreathing(container)).toBe(false)
  })

  it('breathes the working dot at low power usage', () => {
    $powerUsage.set('low')
    working()

    const { container } = render(<SessionStatusDot storedSessionId={STORED} />)

    expect(isBreathing(container)).toBe(true)
  })

  // Unread must stay a colour-only state: the breathe means "producing now",
  // and a finished turn that is merely unseen is not producing anything.
  it('never breathes the unread dot', () => {
    $powerUsage.set('low')
    $unreadFinishedSessionIds.set([STORED])

    const { container } = render(<SessionStatusDot storedSessionId={STORED} />)

    expect(dot(container)).not.toBeNull()
    expect(isBreathing(container)).toBe(false)
  })

  // Background work runs for minutes after the turn ends with no arc of its own,
  // so the dot carries the cue in BOTH modes — that is the whole point of it.
  it('breathes the background dot at normal power usage', () => {
    backgroundRunning()

    const { container } = render(<SessionStatusDot storedSessionId={STORED} />)

    expect(isBreathing(container)).toBe(true)
    expect(dot(container)?.className).toContain('ui-purple')
  })

  it('breathes the background dot at low power usage too', () => {
    $powerUsage.set('low')
    backgroundRunning()

    const { container } = render(<SessionStatusDot storedSessionId={STORED} />)

    expect(isBreathing(container)).toBe(true)
  })

  // A live turn outranks background work (one dot, one answer): the working
  // claim wins, and the breathe comes from the power-usage setting, not from the
  // process behind it.
  it('resolves a session that is both working and background-running to working', () => {
    backgroundRunning()
    working()

    const { container } = render(<SessionStatusDot storedSessionId={STORED} />)

    expect(isBreathing(container)).toBe(false)
    expect(dot(container)?.className).toContain('ui-accent')
  })
})
