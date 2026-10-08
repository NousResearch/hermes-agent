import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $sessionColorOverrides, setSessionColorOverride } from '@/store/session-color'
import { $sessionStates } from '@/store/session-states'
import type { ClientSessionState } from '@/app/types'
import type { SessionInfo } from '@/types/hermes'

import { SessionStatusDot } from './session-status-dot'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      sidebar: {
        row: {
          backgroundRunning: 'Background',
          draftSession: 'Draft',
          finishedUnread: 'Unread',
          needsInput: 'Needs input',
          sessionRunning: 'Running',
          waitingForAnswer: 'Waiting'
        }
      }
    }
  })
}))

afterEach(() => {
  cleanup()
  $sessionColorOverrides.set({})
  $sessionStates.set({})
})

const session = { id: '20260922_120000_abcdef' } as unknown as SessionInfo

// The live stores carry the full state object — the draft predicate reads
// `messages`, so a partial one throws inside the derived atoms.
const liveState = (over: Partial<ClientSessionState>): ClientSessionState =>
  ({ messages: [], storedSessionId: session.id, ...over }) as ClientSessionState

function rowDot(container: HTMLElement): HTMLElement {
  const dot = [...container.querySelectorAll('span')].find(span => /(^|\s)size-1(\.5)?(\s|$)/.test(span.className))
  if (!dot) throw new Error('no status dot rendered')
  return dot
}

const paint = (container: HTMLElement) => rowDot(container).getAttribute('style')

describe('SessionStatusDot session colour', () => {
  it('repaints a live dot when the override changes, matching a fresh mount', () => {
    setSessionColorOverride(session.id, 'hsl(0 68% 58%)')
    const { container } = render(<SessionStatusDot session={session} storedSessionId={session.id} />)
    expect(paint(container)).not.toBeNull()

    // Changing the override must reach the row that is already on screen; the
    // colour used to survive only as the value read at mount time.
    act(() => setSessionColorOverride(session.id, 'hsl(180 68% 58%)'))
    const { container: remounted } = render(<SessionStatusDot session={session} storedSessionId={session.id} />)
    const fresh = paint(remounted)
    expect(fresh).not.toBeNull()
    expect(paint(container)).toBe(fresh)
  })

  it('paints nothing when the session has no colour', () => {
    const { container } = render(<SessionStatusDot session={session} storedSessionId={session.id} />)
    expect(paint(container)).toBeNull()
  })

  it('keeps the session colour while the session is working', () => {
    setSessionColorOverride(session.id, 'hsl(0 68% 58%)')
    // `busy` is the authoritative claim $workingSessionIds derives from —
    // setting that derived atom would be a no-op.
    $sessionStates.set({ [session.id]: liveState({ busy: true }) })
    const { container } = render(<SessionStatusDot session={session} storedSessionId={session.id} />)

    // The active dot, not the idle one: a running turn must not hide the colour
    // the user picked for that session. (The DOM normalises hsl() to rgb().)
    expect(rowDot(container).className).toContain('size-1.5')
    expect(paint(container)).toContain('rgb(221, 75, 75)')
  })

  it('leaves the attention colours alone', () => {
    setSessionColorOverride(session.id, 'hsl(0 68% 58%)')
    $sessionStates.set({ [session.id]: liveState({ needsInput: true }) })
    const { container } = render(<SessionStatusDot session={session} storedSessionId={session.id} />)

    // Amber means "answer me" — the tint must never replace it.
    expect(paint(container)).toBeNull()
  })
})
