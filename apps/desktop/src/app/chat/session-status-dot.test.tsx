import { act, cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SessionStatusDot } from './session-status-dot'

afterEach(() => {
  cleanup()
  vi.resetModules()
})

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

vi.mock('@/store/session', () => ({
  $sessions: atom([]),
  sessionPinId: (session: Pick<SessionInfo, '_lineage_root_id' | 'id'>) => session._lineage_root_id ?? session.id
}))

vi.mock('@/store/projects', () => ({
  $projectOwnerBySessionId: atom(new Map()),
  $projects: atom([])
}))

vi.mock('@/store/session-dot-state', () => ({
  $sessionDotStateById: atom({})
}))

const session = { _lineage_root_id: null, id: 's1', title: 'One' } as unknown as SessionInfo

function idleDot(container: HTMLElement): HTMLElement {
  const dot = container.querySelector('span[aria-hidden="true"].size-1')

  if (!dot) {
    throw new Error('idle dot not rendered')
  }

  return dot as HTMLElement
}

describe('SessionStatusDot', () => {
  it('repaints the idle dot when the session color override changes', async () => {
    const { $sessionColorOverrides, setSessionColorOverride } = await import('@/store/session-color')

    $sessionColorOverrides.set({})

    const { container } = render(<SessionStatusDot session={session} storedSessionId="s1" />)

    expect(idleDot(container).style.backgroundColor).toBe('')

    act(() => setSessionColorOverride('s1', 'rgb(255, 0, 0)'))

    expect(idleDot(container).style.backgroundColor).toBe('rgb(255, 0, 0)')
  })
})
