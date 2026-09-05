import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider, TRANSLATIONS } from '@/i18n'
import { LOCALE_OPTIONS } from '@/i18n/languages'
import { createClientSessionState } from '@/lib/chat-runtime'
import { setSessions } from '@/store/session'
import { clearAllSessionStates, publishSessionState } from '@/store/session-states'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'
import type { SessionInfo } from '@/types/hermes'

import { sessionDotClassName, SessionStatusDot } from './session-status-dot'

const session = {
  id: 'stored-1',
  message_count: 1,
  source: 'cli',
  started_at: 0,
  title: 'Parent session'
} as SessionInfo

afterEach(() => {
  cleanup()
  clearAllSessionStates()
  $subagentsBySession.set({})
  setSessions([])
})

describe('SessionStatusDot subagent state', () => {
  it('uses the same filled activity mark for background processes and subagents', () => {
    expect(sessionDotClassName('background')).toBe(sessionDotClassName('subagents'))
  })

  it('renders the violet dot with a localized plural count, then the singular count', () => {
    setSessions([session])
    publishSessionState('runtime-1', { ...createClientSessionState('stored-1'), busy: false })
    upsertSubagent('runtime-1', { goal: 'first', status: 'running', subagent_id: 'a', task_index: 0 })
    upsertSubagent('runtime-1', { goal: 'second', status: 'queued', subagent_id: 'b', task_index: 1 })

    render(
      <I18nProvider configClient={null} initialLocale="en">
        <SessionStatusDot session={session} storedSessionId="stored-1" />
      </I18nProvider>
    )

    const plural = screen.getByRole('status', { name: '2 subagents active' })
    expect(plural.getAttribute('title')).toBe('2 subagents active')
    expect(plural.classList.contains('bg-(--ui-purple)')).toBe(true)

    act(() => {
      upsertSubagent('runtime-1', { status: 'completed', subagent_id: 'a', task_index: 0 }, false, 'subagent.complete')
    })

    const singular = screen.getByRole('status', { name: '1 subagent active' })
    expect(singular.getAttribute('title')).toBe('1 subagent active')
  })

  it('provides count-aware copy in every supported locale', () => {
    for (const { id } of LOCALE_OPTIONS) {
      const activeSubagents = TRANSLATIONS[id].sidebar.row.activeSubagents

      expect(activeSubagents(1), id).not.toBe('')
      expect(activeSubagents(2), id).not.toBe('')
      expect(activeSubagents(1), id).not.toBe(activeSubagents(2))
    }
  })
})
