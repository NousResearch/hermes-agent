import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $backgroundStatusBySession } from '@/store/composer-status'
import { $sessions } from '@/store/session'
import { $sessionStates } from '@/store/session-states'
import { resetSessionTodoOverview } from '@/store/session-todos-overview'
import { $todosBySession } from '@/store/todos'

import { AgentsPanelContent } from './index'

/** Regression: a background process running in a session the user isn't
 *  currently viewing must still surface in the Agents panel — previously
 *  $backgroundStatusBySession only ever reached the composer's own status
 *  stack for the ONE focused session tab. */
describe('Agents panel background processes section', () => {
  beforeEach(() => {
    resetSessionTodoOverview()
    $todosBySession.set({})
    $sessionStates.set({})
    $sessions.set([])
    $backgroundStatusBySession.set({})
  })

  afterEach(() => {
    cleanup()
    resetSessionTodoOverview()
    $todosBySession.set({})
    $backgroundStatusBySession.set({})
  })

  it('renders a running background process from an unfocused session', () => {
    act(() => {
      $backgroundStatusBySession.set({
        'runtime-unfocused': [{ id: 'proc-1', state: 'running', title: 'npm run dev', type: 'background' }]
      })
    })

    render(<AgentsPanelContent />)

    expect(screen.queryByText('npm run dev')).toBeTruthy()
  })

  it('renders nothing for the background section when there is no background work', () => {
    render(<AgentsPanelContent />)

    expect(screen.queryByText('npm run dev')).toBeNull()
  })
})
