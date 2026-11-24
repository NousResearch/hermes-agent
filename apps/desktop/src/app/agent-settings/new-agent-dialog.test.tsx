import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'

import type { AgentDialogTarget } from './agent-dialogs'
import NewAgentDialog from './new-agent-dialog'

const opened = vi.hoisted(() => ({ target: null as AgentDialogTarget | null }))

vi.mock('./agent-dialogs', () => ({
  AgentDialog: ({ target }: { target: AgentDialogTarget }) => {
    opened.target = target

    return null
  }
}))

afterEach(() => {
  cleanup()
  $apiRequestScope.set({ connectionId: null, profile: null })
})

it('makes the agent on the machine it was opened for, not the window’s own', () => {
  $apiRequestScope.set({ connectionId: 'studio', profile: 'default' })

  render(<NewAgentDialog onClose={() => {}} owner={{ connectionId: 'lab-box', profile: 'ops' }} />)
  expect(opened.target).toEqual({ kind: 'create', owner: { connectionId: 'lab-box', profile: 'ops' } })

  cleanup()
  render(<NewAgentDialog onClose={() => {}} />)
  expect(opened.target?.owner.connectionId).toBe('studio')
})
