import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'
import { $subagentsBySession, type SubagentOutcome, type SubagentProgress } from '@/store/subagents'

import { AgentsView } from './index'

vi.mock('@/lib/use-enter-animation', () => ({ useEnterAnimation: () => () => undefined }))

const item = (outcome: SubagentOutcome): SubagentProgress => ({
  filesRead: [],
  filesWritten: [],
  goal: 'Research Cursor',
  id: `child-${outcome}`,
  outcome,
  parentId: null,
  startedAt: 1,
  status: 'completed',
  stream: [{ at: 1, kind: 'summary', outcome, text: `${outcome} result` }],
  taskCount: 1,
  taskIndex: 0,
  updatedAt: 1
})

function renderAgents(outcome: SubagentOutcome) {
  $subagentsBySession.set({ session: [item(outcome)] })
  render(<AgentsView onClose={() => undefined} />)
}

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
})

describe('AgentsView outcome status', () => {
  it('labels a partial result as partial instead of merely unverified', () => {
    renderAgents('partial')

    expect(screen.getAllByLabelText(en.assistant.tool.statusPartial)).toHaveLength(2)
    expect(screen.queryByLabelText(en.agents.verificationRequired)).toBeNull()
  })

  it('keeps an unverified result behind the verification-required state', () => {
    renderAgents('unverified')

    expect(screen.getAllByLabelText(en.agents.verificationRequired)).toHaveLength(2)
    expect(screen.queryByLabelText(en.assistant.tool.statusPartial)).toBeNull()
  })
})
