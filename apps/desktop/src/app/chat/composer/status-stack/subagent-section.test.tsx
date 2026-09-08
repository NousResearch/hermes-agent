import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { ComposerStatusStack } from './index'

vi.stubGlobal('ResizeObserver', class { disconnect() {} observe() {} })

afterEach(() => { cleanup(); $subagentsBySession.set({}) })

it('automatically previews bounded live work only from the composer session, including queued children', () => {
  for (let i = 0; i < 5; i++) {
    upsertSubagent('owner', { subagent_id: `child-${i}`, goal: `Task ${i}`, status: i ? 'queued' : 'running' })
  }

  upsertSubagent('other-profile', { subagent_id: 'foreign', goal: 'Private foreign task' })
  upsertSubagent('owner', { subagent_id: 'child-0', text: 'Reading actual source' }, false, 'subagent.progress')
  const view = render(<MemoryRouter><ComposerStatusStack queue={null} sessionId="owner" /></MemoryRouter>)
  expect(screen.getByText('Task 0')).toBeTruthy()
  expect(screen.getByText('Reading actual source')).toBeTruthy()
  expect(screen.queryByText('Task 4')).toBeNull()
  expect(screen.queryByText('Private foreign task')).toBeNull()
  expect(screen.getByRole('button', { name: /5 Subagents/ })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: /5 Subagents/ }))
  expect(screen.getByText('Task 4')).toBeTruthy()
  view.rerender(<MemoryRouter><ComposerStatusStack queue={null} sessionId="empty" /></MemoryRouter>)
  expect(screen.queryByText('Task 0')).toBeNull()
})

it('retires the live frame only after every child settles, without depending on the parent busy state', () => {
  upsertSubagent('owner', { subagent_id: 'child', goal: 'Live task' })
  render(<MemoryRouter><ComposerStatusStack queue={null} sessionId="owner" /></MemoryRouter>)
  expect(screen.getByText('Live task')).toBeTruthy()
  act(() => upsertSubagent('owner', { subagent_id: 'child', status: 'completed' }, false, 'subagent.complete'))
  expect(screen.queryByText('Live task')).toBeNull()
})
