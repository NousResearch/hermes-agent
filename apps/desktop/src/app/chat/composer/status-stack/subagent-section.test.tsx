import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { ComposerStatusStack } from './index'

vi.mock('@/lib/use-enter-animation', () => ({ useEnterAnimation: () => undefined }))

vi.stubGlobal(
  'ResizeObserver',
  class {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
)

afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
})

it('shows live work only from the composer session and keeps it hidden after collapse and progress', () => {
  for (let i = 0; i < 5; i++) {
    upsertSubagent('owner', { subagent_id: `child-${i}`, goal: `Task ${i}`, status: i ? 'queued' : 'running' })
  }

  upsertSubagent('other-profile', { subagent_id: 'foreign', goal: 'Private foreign task' })
  upsertSubagent('owner', { subagent_id: 'child-0', text: 'Reading actual source' }, false, 'subagent.progress')

  const view = render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="owner" />
    </MemoryRouter>
  )

  expect(screen.queryByText('Task 0')).toBeNull()
  expect(screen.queryByText('Private foreign task')).toBeNull()
  const header = screen.getByRole('button', { name: /5 Subagents/ })
  fireEvent.click(header)
  expect(screen.getByText('Task 0')).toBeTruthy()
  expect(screen.getByText('Reading actual source')).toBeTruthy()
  fireEvent.click(header)
  expect(screen.queryByText('Task 0')).toBeNull()
  expect(screen.queryByText('Task 4')).toBeNull()
  expect(header.getAttribute('aria-expanded')).toBe('false')
  act(() => upsertSubagent('owner', { subagent_id: 'child-0', text: 'More progress' }, false, 'subagent.progress'))
  expect(screen.queryByText('Task 0')).toBeNull()
  fireEvent.click(header)
  expect(screen.getByText('Task 0')).toBeTruthy()
  expect(screen.getByText('Task 4')).toBeTruthy()
  expect(screen.getByText('More progress')).toBeTruthy()
  view.rerender(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="empty" />
    </MemoryRouter>
  )
  expect(screen.queryByText('Task 0')).toBeNull()
})

it('collapses a single worker and its selected detail using the caret, preserving the steering draft', () => {
  upsertSubagent('owner', { subagent_id: 'child', goal: 'Single task' })

  const view = render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="owner" />
    </MemoryRouter>
  )

  fireEvent.click(screen.getByRole('button', { name: /1 Subagent/ }))
  fireEvent.click(screen.getByRole('button', { name: /Single task/ }))
  expect(view.container.querySelector('[data-slot="composer-subagent-detail"]')).toBeTruthy()
  const draft = screen.getByRole('textbox')
  fireEvent.change(draft, { target: { value: 'Keep this draft' } })
  const header = screen.getByRole('button', { name: /1 Subagent/ })
  fireEvent.click(header.firstElementChild!)
  expect(screen.queryByText('Single task')).toBeNull()
  expect(view.container.querySelector('[data-slot="composer-subagent-detail"]')).toBeNull()
  expect(header.getAttribute('aria-expanded')).toBe('false')
  fireEvent.click(header)
  expect(view.container.querySelector('[data-slot="composer-subagent-detail"]')).toBeTruthy()
  expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('Keep this draft')
})

it('retires the live frame only after every child settles, without depending on the parent busy state', () => {
  upsertSubagent('owner', { subagent_id: 'child', goal: 'Live task' })
  render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="owner" />
    </MemoryRouter>
  )
  fireEvent.click(screen.getByRole('button', { name: /1 Subagent/ }))
  expect(screen.getByText('Live task')).toBeTruthy()
  act(() => upsertSubagent('owner', { subagent_id: 'child', status: 'completed' }, false, 'subagent.complete'))
  expect(screen.queryByText('Live task')).toBeNull()
})
