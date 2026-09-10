import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import * as gateway from '@/store/gateway'
import { _resetSessionOwnerHintsForTests, setSessionOwnerHint } from '@/store/session'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { ComposerStatusStack } from './index'

vi.stubGlobal(
  'ResizeObserver',
  class {
    disconnect() {}
    observe() {}
    unobserve() {}
  }
)
Element.prototype.animate = vi.fn(() => ({ cancel() {} }) as Animation)
afterEach(() => {
  cleanup()
  $subagentsBySession.set({})
  _resetSessionOwnerHintsForTests()
  vi.restoreAllMocks()
})

it('hydrates the empty owner composer and exposes an extended owner-routed transcript without leaking on session change', async () => {
  const text = '  Full transcript\tline with preserved whitespace  \n\n'.repeat(100)

  const request = vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(async (_c, _p, method) => {
    if (method === 'subagent.list') {
      return {
        subagents: [
          { subagent_id: 'worker', goal: 'Recovered work', started_at: 1000, status: 'running', last_tool: 'read_file' }
        ],
        delegations: []
      } as never
    }

    if (method === 'subagent.tail') {
      return { subagent_id: 'worker', available: true, text, truncated: true } as never
    }

    return {} as never
  })

  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'research' })

  const view = render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="parent" />
    </MemoryRouter>
  )

  await screen.findByText('Recovered work')
  expect(screen.getByText('Read File')).toBeTruthy()
  expect(request).toHaveBeenCalledWith('remote-owner', 'research', 'subagent.list', { session_id: 'parent' })
  expect($subagentsBySession.get().parent[0].startedAt).toBe(1000000)
  fireEvent.click(screen.getByRole('button', { name: /Recovered work/ }))
  await waitFor(() => expect(document.querySelector('[data-slot="subagent-transcript"]')?.textContent).toContain(text))
  expect(request).toHaveBeenCalledWith('remote-owner', 'research', 'subagent.tail', {
    session_id: 'parent',
    subagent_id: 'worker'
  })
  view.rerender(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="other" />
    </MemoryRouter>
  )
  expect(document.querySelector('[data-slot="subagent-transcript"]')).toBeNull()
})

it('does not resurrect a child completed while the roster snapshot was in flight', async () => {
  let resolve!: (value: unknown) => void
  vi.spyOn(gateway, 'requestGatewayForAgent').mockImplementation(async (_c, _p, method) => {
    if (method === 'subagent.list') {
      return (await new Promise<unknown>(r => {
        resolve = r
      })) as never
    }

    return {} as never
  })
  setSessionOwnerHint('parent', { connectionId: 'remote-owner', profile: 'research' })
  upsertSubagent('parent', { subagent_id: 'worker', goal: 'Finishing work' })
  render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="parent" />
    </MemoryRouter>
  )
  await waitFor(() => expect(resolve).toBeTypeOf('function'))
  act(() => upsertSubagent('parent', { subagent_id: 'worker', status: 'completed' }, false, 'subagent.complete'))
  await act(async () =>
    resolve({ subagents: [{ subagent_id: 'worker', goal: 'Finishing work', status: 'running' }], delegations: [] })
  )
  expect(screen.queryByText('Finishing work')).toBeNull()
  expect($subagentsBySession.get().parent[0].status).toBe('completed')
})
