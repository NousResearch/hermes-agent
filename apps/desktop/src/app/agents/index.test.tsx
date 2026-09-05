import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { registry } from '@/contrib/registry'
import { $routeTiles, closeRouteTile } from '@/store/route-tiles'

import { watchRouteTiles } from '../chat/route-tile'

import { overviewActions } from './actions'

import { AgentsView } from './index'

it('opens the same Agents route as a native pane tab, not a second floating overlay', async () => {
  window.hermesDesktop = {
    ...window.hermesDesktop,
    getAgentOverview: vi.fn(async () => data)
  } as typeof window.hermesDesktop
  watchRouteTiles()
  const close = vi.fn()
  const overlay = render(<AgentsView onClose={close} />)
  fireEvent.click(screen.getByRole('button', { name: 'Open as tab' }))
  expect($routeTiles.get()).toContainEqual({ path: '/agents', dir: 'right' })
  expect(close).toHaveBeenCalledOnce()
  overlay.unmount()
  const pane = registry.getArea('panes').find(pane => pane.id === 'route-tile:/agents')!
  const view = render(<PaneVisibleContext.Provider value={true}>{pane.render!()}</PaneVisibleContext.Provider>)
  await screen.findByTestId('agent-overview')
  expect(view.container.querySelector('[data-overlay-surface]')).toBeNull()
  expect(screen.queryByRole('button', { name: 'Open as tab' })).toBeNull()
  view.rerender(<PaneVisibleContext.Provider value={false}>{pane.render!()}</PaneVisibleContext.Provider>)
  view.unmount()
  closeRouteTile('/agents')
})
import type { AgentOverview } from '../../..//electron/agent-overview'

const data: AgentOverview = {
  fetchedAt: 1,
  sources: [
    {
      connectionId: 'remote',
      label: 'Lab',
      kind: 'remote',
      state: 'ready',
      sessions: [
        {
          id: 'one',
          profile: 'coder',
          title: 'Build rocket',
          preview: 'Checking engines',
          model: 'any-model',
          provider: 'any-provider',
          last_active: Date.now() / 1000,
          unread: true
        }
      ],
      canonical: [],
      live: [],
      profiles: [{ name: 'coder', display_name: 'Engineer', description: 'Build spacecraft' }],
      total: 1,
      offset: 1,
      complete: true,
      liveCoverage: 'process',
      errors: []
    }
  ]
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
it('sends a real owner-bound reply without changing the foreground composer', async () => {
  window.hermesDesktop = {
    ...window.hermesDesktop,
    getAgentOverview: vi.fn(async () => data)
  } as typeof window.hermesDesktop
  const reply = vi.spyOn(overviewActions, 'reply').mockResolvedValue()
  render(<AgentsView onClose={vi.fn()} />)
  await waitFor(() => expect(screen.getAllByTestId('agent-row')).toHaveLength(1))
  fireEvent.click(screen.getByRole('button', { name: /Build rocket/ }))
  fireEvent.change(screen.getByRole('textbox', { name: 'Reply' }), { target: { value: 'Proceed carefully' } })
  fireEvent.click(screen.getByRole('button', { name: 'Reply' }))
  await waitFor(() =>
    expect(reply).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'remote', profile: 'coder', id: 'one' }),
      'Proceed carefully'
    )
  )
  await waitFor(() => expect((screen.getByRole('textbox', { name: 'Reply' }) as HTMLTextAreaElement).value).toBe(''))
})
it('offers a real Connect action for parked registry sources', async () => {
  window.hermesDesktop = {
    ...window.hermesDesktop,
    getAgentOverview: vi.fn(async () => ({
      fetchedAt: 1,
      sources: [
        {
          ...data.sources[0]!,
          connectionId: 'parked-ssh',
          label: 'Parked lab',
          sessions: [],
          state: 'on-demand',
          complete: false
        }
      ]
    }))
  } as typeof window.hermesDesktop
  const connect = vi.spyOn(overviewActions, 'connect').mockResolvedValue()
  render(<AgentsView onClose={vi.fn()} />)
  const button = await screen.findByRole('button', { name: 'Connect Parked lab' })
  fireEvent.click(button)
  await waitFor(() => expect(connect).toHaveBeenCalledWith('parked-ssh'))
})
it('defaults to a searchable Sessions overview, previews without navigation and preserves the foreground draft/focus across close', async () => {
  window.hermesDesktop = {
    ...window.hermesDesktop,
    getAgentOverview: vi.fn(async () => data)
  } as typeof window.hermesDesktop
  const onClose = vi.fn()
  const foreground = globalThis.document.createElement('textarea')
  foreground.value = 'keep my draft'
  globalThis.document.body.append(foreground)
  foreground.focus()
  const view = render(<AgentsView onClose={onClose} />)
  // The subtitle is the live headline, not a description of the feature.
  expect(screen.getAllByText('All quiet').length).toBeGreaterThan(0)
  expect(screen.getByRole('button', { name: 'Sessions' }).getAttribute('aria-pressed')).toBe('true')
  await waitFor(() => expect(screen.getAllByTestId('agent-row')).toHaveLength(1))
  fireEvent.click(screen.getByRole('button', { name: /Build rocket/ }))
  expect(screen.getAllByText('Checking engines')).toHaveLength(2)
  expect(screen.getByRole('button', { name: 'Open conversation' })).toBeTruthy()
  fireEvent.change(screen.getByRole('textbox', { name: 'Search sessions' }), { target: { value: 'missing' } })
  expect(screen.queryAllByTestId('agent-row')).toHaveLength(0)
  fireEvent.click(screen.getByRole('button', { name: 'Spawn tree' }))
  expect(screen.getByText('No live subagents')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: 'Close agents' }))
  expect(onClose).toHaveBeenCalledOnce()
  view.unmount()
  await act(async () => {})
  expect(foreground.value).toBe('keep my draft')
  expect(globalThis.document.activeElement).toBe(foreground)
  foreground.remove()
})
it('headlines live attention counts and opens a row on double-click through the exact owner', async () => {
  window.hermesDesktop = {
    ...window.hermesDesktop,
    getAgentOverview: vi.fn(async () => ({
      ...data,
      sources: [
        {
          ...data.sources[0]!,
          live: [
            { id: 'two', profile: 'coder', title: 'Deploy', status: 'working', runtime_id: 'r2' },
            { id: 'three', profile: 'coder', title: 'Ask', status: 'waiting', runtime_id: 'r3' },
            { id: 'four', profile: 'coder', title: 'Ask too', status: 'waiting', runtime_id: 'r4' }
          ]
        }
      ]
    }))
  } as typeof window.hermesDesktop
  const open = vi.spyOn(overviewActions, 'open').mockResolvedValue()
  render(<AgentsView onClose={vi.fn()} />)
  await waitFor(() => expect(screen.getByText('2 need you · 1 working')).toBeTruthy())
  fireEvent.doubleClick(screen.getByRole('button', { name: /Deploy/ }))
  await waitFor(() =>
    expect(open).toHaveBeenCalledWith(
      expect.objectContaining({ connectionId: 'remote', profile: 'coder', id: 'two' }),
      expect.any(Function)
    )
  )
})
