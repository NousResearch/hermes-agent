// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { setSidebarGrouping } from '@/store/layout'
import { $profiles } from '@/store/profile'
import { $sessions } from '@/store/session'
import { makeSessionInfo } from '@/test/session-info'

import { $gatewayGroupOrder } from './gateway-group-preferences'

import { ChatSidebar } from './index'

// Gateway/profile groups reorder by drag as well as by the ⋯ menu's Move
// up/down. The grab handle only reveals itself on hover, so the visible
// affordance is the header itself: a press on the label must arm the dnd
// sortable (the lead handle already did). Proven red on the version where only
// the handle carried the listeners.

const noop = () => {}

const mount = () =>
  render(
    <MemoryRouter>
      <SidebarProvider>
        <ChatSidebar
          currentView="chat"
          onArchiveSession={noop}
          onBranchSession={noop}
          onDeleteSession={noop}
          onLoadMoreSessions={noop}
          onManageCronJob={noop}
          onNavigate={noop}
          onNewSessionInWorkspace={noop}
          onNewSessionSplit={noop}
          onResumeSession={vi.fn()}
          onTriggerCronJob={async () => {}}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

afterEach(() => {
  cleanup()
  $gatewayGroupOrder.set([])
})

it('reorders gateway sections by dragging the header label, not only the grab handle', async () => {
  mount()
  act(() => {
    $connectionsRegistry.set({
      version: 2,
      primary: 'local',
      secureTokenStorage: true,
      connections: [
        { id: 'local', label: 'This device', kind: 'local', tokenSet: false, tokenPreview: null },
        { id: 'remote-1', label: 'Homelab', kind: 'remote', tokenSet: false, tokenPreview: null }
      ]
    })
    $profiles.set([{ name: 'default', is_default: true }] as typeof $profiles.value)
    setSidebarGrouping('profile')
    $sessions.set(
      ['local', 'remote-1'].map(connection_id =>
        makeSessionInfo({ id: connection_id, connection_id, profile: 'default', last_active: Date.now() / 1000 })
      )
    )
  })

  const sectionIds = () =>
    [...document.querySelectorAll('[data-gateway-section]')].map(node => node.getAttribute('data-gateway-section'))

  const device = screen.getByText('This device').closest('[data-gateway-section]') as HTMLElement
  expect(sectionIds()).toEqual([JSON.stringify(['gateway', 'remote-1']), JSON.stringify(['gateway', 'local'])])

  // dnd-kit's KeyboardSensor: Space on the focused sortable arms the drag,
  // ArrowUp moves it over the previous item, Space drops. Drives the same
  // listeners the pointer path binds, without needing layout in jsdom.
  // The fold link, not the row (which dnd-kit also names as a sortable button).
  const label = within(device).getByRole('button', { expanded: true, name: 'This device' })
  label.focus()
  await act(async () => {
    fireEvent.keyDown(label, { code: 'Space', key: ' ' })
    await Promise.resolve()
  })
  await act(async () => {
    fireEvent.keyDown(label, { code: 'ArrowUp', key: 'ArrowUp' })
    await Promise.resolve()
  })
  await act(async () => {
    fireEvent.keyDown(label, { code: 'Space', key: ' ' })
    await Promise.resolve()
  })

  expect($gatewayGroupOrder.get()[0]).toBe(JSON.stringify(['gateway', 'local']))
  expect(sectionIds()[0]).toBe(JSON.stringify(['gateway', 'local']))
})
