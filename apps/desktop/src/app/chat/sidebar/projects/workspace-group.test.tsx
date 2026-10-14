import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { SidebarProvider } from '@/components/ui/sidebar'
import type { SessionInfo } from '@/hermes'

import { SidebarWorkspaceGroup } from './workspace-group'
import type { SidebarSessionGroup } from './workspace-groups'

afterEach(cleanup)

const branchLane = (over: Partial<SidebarSessionGroup> = {}): SidebarSessionGroup => ({
  id: '/repo::branch::dev',
  label: 'dev',
  path: '/repo',
  sessions: [],
  ...over
})

const mount = (over: Partial<SidebarSessionGroup>, props: Record<string, unknown> = {}) =>
  render(
    <MemoryRouter>
      <SidebarProvider>
        <SidebarWorkspaceGroup
          group={branchLane(over)}
          onNewSession={vi.fn()}
          renderRows={(sessions: SessionInfo[]) => sessions.map(session => <div key={session.id}>{session.id}</div>)}
          {...props}
        />
      </SidebarProvider>
    </MemoryRouter>
  )

describe('SidebarWorkspaceGroup "+" (#138445)', () => {
  it('seeds the new session at newSessionPath when the caller overrides the lane path', async () => {
    const onNewSession = vi.fn()
    mount({}, { newSessionPath: '/repo/engagement', onNewSession })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /New session in dev/ }))
    })
    expect(onNewSession).toHaveBeenCalledWith('/repo/engagement')
  })

  it('keeps the lane path when no override is given', async () => {
    const onNewSession = vi.fn()
    mount({}, { onNewSession })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /New session in dev/ }))
    })
    expect(onNewSession).toHaveBeenCalledWith('/repo')
  })
})
