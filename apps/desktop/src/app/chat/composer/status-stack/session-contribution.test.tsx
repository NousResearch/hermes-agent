import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { SESSION_AREAS, type SessionContributionProps } from '@/contrib/session'
import { $sessionTiles } from '@/store/session-states'

import { ComposerStatusStack } from './index'

const disposers: Array<() => void> = []
afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(d => d())
  $sessionTiles.set([])
})
it('mounts a plugin status row in each owning stack with its own action context', () => {
  $sessionTiles.set(
    ['a', 'b'].map(id => ({
      storedSessionId: `stored-${id}`,
      runtimeId: `runtime-${id}`,
      ownerRoute: { connectionId: id, profile: id }
    }))
  )
  const action = vi.fn()
  const ctx = createPluginContext('status-fixture', d => disposers.push(d))
  ctx.register({
    id: 'row',
    area: SESSION_AREAS.statusStack,
    data: {
      render: ({ session }: SessionContributionProps) => (
        <button onClick={() => action(session)}>{session.profile} live</button>
      )
    }
  })
  render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="runtime-a" />
      <ComposerStatusStack queue={null} sessionId="runtime-b" />
    </MemoryRouter>
  )
  fireEvent.click(screen.getByText('b live'))
  expect(action).toHaveBeenCalledWith({
    runtimeSessionId: 'runtime-b',
    storedSessionId: 'stored-b',
    connectionId: 'b',
    profile: 'b'
  })
  expect(screen.getByText('a live').closest('[data-slot="composer-status-stack"]')).toBeTruthy()
})
