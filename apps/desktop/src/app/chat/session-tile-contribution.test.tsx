import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it } from 'vitest'

import { registry } from '@/contrib/registry'
import { SESSION_AREAS, type SessionContributionProps } from '@/contrib/session'
import { $sessionTiles } from '@/store/session-states'

import { watchSessionTiles } from './session-tile'

afterEach(() => {
  cleanup()
  $sessionTiles.set([])
})
it('renders each real tile tab contribution against its own stored/runtime binding', () => {
  const dispose = registry.register({
    id: 'tile-fixture',
    area: SESSION_AREAS.tileBadge,
    data: {
      render: ({ session }: SessionContributionProps) => (
        <span>
          {session.connectionId}:{session.runtimeSessionId}
        </span>
      )
    }
  })

  try {
    $sessionTiles.set(
      ['a', 'b'].map(id => ({
        storedSessionId: `stored-${id}`,
        runtimeId: `runtime-${id}`,
        ownerRoute: { connectionId: id, profile: 'default' }
      }))
    )
    watchSessionTiles()
    const tabs = registry.getArea('panes').filter(c => c.id.startsWith('session-tile:'))
    render(
      <>
        {tabs.map(c => (
          <span key={c.id}>{(c.data as { tabLead: () => ReactNode }).tabLead()}</span>
        ))}
      </>
    )
    expect(screen.getByText('a:runtime-a')).toBeTruthy()
    expect(screen.getByText('b:runtime-b')).toBeTruthy()
  } finally {
    dispose()
  }
})
