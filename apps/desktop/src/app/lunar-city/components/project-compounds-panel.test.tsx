// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { entityKey } from '../identity'
import type { LunarCitySnapshot, LunarEntity } from '../model'
import { createProjectCompoundController } from '../world/project-compounds'

import { ProjectCompoundsPanel } from './project-compounds-panel'

afterEach(cleanup)
it('keeps session-only overflow projects inspectable by exact owner identity', () => {
  const entities = ['first', 'second'].map(connectionId => {
    const identity = { kind: 'session', connectionId, profile: 'p', sessionId: 'same-session' } as const

    return { identity, key: entityKey(identity), projectId: '/same-repo', authority: 'authoritative', destination: 'project', animation: 'work', sourceState: 'running', observedAt: 1 } satisfies LunarEntity
  })

  const snapshot: LunarCitySnapshot = { revision: 1, observedAt: 1, sources: [], entities: new Map(entities.map(entity => [entity.key, entity])) }
  const report = createProjectCompoundController([]).update(snapshot)
  const select = vi.fn()
  render(<ProjectCompoundsPanel onSelectEntity={select} report={report} snapshot={snapshot} />)
  expect(screen.getAllByText('Overflow · no physical site available')).toHaveLength(2)
  fireEvent.click(screen.getByRole('button', { name: 'Inspect session same-session · p · second' }))
  expect(select).toHaveBeenCalledWith(entities[1])
})
