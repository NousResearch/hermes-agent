// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'

import { entityKey } from '../identity'
import type { LunarCitySnapshot, LunarEntity } from '../model'

import { AlertsPanel } from './alerts-panel'

it('inspects the exact owner and acknowledges locally without resolving disappearing incidents', () => {
  const identity = { connectionId: 'remote', kind: 'profile' as const, profile: 'worker' }

  const entity: LunarEntity = {
    identity,
    key: entityKey(identity),
    authority: 'authoritative',
    animation: 'blocked',
    sourceState: 'blocked',
    destination: 'triage',
    observedAt: 1
  }

  const snapshot: LunarCitySnapshot = {
    revision: 1,
    observedAt: 1,
    entities: new Map([[entity.key, entity]]),
    sources: []
  }

  const onSelectEntity = vi.fn()
  const { rerender } = render(<AlertsPanel onSelectEntity={onSelectEntity} snapshot={snapshot} />)
  fireEvent.click(screen.getByRole('button', { name: 'Inspect worker · remote' }))
  expect(onSelectEntity).toHaveBeenCalledWith(entity)
  fireEvent.click(screen.getByRole('button', { name: 'Acknowledge worker · remote' }))
  expect(screen.getByRole('status').textContent).toContain('0 unacknowledged')
  expect(entity.sourceState).toBe('blocked')
  rerender(<AlertsPanel onSelectEntity={onSelectEntity} snapshot={{ ...snapshot, revision: 2, entities: new Map() }} />)
  expect(screen.getByText(/Entity currently unavailable/)).toBeTruthy()
  expect(screen.getByRole('status').textContent).toContain('1 attention items')
  rerender(
    <AlertsPanel
      onSelectEntity={onSelectEntity}
      snapshot={{
        ...snapshot,
        revision: 3,
        entities: new Map([[entity.key, { ...entity, sourceState: 'completed', observedAt: 2 }]])
      }}
    />
  )
  expect(screen.getByRole('status').textContent).toContain('0 attention items')
  fireEvent.click(screen.getByRole('checkbox', { name: 'Show resolved history' }))
  expect(screen.getByText(/· resolved/)).toBeTruthy()
})
