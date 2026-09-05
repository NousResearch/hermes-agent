import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { DispatcherCube } from './dispatcher-cube'
import type { WorldActionRunner } from './world-actions'

function context() {
  return {
    actionRunner: { run: vi.fn<WorldActionRunner['run']>(async () => ({ kind: 'completed', ok: true })) },
    conditions: [],
    events: []
  }
}

describe('DispatcherCube', () => {
  it('offers new task, new session, and situation report inside the world', () => {
    render(<DispatcherCube context={context()} />)

    expect(screen.getByRole('button', { name: 'New task' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'New session' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'What needs my attention?' })).toBeTruthy()
  })
})
