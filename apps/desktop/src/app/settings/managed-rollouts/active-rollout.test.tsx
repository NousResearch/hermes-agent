import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ActiveRollout } from './active-rollout'

describe('active rollout progress', () => {
  it('keeps receipt and readiness separate and renders authoritative counts', () => {
    render(<ActiveRollout draft={{ mode: 'manual', concurrency: 1, canaryInstallId: 'a', selectedInstallIds: ['a', 'b'] }} state={{ phase: 'running', completed: 1, total: 2, receipt: { outcome: 'accepted', correlationId: 'c1' }, readiness: { ready: false, reason: 'scope pending' }, canaryGate: 'approved', restartRequired: false }} />)
    expect(screen.getByText(/Progress: 1 of 2/)).toBeTruthy()
    expect(screen.getByText(/Receipt: accepted/)).toBeTruthy()
    expect(screen.getByText(/Readiness: scope pending/)).toBeTruthy()
  })
})
