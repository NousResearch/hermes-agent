import { describe, expect, it, vi } from 'vitest'

import {
  createManagedRolloutProvider,
  type ManagedRolloutProviderDependencies
} from './managed-rollout-provider'

function completeDependencies(measuredMaxInstallations?: () => number) {
  const resolveTarget = vi.fn(async () => {
    throw new Error('resolution-must-not-run-without-measured-capacity')
  })

  const dependencies = {
    inventoryReader: { capture: async () => null },
    sourceReader: { git: async () => '', nowMono: () => 1000 },
    assuranceReader: { readEvidence: async () => null, readProfile: async () => null },
    resolveTarget,
    journal: {
      create: vi.fn(),
      record: vi.fn(),
      read: vi.fn(),
      history: vi.fn(() => ({ items: [] })),
      events: vi.fn()
    },
    managedSshUpdateService: {
      issueLaunchCapability: vi.fn(),
      requestCoordinator: vi.fn()
    },
    observe: { observe: vi.fn() },
    evidence: { sweep: vi.fn() },
    ready: () => true,
    measuredMaxInstallations,
    processGeneration: 1
  } as unknown as ManagedRolloutProviderDependencies

  return { dependencies, resolveTarget }
}

describe('managed rollout capacity evidence', () => {
  it('does not advertise 500 installations merely because local dependencies are present', async () => {
    const { dependencies, resolveTarget } = completeDependencies()
    const provider = createManagedRolloutProvider(dependencies)

    await expect(provider.capabilities()).resolves.toEqual({
      protocol: 1,
      available: false,
      reason: 'unverified-capacity',
      maxConcurrency: 0,
      maxInstallations: 0
    })
    await expect(provider.resolveTarget({
      connectionIds: ['11111111-1111-4111-8111-111111111111'],
      inventoryRevision: 'inventory-1',
      retryOf: null
    })).rejects.toThrow('unverified-capacity')
    expect(resolveTarget).not.toHaveBeenCalled()
  })

  it('advertises only a valid measured count with serial update execution', async () => {
    const measured = createManagedRolloutProvider(completeDependencies(() => 2).dependencies)
    const unbounded = createManagedRolloutProvider(completeDependencies(() => 501).dependencies)

    await expect(measured.capabilities()).resolves.toEqual({
      protocol: 1,
      available: true,
      reason: null,
      maxConcurrency: 1,
      maxInstallations: 2
    })
    await expect(unbounded.capabilities()).resolves.toMatchObject({
      available: false,
      reason: 'unverified-capacity',
      maxInstallations: 0
    })
  })
})
