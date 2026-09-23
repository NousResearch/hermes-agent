import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { managedRolloutsAr, managedRolloutsEn } from '@/i18n/managed-rollouts'
import type { ManagedRolloutInventory } from '@/store/managed-rollouts'

import { FleetOverview, targetsFromInventory } from './fleet-overview'

const inventory: ManagedRolloutInventory = {
  inventoryRevision: 'inventory-1',
  capturedMono: 123,
  observations: [
    {
      installId: 'a'.repeat(32),
      connectionId: '11111111-1111-4111-8111-111111111111',
      aliasConnectionIds: ['22222222-2222-4222-8222-222222222222'],
      codeRoot: '/srv/a',
      repositoryId: 'github.com/NousResearch/hermes-agent',
      headSha: 'b'.repeat(40),
      requiredScopeIds: ['main'],
      source: { connectionId: '11111111-1111-4111-8111-111111111111', verifiedHostKeyFingerprint: 'same-host' }
    },
    {
      installId: 'c'.repeat(32),
      connectionId: '33333333-3333-4333-8333-333333333333',
      aliasConnectionIds: [],
      codeRoot: '/srv/b',
      repositoryId: 'github.com/NousResearch/hermes-agent',
      headSha: 'd'.repeat(40),
      requiredScopeIds: ['main'],
      source: { connectionId: '33333333-3333-4333-8333-333333333333', verifiedHostKeyFingerprint: 'same-host' }
    }
  ]
}

describe('managed rollout fleet overview', () => {
  it('folds aliases into one installation row and warns for two installs on one host', () => {
    const targets = targetsFromInventory(inventory)
    const onToggle = vi.fn()
    render(<FleetOverview onToggle={onToggle} selected={new Set()} targets={targets} />)

    expect(screen.getAllByRole('button', { name: 'Select' })).toHaveLength(2)
    expect(screen.getAllByText(/Shared machine/)).toHaveLength(2)
    expect(screen.getByText(/22222222-2222-4222-8222-222222222222/)).toBeTruthy()
    fireEvent.click(screen.getAllByRole('button', { name: 'Select' })[0])
    expect(onToggle).toHaveBeenCalledWith(JSON.stringify(['same-host', 'a'.repeat(32)]))
  })

  it('shows the observed HEAD and states which eligibility and wall time facts are unknown', () => {
    const [target] = targetsFromInventory(inventory)

    expect(target.headSha).toBe('b'.repeat(40))
    expect(target.eligibility).toBe('unknown')
    expect(target.observedAt).toBeNull()
    render(<FleetOverview onToggle={() => undefined} selected={new Set()} targets={[target]} />)
    expect(screen.getByText('b'.repeat(40))).toBeTruthy()
    expect(screen.getByText(/Eligibility: unknown/i)).toBeTruthy()
    expect(screen.getByText(/Observation time: unknown/i)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Select' })).toHaveProperty('disabled', false)
  })

  it('makes every full alias discoverable through a keyboard accessible disclosure', () => {
    const longAlias = 'fleet-alias-' + 'x'.repeat(100)
    const [target] = targetsFromInventory({ ...inventory, observations: [{ ...inventory.observations[0], aliasConnectionIds: [longAlias] }] })
    render(<FleetOverview onToggle={() => undefined} selected={new Set()} targets={[target]} />)

    const disclosure = screen.getByText(/Aliases \(1\)/)
    expect(disclosure.tagName).toBe('SUMMARY')
    expect(screen.getByText(longAlias)).toBeTruthy()
    expect(disclosure.closest('details')).toBeTruthy()
  })

  it('makes an empty observed inventory explicit', () => {
    render(<FleetOverview onToggle={() => undefined} selected={new Set()} targets={[]} />)
    expect(screen.getByText(/No managed SSH installations were observed/)).toBeTruthy()
  })

  it('renders Arabic selection and shared-machine warnings without English fallback', () => {
    const targets = targetsFromInventory(inventory)

    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <FleetOverview onToggle={() => undefined} selected={new Set([JSON.stringify(['same-host', 'a'.repeat(32)])])} targets={targets} />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: managedRolloutsAr.actions.selected })).toBeTruthy()
    expect(screen.getByRole('button', { name: managedRolloutsAr.actions.select })).toBeTruthy()
    expect(screen.getAllByText(managedRolloutsAr.warnings.sharedMachine)).toHaveLength(2)
    expect(screen.getAllByText(`${managedRolloutsAr.labels.eligibility}: ${managedRolloutsAr.labels.unknownFact}`)).toHaveLength(2)
    expect(screen.getAllByText(`${managedRolloutsAr.labels.observationTime}: ${managedRolloutsAr.labels.unknownFact}`)).toHaveLength(2)
    expect(screen.queryByText(managedRolloutsEn.warnings.sharedMachine)).toBeNull()
  })
})
