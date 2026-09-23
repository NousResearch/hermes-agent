import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { managedRolloutsAr, managedRolloutsEn } from '@/i18n/managed-rollouts'

import { PreparationPanel } from './preparation-panel'

const target = { installId: 'install-a', machineId: 'machine-a', label: 'Lab', headSha: 'a'.repeat(40), eligibility: 'unknown' as const, observedAt: null, sharedMachine: false }

describe('managed rollout preparation panel', () => {
  it('requires explicit confirmation and selection before preparation', () => {
    const onPrepare = vi.fn()
    render(<PreparationPanel onPrepare={onPrepare} reviewGeneration="g1" targets={[target]} />)
    const prepare = screen.getByRole('button', { name: /prepare selected/i })
    expect(prepare).toHaveProperty('disabled', true)
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    expect(prepare).toHaveProperty('disabled', true)
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(prepare)
    expect(onPrepare).toHaveBeenCalledWith([target])
  })

  it('clears confirmation when selection changes, forcing requalification', () => {
    const onPrepare = vi.fn()
    render(<PreparationPanel onPrepare={onPrepare} reviewGeneration="g1" targets={[target]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select' }))
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: 'Selected' }))
    expect(screen.getByRole('button', { name: /prepare selected/i })).toHaveProperty('disabled', true)
  })

  it('keeps review and preparation controls above a bounded list of 160 searchable targets', () => {
    const targets = Array.from({ length: 160 }, (_, index) => ({ ...target, installId: `install-${index}`, label: `Lab ${index}` }))
    const onReview = vi.fn()
    render(<PreparationPanel onPrepare={vi.fn()} onReview={onReview} reviewGeneration="g160" targets={targets} />)

    const panel = screen.getByRole('region', { name: managedRolloutsEn.sections.preparation })
    const review = within(panel).getByRole('button', { name: /review selected target/i })
    const list = within(panel).getByRole('region', { name: managedRolloutsEn.sections.fleet })
    expect(review.compareDocumentPosition(list) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(list.className).toContain('max-h-')
    fireEvent.change(within(panel).getByRole('textbox', { name: /search targets/i }), { target: { value: 'Lab 159' } })
    expect(within(list).getAllByRole('button', { name: 'Select' })).toHaveLength(1)
    fireEvent.click(within(list).getByRole('button', { name: 'Select' }))
    expect(review).toHaveProperty('disabled', false)
    fireEvent.click(review)
    expect(onReview).toHaveBeenCalledWith([targets[159]])
  })

  it('localizes below fold preparation copy, controls and region labels in Arabic', () => {
    render(<I18nProvider configClient={null} initialLocale="ar"><PreparationPanel onPrepare={vi.fn()} onReview={vi.fn()} reviewGeneration="ar" targets={[target]} /></I18nProvider>)
    const panel = screen.getByRole('region', { name: managedRolloutsAr.sections.preparation })
    expect(within(panel).getByRole('region', { name: managedRolloutsAr.sections.fleet })).toBeTruthy()
    expect(within(panel).getByRole('button', { name: managedRolloutsAr.actions.prepare })).toBeTruthy()
    expect(within(panel).getByRole('checkbox', { name: managedRolloutsAr.a11y.confirmPreparation })).toBeTruthy()
    expect(within(panel).queryByText(/Preparation runs the existing individual updater/)).toBeNull()
  })
})
