import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { managedRolloutsAr } from '@/i18n/managed-rollouts'

import { RolloutHistory } from './rollout-history'

const entry = (id: string) => ({
  id,
  phase: 'completed',
  updatedAt: 'now',
  unresolved: 0,
  archived: false,
  reason: null
})

afterEach(() => {
  document.documentElement.dir = 'ltr'
})

describe('managed rollout history', () => {
  it('bounds history and preserves selected detail identity', () => {
    const onSelect = vi.fn()
    const entries = Array.from({ length: 51 }, (_, index) => entry(`r${index}`))

    render(<RolloutHistory entries={entries} onSelect={onSelect} />)

    expect(screen.getAllByRole('button')).toHaveLength(50)
    expect(screen.getByText(/r0/)).toBeTruthy()
    expect(screen.queryByText(/r50/)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /r0/ }))

    expect(onSelect).toHaveBeenCalledWith('r0')
  })

  it('keeps history rows keyboard reachable without submitting an owning form', () => {
    const onSelect = vi.fn()
    const onSubmit = vi.fn(event => event.preventDefault())

    render(
      <form onSubmit={onSubmit}>
        <RolloutHistory entries={[entry('r1')]} onSelect={onSelect} />
      </form>
    )

    const row = screen.getByRole('button', { name: /r1/ })
    expect(row).toHaveProperty('type', 'button')
    expect(row).toHaveProperty('tabIndex', 0)

    row.focus()
    expect(document.activeElement).toBe(row)

    fireEvent.click(row)

    expect(onSelect).toHaveBeenCalledWith('r1')
    expect(onSubmit).not.toHaveBeenCalled()
    expect(row.getAttribute('aria-pressed')).toBe('true')
  })

  it('uses localized RTL copy and logical alignment in compact layouts', () => {
    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <RolloutHistory
          entries={[{ ...entry('r-ar'), unresolved: 1, archived: true, reason: 'operator stopped' }]}
          onSelect={() => undefined}
        />
      </I18nProvider>
    )

    const row = screen.getByRole('button', { name: managedRolloutsAr.a11y.historyEntry('r-ar') })

    expect(document.documentElement.dir).toBe('rtl')
    expect(screen.getByRole('region', { name: managedRolloutsAr.sections.history })).toBeTruthy()
    expect(row.className).toContain('min-w-0')
    expect(row.className).toContain('text-start')
    expect(row.className).not.toContain('text-left')
    expect(screen.getByText(managedRolloutsAr.labels.reason('operator stopped'))).toBeTruthy()
  })

  it('keeps the rendered history bounded and width-safe at fleet scale', () => {
    const entries = Array.from({ length: 500 }, (_, index) => entry(`scale-${index}`))

    render(<RolloutHistory entries={entries} onSelect={() => undefined} />)

    const history = screen.getByRole('region', { name: 'Managed rollout history' })
    expect(screen.getAllByRole('button')).toHaveLength(50)
    expect(history.className).toContain('min-w-0')
    expect(screen.queryByText(/scale-499/)).toBeNull()
  })
})
