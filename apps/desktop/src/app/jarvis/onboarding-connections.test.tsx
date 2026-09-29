import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { pl } from '@/i18n/pl'

import { ConnectionsStep } from './onboarding-connections'

afterEach(cleanup)

describe('ConnectionsStep roles', () => {
  it('lets the person say what they do, and reports the choice so the tools can be preselected', () => {
    const onSelectRole = vi.fn()

    render(
      <I18nProvider configClient={null} initialLocale="pl">
        <ConnectionsStep
          catalog={pl.jarvisConnections}
          copy={pl.jarvisOnboarding.connections}
          onSelectRole={onSelectRole}
          onToggle={vi.fn()}
          role="shop"
          selected={[]}
        />
      </I18nProvider>
    )

    expect(screen.getByRole('radio', { name: 'Sklep internetowy' }).getAttribute('aria-checked')).toBe('true')
    fireEvent.click(screen.getByRole('radio', { name: 'Programowanie' }))

    expect(onSelectRole).toHaveBeenCalledWith('developer')
  })

  it('shows no role picker where the step is used without one', () => {
    render(
      <I18nProvider configClient={null} initialLocale="pl">
        <ConnectionsStep catalog={pl.jarvisConnections} copy={pl.jarvisOnboarding.connections} onToggle={vi.fn()} selected={[]} />
      </I18nProvider>
    )

    expect(screen.queryByRole('radiogroup')).toBeNull()
  })
})
