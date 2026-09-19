import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { SidebarFilterMenu } from './filter-menu'

vi.mock('@/components/ui/button', () => ({
  Button: ({ children, ...props }: { children?: ReactNode; [key: string]: unknown }) => (
    <button aria-label={props['aria-label'] as string | undefined}>{children}</button>
  )
}))

vi.mock('@/components/ui/codicon', () => ({ Codicon: () => null }))

vi.mock('@/components/ui/dropdown-menu', () => {
  const Wrapper = ({ children }: { children?: ReactNode }) => <div>{children}</div>

  return {
    DropdownMenu: Wrapper,
    DropdownMenuCheckboxItem: Wrapper,
    DropdownMenuContent: Wrapper,
    DropdownMenuGroup: Wrapper,
    DropdownMenuItem: Wrapper,
    DropdownMenuLabel: Wrapper,
    DropdownMenuRadioGroup: Wrapper,
    DropdownMenuRadioItem: Wrapper,
    DropdownMenuSeparator: () => null,
    DropdownMenuSub: Wrapper,
    DropdownMenuSubContent: Wrapper,
    DropdownMenuSubTrigger: Wrapper,
    DropdownMenuTrigger: Wrapper
  }
})

vi.mock('@/lib/desktop-git', () => ({
  desktopGit: () => ({ review: { prList: vi.fn() } })
}))

describe('SidebarFilterMenu i18n', () => {
  afterEach(cleanup)

  it('renders the session view controls from the French locale', () => {
    render(
      <I18nProvider configClient={null} initialLocale="fr">
        <SidebarFilterMenu />
      </I18nProvider>
    )

    expect(screen.getByRole('button', { name: 'Filtrer les sessions' })).toBeTruthy()

    for (const label of [
      'Regroupement',
      'Tri',
      'Afficher',
      'Style boîte de réception',
      'Filtres',
      'Demande de fusion',
      'Archivées',
      'Dernière activité',
      'Création',
      'Jetons',
      'Saisie requise',
      'En cours',
      'Non lues',
      'Inactives',
      'Ouvertes',
      'Brouillons',
      'Fusionnées',
      'Fermées',
      'Sans PR',
      'Tout marquer comme lu'
    ]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0)
    }

    for (const englishLabel of ['Grouping', 'Ordering', 'Inbox style', 'Needs input', 'Reset to defaults']) {
      expect(screen.queryByText(englishLabel)).toBeNull()
    }
  })
})
