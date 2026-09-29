import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import * as profiles from '@/store/profile'
import { ThemeProvider } from '@/themes/context'

import { JarvisShell } from './shell'
import { $jarvisUi } from './store'

function renderShell(initialView: React.ComponentProps<typeof JarvisShell>['initialView'] = 'jarvis') {
  return render(
    <I18nProvider configClient={null} initialLocale="pl">
      <JarvisShell initialView={initialView} />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
})

describe('Agent CzesiekShell', () => {
  it('opens a fresh conversation in the active profile', () => {
    const start = vi.spyOn(profiles, 'newSessionInProfile').mockImplementation(() => {})
    renderShell('jarvis')
    fireEvent.click(screen.getByRole('button', { name: 'Nowa sesja' }))
    expect(start).toHaveBeenCalledWith(profiles.$activeGatewayProfile.get() || 'default')
    start.mockRestore()
  })

  it('renders the focused product navigation with 44px targets', () => {
    renderShell('jarvis')

    const navigation = screen.getByRole('navigation', { name: 'Główna nawigacja' })

    const labels = [
      'Pulpit',
      'Zadania',
      'Agenci',
      'Nowa sesja',
      'Historia',
      'Mapa wiedzy',
      'Moje prompty',
      'Pliki i wyniki',
      'Wspomnienia',
      'Integracje',
      'Automatyzacje',
      'Narzędzia',
      'Monitor systemu',
      'Ustawienia'
    ]

    expect(
      within(navigation)
        .getAllByRole('button')
        .map(button => button.textContent)
    ).toEqual(labels)

    for (const label of labels) {
      expect(screen.getByRole('button', { name: label }).className).toContain('min-h-11')
    }

    expect(screen.getByRole('button', { name: 'Pulpit' }).getAttribute('aria-current')).toBe('page')
    expect(screen.getByRole('button', { name: 'Ustawienia' }).className).toContain('min-h-11')
    expect(screen.getByRole('button', { name: 'Profil' }).className).toContain('min-h-11')
  })

  it('moves through navigation by keyboard without leaving the nav group', () => {
    renderShell('jarvis')

    const jarvis = screen.getByRole('button', { name: 'Pulpit' })
    jarvis.focus()
    fireEvent.keyDown(jarvis, { key: 'ArrowDown' })

    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Zadania' }))

    fireEvent.keyDown(screen.getByRole('button', { name: 'Zadania' }), { key: 'End' })

    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Ustawienia' }))
  })

  it('uses the real Agent Czesiek UI store and Agent CzesiekCore on the default Agent Czesiek screen', () => {
    $jarvisUi.set({
      activeTool: null,
      activity: [],
      sessionId: 's1',
      task: { id: 't1', phase: 'running' },
      voice: 'listening'
    })

    renderShell('jarvis')

    const core = screen.getByTestId('jarvis-core')
    expect(core.getAttribute('data-voice')).toBe('listening')
    expect(core.getAttribute('data-task')).toBe('running')
    expect(screen.getByRole('main').getAttribute('data-jarvis-view')).toBe('jarvis')
  })

  it('switches the active semantic view without remounting another store', () => {
    renderShell('jarvis')

    fireEvent.click(screen.getByRole('button', { name: 'Wspomnienia' }))

    expect(screen.getByRole('button', { name: 'Wspomnienia' }).getAttribute('aria-current')).toBe('page')
    expect(screen.getByRole('main').getAttribute('data-jarvis-view')).toBe('memory')
  })

  it('does not nest a main landmark around runtime children that own their surface landmark', () => {
    render(
      <I18nProvider configClient={null} initialLocale="pl">
        <JarvisShell initialView="settings">
          <main data-testid="runtime-main">Settings surface</main>
        </JarvisShell>
      </I18nProvider>
    )

    expect(screen.getAllByRole('main')).toHaveLength(1)
    expect(screen.getByTestId('runtime-main')).toBe(screen.getByRole('main'))
  })

  it('switches the whole app between light and dark from the rail', () => {
    render(
      <ThemeProvider>
        <I18nProvider configClient={null} initialLocale="pl">
          <JarvisShell initialView="jarvis" />
        </I18nProvider>
      </ThemeProvider>
    )

    fireEvent.click(screen.getByRole('radio', { name: 'Ciemny' }))
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(screen.getByRole('radio', { name: 'Ciemny' }).getAttribute('aria-checked')).toBe('true')

    fireEvent.click(screen.getByRole('radio', { name: 'Jasny' }))
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })
})
