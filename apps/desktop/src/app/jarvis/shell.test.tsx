import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'

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

describe('JarvisShell', () => {
  it('renders the focused product navigation with 44px targets', () => {
    renderShell('jarvis')

    expect(screen.getByRole('navigation', { name: 'Główna nawigacja' })).toBeTruthy()

    for (const label of ['Jarvis', 'Zadania', 'Pamięć', 'Narzędzia']) {
      expect(screen.getByRole('button', { name: label }).className).toContain('min-h-11')
    }

    expect(screen.getByRole('button', { name: 'Jarvis' }).getAttribute('aria-current')).toBe('page')
    expect(screen.getByRole('button', { name: 'Ustawienia' }).className).toContain('min-h-11')
    expect(screen.getByRole('button', { name: 'Profil' }).className).toContain('min-h-11')
  })

  it('moves through navigation by keyboard without leaving the nav group', () => {
    renderShell('jarvis')

    const jarvis = screen.getByRole('button', { name: 'Jarvis' })
    jarvis.focus()
    fireEvent.keyDown(jarvis, { key: 'ArrowDown' })

    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Zadania' }))

    fireEvent.keyDown(screen.getByRole('button', { name: 'Zadania' }), { key: 'End' })

    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Narzędzia' }))
  })

  it('uses the real Jarvis UI store and JarvisCore on the default Jarvis screen', () => {
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

    fireEvent.click(screen.getByRole('button', { name: 'Pamięć' }))

    expect(screen.getByRole('button', { name: 'Pamięć' }).getAttribute('aria-current')).toBe('page')
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
})
