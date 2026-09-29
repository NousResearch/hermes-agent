import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { GoogleConnectDialog } from './google-connect-dialog'

const openExternal = vi.fn()

function mount(routes: Record<string, unknown>) {
  const api = vi.fn(async ({ path }: { path: string }) => {
    const hit = Object.entries(routes).find(([prefix]) => path.startsWith(prefix))

    if (!hit) {
      throw new Error(`unexpected ${path}`)
    }

    return typeof hit[1] === 'function' ? (hit[1] as () => unknown)() : hit[1]
  })

  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = { api, openExternal }

  render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale="pl">
        <GoogleConnectDialog onClose={vi.fn()} open />
      </I18nProvider>
    </MemoryRouter>
  )

  return api
}

afterEach(() => {
  cleanup()
  openExternal.mockReset()
})

describe('GoogleConnectDialog', () => {
  it('resumes at sign-in when the client file is already stored, and finishes with the pasted address', async () => {
    const status = vi.fn().mockResolvedValueOnce({ client_secret: true, token: false })

    const api = mount({
      '/api/google/auth-code': { ok: true, warning: null },
      '/api/google/auth-url': { url: 'https://accounts.google.com/o/oauth2/auth?x=1' },
      '/api/google/status': status
    })

    await waitFor(() => expect(screen.getByText('2 · Logowanie')).toBeTruthy())
    // The step that needs the client file is not shown: nothing to redo.
    expect(screen.queryByText('1 · Twoje dane logowania Google')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Zaloguj przez Google' }))
    await waitFor(() => expect(openExternal).toHaveBeenCalledWith('https://accounts.google.com/o/oauth2/auth?x=1'))

    fireEvent.change(screen.getByLabelText('Wklej tutaj kod lub cały adres'), {
      target: { value: 'http://localhost:1/?code=abc&state=s' }
    })
    status.mockResolvedValue({ client_secret: true, token: true })
    fireEvent.click(screen.getByRole('button', { name: 'Dokończ logowanie' }))

    await waitFor(() =>
      expect(api).toHaveBeenCalledWith(
        expect.objectContaining({ body: { code: 'http://localhost:1/?code=abc&state=s' }, path: '/api/google/auth-code' })
      )
    )
    await waitFor(() => expect(screen.getByText('3 · Sprawdzenie')).toBeTruthy())
  })

  it('proves the connection by reading real data back, and says which half did not answer', async () => {
    mount({
      '/api/google/status': { client_secret: true, token: true },
      '/api/google/verify': {
        calendar_ok: true,
        events: [{ start: '2026-10-01T09:00:00+02:00', summary: 'Spotkanie z Anną' }],
        gmail_ok: false,
        ok: true,
        unread: []
      }
    })

    await waitFor(() => expect(screen.getByText('3 · Sprawdzenie')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Sprawdź połączenie' }))

    await waitFor(() => expect(screen.getByText('Google jest połączony i odpowiada.')).toBeTruthy())
    expect(screen.getByText('Spotkanie z Anną')).toBeTruthy()
    expect(screen.getByText('Poczta nie odpowiedziała (brakuje uprawnienia?).')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Pokaż mój dzień' })).toBeTruthy()
  })
})
