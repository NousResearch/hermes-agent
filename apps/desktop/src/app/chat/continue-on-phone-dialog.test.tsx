import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ContinueOnPhoneDialog } from './continue-on-phone-dialog'
import { SessionActionsMenu } from './sidebar/session-actions-menu'

const openExternal = vi.fn().mockResolvedValue(undefined)

beforeEach(() => {
  openExternal.mockClear()
  window.hermesDesktop = {
    ...(window.hermesDesktop ?? {}),
    openExternal
  } as Window['hermesDesktop']
})

function renderDialog(
  overrides: Partial<React.ComponentProps<typeof ContinueOnPhoneDialog>> = {}
) {
  return render(
    <I18nProvider>
      <ContinueOnPhoneDialog
        generateQr={vi.fn().mockResolvedValue('data:image/png;base64,qr')}
        onOpenChange={vi.fn()}
        open
        profile="work"
        resolveUrl={vi.fn().mockResolvedValue({
          ok: true,
          url: 'https://hermes.example.com/chat?resume=session-42&profile=work'
        })}
        sessionId="session-42"
        {...overrides}
      />
    </I18nProvider>
  )
}

describe('ContinueOnPhoneDialog', () => {
  it('is reachable from the session actions menu', async () => {
    render(
      <I18nProvider>
        <SessionActionsMenu sessionId="session-42" title="Research session">
          <button type="button">Session menu</button>
        </SessionActionsMenu>
      </I18nProvider>
    )

    fireEvent.pointerDown(screen.getByRole('button', { name: 'Session menu' }), {
      button: 0,
      ctrlKey: false,
      pointerType: 'mouse'
    })
    fireEvent.click(await screen.findByText('Continue on phone'))

    expect(await screen.findByRole('dialog', { name: 'Continue on phone' })).toBeTruthy()
  })

  it('shows a scannable continuation link and can open the same URL', async () => {
    const resolveUrl = vi.fn().mockResolvedValue({
      ok: true,
      url: 'https://hermes.example.com/chat?resume=session-42&profile=work'
    })

    const generateQr = vi.fn().mockResolvedValue('data:image/png;base64,qr')

    renderDialog({ generateQr, resolveUrl })

    const qr = await screen.findByRole('img', { name: 'QR code for this Hermes session' })
    expect(qr.getAttribute('src')).toBe('data:image/png;base64,qr')
    expect(resolveUrl).toHaveBeenCalledWith('session-42', 'work')
    expect(generateQr).toHaveBeenCalledWith(
      'https://hermes.example.com/chat?resume=session-42&profile=work'
    )

    fireEvent.click(screen.getByRole('button', { name: 'Open in browser' }))

    await waitFor(() =>
      expect(openExternal).toHaveBeenCalledWith(
        'https://hermes.example.com/chat?resume=session-42&profile=work'
      )
    )
  })

  it('shows a recoverable error when secure remote access is unavailable', async () => {
    const resolveUrl = vi.fn().mockResolvedValue({ ok: false, reason: 'auth-required' })

    renderDialog({ resolveUrl })

    expect(await screen.findByText('Remote continuation is not ready')).toBeTruthy()
    expect(screen.getByText('Configure an HTTPS dashboard public URL with OAuth browser sign-in, then try again. Token-authenticated dashboards cannot be opened from a phone browser.')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))

    await waitFor(() => expect(resolveUrl).toHaveBeenCalledTimes(2))
  })
})
