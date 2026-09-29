import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ReleaseUpdateCard } from './release-update-card'

const openExternal = vi.fn()

function mount(result: unknown) {
  ;(window as unknown as { hermesDesktop: unknown }).hermesDesktop = {
    openExternal,
    updates: { releaseCheck: vi.fn(async () => result) }
  }

  return render(
    <I18nProvider configClient={null} initialLocale="pl">
      <ReleaseUpdateCard />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  openExternal.mockReset()
})

describe('ReleaseUpdateCard', () => {
  it('offers the installer for this machine when a newer release exists', async () => {
    mount({
      asset: { name: 'Setup.exe', url: 'https://example.test/Setup.exe' },
      currentVersion: '0.21.1',
      latestVersion: '0.22.0',
      notes: 'Vault',
      ok: true,
      pageUrl: 'https://example.test/release',
      publishedAt: null,
      updateAvailable: true
    })

    await waitFor(() => expect(screen.getByText('Dostępna jest wersja 0.22.0 (masz 0.21.1).')).toBeTruthy())
    fireEvent.click(screen.getByRole('button', { name: 'Pobierz instalator' }))

    expect(openExternal).toHaveBeenCalledWith('https://example.test/Setup.exe')
  })

  it('says so when up to date, and explains a missing release instead of failing silently', async () => {
    const { unmount } = mount({
      asset: null,
      currentVersion: '0.22.0',
      latestVersion: '0.22.0',
      notes: '',
      ok: true,
      pageUrl: 'https://example.test/release',
      publishedAt: null,
      updateAvailable: false
    })

    await waitFor(() => expect(screen.getByText('Masz najnowszą wersję (0.22.0).')).toBeTruthy())
    expect(screen.queryByRole('button', { name: 'Pobierz instalator' })).toBeNull()
    unmount()

    mount({ currentVersion: '0.22.0', ok: false, pageUrl: 'https://example.test/release', reason: 'no-release' })

    await waitFor(() => expect(screen.getByText(/Nie znaleziono opublikowanego wydania/)).toBeTruthy())
    expect(screen.getByRole('button', { name: 'Otwórz stronę wydań' })).toBeTruthy()
  })
})
