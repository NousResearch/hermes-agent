import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/hermes'
import { type I18nConfigClient, I18nProvider } from '@/i18n'

import { AppearanceSettings } from './appearance-settings'

class TestResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)

Element.prototype.scrollIntoView = function scrollIntoView() {}

function pendingPromise<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('AppearanceSettings', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('keeps the language description stable while the selected locale is saving', async () => {
    const save = pendingPromise<{ ok: boolean }>()
    const latestConfig: HermesConfigRecord = { display: { language: 'en', skin: 'slate' } }

    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue(latestConfig),
      saveConfig: vi.fn().mockReturnValue(save.promise)
    }

    render(
      <I18nProvider configClient={configClient}>
        <AppearanceSettings />
      </I18nProvider>
    )

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Switch language' }).hasAttribute('disabled')).toBe(false)
    })

    fireEvent.click(screen.getByRole('button', { name: 'Switch language' }))
    fireEvent.click(screen.getByRole('option', { name: /日本語/i }))

    await waitFor(() => {
      expect(screen.queryByText('デスクトップインターフェイスの言語を選択します。')).not.toBeNull()
    })
    expect(screen.queryByText('言語を保存中…')).toBeNull()

    save.resolve({ ok: true })
  })
})
