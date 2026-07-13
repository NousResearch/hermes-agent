// @vitest-environment jsdom
import { renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { type I18nConfigClient, I18nProvider, TRANSLATIONS } from '@/i18n'

import { useComposerPlaceholder } from './use-composer-placeholder'

describe('useComposerPlaceholder', () => {
  it('refreshes the resting placeholder after the configured locale loads', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'ar' } }),
      saveConfig: vi.fn()
    }
    const wrapper = ({ children }: { children: ReactNode }) => (
      <I18nProvider configClient={configClient}>{children}</I18nProvider>
    )

    const { result } = renderHook(
      () => useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null }),
      { wrapper }
    )

    await waitFor(() => {
      expect(TRANSLATIONS.ar.composer.newSessionPlaceholders).toContain(result.current)
    })
    expect(TRANSLATIONS.en.composer.newSessionPlaceholders).not.toContain(result.current)
  })
})
