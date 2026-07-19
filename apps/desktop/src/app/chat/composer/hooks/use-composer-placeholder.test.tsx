import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { describe, expect, it } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'

import { useComposerPlaceholder } from './use-composer-placeholder'

function Wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider configClient={null}>{children}</I18nProvider>
}

describe('useComposerPlaceholder', () => {
  it('re-picks the resting placeholder when the locale changes', async () => {
    const { result } = renderHook(
      () => {
        const i18n = useI18n()
        const placeholder = useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })

        return { i18n, placeholder }
      },
      { wrapper: Wrapper }
    )

    const englishPlaceholders = result.current.i18n.t.composer.newSessionPlaceholders

    expect(englishPlaceholders).toContain(result.current.placeholder)

    await act(() => result.current.i18n.setLocale('ko'))

    expect(result.current.i18n.t.composer.newSessionPlaceholders).toContain(result.current.placeholder)
    expect(englishPlaceholders).not.toContain(result.current.placeholder)
  })
})
