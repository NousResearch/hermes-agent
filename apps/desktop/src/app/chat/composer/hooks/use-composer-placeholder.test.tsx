import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import { $perSessionBrowse, browseBackward } from '@/store/composer-input-history'

import { useComposerPlaceholder } from './use-composer-placeholder'

function Wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider configClient={null}>{children}</I18nProvider>
}

beforeEach(() => {
  $perSessionBrowse.set({})
})

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

  it('preserves composer browse state when the locale changes', async () => {
    const sessionId = 'session-1'
    browseBackward(sessionId, 'unsent draft', ['previous message'])
    const stateBeforeLocaleChange = $perSessionBrowse.get()[sessionId]

    const { result } = renderHook(
      () => {
        const i18n = useI18n()
        useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId })

        return i18n
      },
      { wrapper: Wrapper }
    )

    await act(() => result.current.setLocale('ko'))

    expect($perSessionBrowse.get()[sessionId]).toEqual(stateBeforeLocaleChange)
  })
})
