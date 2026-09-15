import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'

import { useComposerPlaceholder } from './use-composer-placeholder'

function wrapper({ children }: { children: ReactNode }) {
  return <I18nProvider configClient={null}>{children}</I18nProvider>
}

describe('useComposerPlaceholder', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('updates the resting placeholder when the locale changes', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.99)

    const { result } = renderHook(
      () => {
        const i18n = useI18n()

        const placeholder = useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })

        return { ...i18n, placeholder }
      },
      { wrapper }
    )

    expect(result.current.placeholder).toBe('Start with a goal')

    await act(async () => {
      await result.current.setLocale('zh-hant')
    })

    await waitFor(() => expect(result.current.placeholder).toBe('從一個目標開始'))
  })

  it('keeps starter copy when a new session is persisted before a locale switch', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.99)
    const initialProps: { sessionId: null | string } = { sessionId: null }

    const { rerender, result } = renderHook(
      ({ sessionId }: { sessionId: null | string }) => {
        const i18n = useI18n()
        const placeholder = useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId })

        return { ...i18n, placeholder }
      },
      { initialProps, wrapper }
    )

    rerender({ sessionId: 'persisted-session' })
    expect(result.current.placeholder).toBe('Start with a goal')

    await act(async () => {
      await result.current.setLocale('zh-hant')
    })

    await waitFor(() => expect(result.current.placeholder).toBe('從一個目標開始'))
  })
})
