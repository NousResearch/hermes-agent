import { act, cleanup, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, expect, it } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'

import { useComposerPlaceholder } from './use-composer-placeholder'

afterEach(cleanup)

it('refreshes a mounted composer when Arabic resolves without a session change', async () => {
  const { result } = renderHook(
    () => ({
      i18n: useI18n(),
      placeholder: useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId: null })
    }),
    {
      wrapper: ({ children }: { children: ReactNode }) => (
        <I18nProvider configClient={null} initialLocale="en">
          {children}
        </I18nProvider>
      )
    }
  )

  expect(result.current.i18n.t.composer.newSessionPlaceholders).toContain(result.current.placeholder)
  await act(() => result.current.i18n.setLocale('ar'))
  expect(result.current.i18n.t.composer.newSessionPlaceholders).toContain(result.current.placeholder)
  expect(result.current.placeholder).toMatch(/[\u0600-\u06ff]/)
})

it('preserves the starter when a new session receives its ID', () => {
  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: null | string }) =>
      useComposerPlaceholder({ disabled: false, reconnecting: false, sessionId }),
    {
      initialProps: { sessionId: null as null | string },
      wrapper: ({ children }: { children: ReactNode }) => <I18nProvider configClient={null}>{children}</I18nProvider>
    }
  )

  const starter = result.current

  rerender({ sessionId: 'persisted' })
  expect(result.current).toBe(starter)
})
