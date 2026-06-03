import { act, renderHook } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider, useI18n } from './context'

const STORAGE_KEY = 'hermes-desktop-locale'

// This jsdom build ships a partial localStorage (missing removeItem/clear), so
// back it with a Map for a deterministic, self-contained test.
function installStorageMock() {
  const store = new Map<string, string>()

  const mock: Storage = {
    get length() {
      return store.size
    },
    clear: () => store.clear(),
    getItem: key => store.get(key) ?? null,
    key: index => Array.from(store.keys())[index] ?? null,
    removeItem: key => void store.delete(key),
    setItem: (key, value) => void store.set(key, String(value))
  }

  Object.defineProperty(window, 'localStorage', { configurable: true, value: mock })
}

const wrapper = ({ children }: { children: ReactNode }) => <I18nProvider>{children}</I18nProvider>

describe('I18nProvider', () => {
  beforeEach(installStorageMock)

  it('defaults to English', () => {
    const { result } = renderHook(() => useI18n(), { wrapper })

    expect(result.current.locale).toBe('en')
    expect(result.current.t.language.label).toBe('Language')
  })

  it('switches translations and persists the locale', () => {
    const { result } = renderHook(() => useI18n(), { wrapper })

    act(() => result.current.setLocale('zh'))

    expect(result.current.locale).toBe('zh')
    expect(result.current.t.language.label).toBe('语言')
    expect(window.localStorage.getItem(STORAGE_KEY)).toBe('zh')
  })

  it('restores a persisted locale on mount', () => {
    window.localStorage.setItem(STORAGE_KEY, 'zh')

    const { result } = renderHook(() => useI18n(), { wrapper })

    expect(result.current.locale).toBe('zh')
  })
})
