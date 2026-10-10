import { act, renderHook } from '@testing-library/react'
import { beforeEach, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n/context'
import { dropPreviewTabsForProfile, migratePreviewTabsForProfile, setPreviewScope } from '@/store/preview'

import { $lensCards, registerLensGuest } from './store'
import { useLens } from './use-lens'

const source = {
  url: 'https://example.com/item',
  title: 'Source',
  text: 'Evidence',
  selector: '#item',
  tag: 'P',
  truncated: false
}

beforeEach(() => {
  localStorage.clear()
  setPreviewScope('pending')
})

it.each(['delete', 'rename', 'switch-back', 'unmount', 'other-window'] as const)(
  'does not commit a pending capture after %s',
  async change => {
    let finish!: (value: unknown) => void

    const pending = new Promise<unknown>(resolve => {
      finish = resolve
    })

    const guest = {
      getURL: () => source.url,
      executeJavaScript: () => pending,
      addEventListener() {},
      removeEventListener() {}
    }

    const unregister = registerLensGuest(guest)

    const hook = renderHook(() => useLens(() => guest), {
      wrapper: ({ children }) => <I18nProvider initialLocale="en">{children}</I18nProvider>
    })

    act(() => hook.result.current.onPin('page'))
    act(() => {
      if (change === 'delete') {
        dropPreviewTabsForProfile('pending')
      }

      if (change === 'rename') {
        migratePreviewTabsForProfile('pending', 'renamed')
      }

      if (change === 'switch-back') {
        setPreviewScope('elsewhere')
        setPreviewScope('pending')
      }

      if (change === 'unmount') {
        hook.unmount()
      }

      if (change === 'other-window') {
        localStorage.setItem(
          'hermes.desktop.lens.epoch.v1.' + encodeURIComponent('conn:local::pending'),
          'changed-elsewhere'
        )
      }
    })
    await act(async () => {
      finish(source)
      await pending
    })
    act(() => setPreviewScope('pending'))
    expect($lensCards.get()).toEqual([])
    expect(Object.keys(localStorage).filter(key => key.startsWith('hermes.desktop.lens.card.v1.'))).toEqual([])
    unregister()

    if (change !== 'unmount') {
      hook.unmount()
    }
  }
)

it('commits an owned capture and rejects a guest from another scope', async () => {
  const guest = {
    getURL: () => source.url,
    executeJavaScript: async () => source,
    addEventListener() {},
    removeEventListener() {}
  }

  const unregister = registerLensGuest(guest)

  const hook = renderHook(() => useLens(() => guest), {
    wrapper: ({ children }) => <I18nProvider initialLocale="en">{children}</I18nProvider>
  })

  await act(async () => {
    hook.result.current.onPin('page')
  })
  expect($lensCards.get()).toHaveLength(1)
  act(() => setPreviewScope('elsewhere'))
  await act(async () => {
    hook.result.current.onPin('page')
  })
  expect($lensCards.get()).toEqual([])
  expect(hook.result.current.error).toBeTruthy()
  hook.unmount()
  unregister()
})
