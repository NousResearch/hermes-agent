import { afterEach, describe, expect, it, vi } from 'vitest'

import { getAppLocale, publishAppLocale, subscribeAppLocale } from './app-locale'
import { DEFAULT_LOCALE } from './languages'
import { host } from '@/sdk'

const reset = () => publishAppLocale(DEFAULT_LOCALE)

describe('app locale publisher', () => {
  afterEach(reset)

  it('starts on the default locale', () => {
    expect(getAppLocale()).toBe(DEFAULT_LOCALE)
  })

  it('emits the current value on subscribe, then every change', () => {
    const listener = vi.fn()
    const dispose = subscribeAppLocale(listener)

    expect(listener).toHaveBeenCalledTimes(1)
    expect(listener).toHaveBeenLastCalledWith(DEFAULT_LOCALE)

    publishAppLocale('zh')
    expect(listener).toHaveBeenLastCalledWith('zh')

    publishAppLocale('ja')
    expect(listener).toHaveBeenLastCalledWith('ja')
    expect(listener).toHaveBeenCalledTimes(3)

    dispose()
  })

  it('does not fire when the value is unchanged', () => {
    const listener = vi.fn()
    const dispose = subscribeAppLocale(listener)
    listener.mockClear()

    // A registry-version re-render republishes the same locale; listeners that
    // only care about the language must not wake up.
    publishAppLocale(DEFAULT_LOCALE)
    expect(listener).not.toHaveBeenCalled()

    dispose()
  })

  it('stops notifying after the disposer runs', () => {
    const listener = vi.fn()
    const dispose = subscribeAppLocale(listener)

    dispose()
    publishAppLocale('zh')

    expect(listener).toHaveBeenCalledTimes(1)
    expect(getAppLocale()).toBe('zh')
  })

  it('reaches plugins through host.i18n', () => {
    expect(host.i18n.locale()).toBe(DEFAULT_LOCALE)

    const listener = vi.fn()
    const dispose = host.i18n.subscribeLocale(listener)

    expect(listener).toHaveBeenLastCalledWith(DEFAULT_LOCALE)

    publishAppLocale('zh')
    expect(host.i18n.locale()).toBe('zh')
    expect(listener).toHaveBeenLastCalledWith('zh')

    dispose()
    publishAppLocale('ja')
    expect(listener).toHaveBeenCalledTimes(2)
  })
})
