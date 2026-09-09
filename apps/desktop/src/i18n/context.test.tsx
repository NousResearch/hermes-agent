import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord } from '@/hermes'

import { type I18nConfigClient, I18nProvider, useI18n } from './context'
import type { Locale } from './types'

function LanguageProbe({ target = 'zh' }: { target?: Locale }) {
  const { isLoadingConfig, isSavingLocale, locale, saveError, setLocale, t } = useI18n()

  return (
    <div>
      <p data-testid="locale">{locale}</p>
      <p data-testid="label">{t.language.label}</p>
      <p data-testid="save">{t.common.save}</p>
      <p data-testid="loading">{String(isLoadingConfig)}</p>
      <p data-testid="saving">{String(isSavingLocale)}</p>
      <p data-testid="save-error">{saveError?.message ?? ''}</p>
      <button onClick={() => void setLocale(target).catch(() => undefined)} type="button">
        switch
      </button>
    </div>
  )
}

describe('I18nProvider', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('defaults to English without a config client', () => {
    render(
      <I18nProvider configClient={null}>
        <LanguageProbe />
      </I18nProvider>
    )

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
  })

  it('normalizes an initial locale alias and switches translations', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="zh-CN">
        <LanguageProbe target="en" />
      </I18nProvider>
    )

    expect(screen.getByTestId('locale').textContent).toBe('zh')
    expect(screen.getByTestId('label').textContent).toBe('语言')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('en'))
    expect(screen.getByTestId('label').textContent).toBe('Language')
  })

  it('loads the initial locale from display.language config', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'zh-Hans' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('zh')
    expect(screen.getByTestId('label').textContent).toBe('语言')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('keeps English usable when config loading fails', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockRejectedValue(new Error('config unavailable')),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('loads zh-hant from display.language config', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'zh-TW' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('zh-hant')
    expect(screen.getByTestId('save').textContent).toBe('儲存')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('loads ja from display.language config', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'ja-JP' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('ja')
    expect(screen.getByTestId('save').textContent).toBe('保存')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('does not overwrite unsupported configured languages', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'de' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('reads latest config before saving language and preserves unrelated values', async () => {
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })

    const latestConfig: HermesConfigRecord = {
      display: { language: 'en', skin: 'slate' },
      terminal: { cwd: '/new' }
    }

    const configClient: I18nConfigClient = {
      getConfig: vi
        .fn()
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'mono' }, terminal: { cwd: '/old' } })
        .mockResolvedValueOnce(latestConfig),
      saveConfig
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledTimes(1))
    expect(saveConfig).toHaveBeenCalledWith({
      display: { language: 'zh', skin: 'slate' },
      terminal: { cwd: '/new' }
    })
  })

  it('saves newly supported locales to display.language', async () => {
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })

    const configClient: I18nConfigClient = {
      getConfig: vi
        .fn()
        .mockResolvedValueOnce({ display: { language: 'en' } })
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'mono' } }),
      saveConfig
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe target="ja" />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(saveConfig).toHaveBeenCalledTimes(1))
    expect(saveConfig).toHaveBeenCalledWith({ display: { language: 'ja', skin: 'mono' } })
    expect(screen.getByTestId('locale').textContent).toBe('ja')
  })

  it('applies RTL direction for Arabic and restores LTR on switch back', async () => {
    render(
      <I18nProvider configClient={null} initialLocale="ar">
        <LanguageProbe target="en" />
      </I18nProvider>
    )

    expect(screen.getByTestId('locale').textContent).toBe('ar')
    expect(document.documentElement.dir).toBe('rtl')
    expect(document.documentElement.lang).toBe('ar')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('en'))
    expect(document.documentElement.dir).toBe('ltr')
    expect(document.documentElement.lang).toBe('en')
  })

  it('rolls back the visible locale when saving fails', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'en' } }),
      saveConfig: vi.fn().mockRejectedValue(new Error('save failed'))
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await waitFor(() => expect(screen.getByTestId('save-error').textContent).toBe('save failed'))

    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Language')
  })

  it('retries a transient config failure and applies the persisted locale', async () => {
    const getConfig = vi
      .fn()
      .mockRejectedValueOnce(new Error('backend not ready yet'))
      .mockResolvedValueOnce({ display: { language: 'zh-Hans' } })

    const configClient: I18nConfigClient = {
      getConfig,
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    // First attempt fails → settles on English (permanent-failure contract).
    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    expect(screen.getByTestId('locale').textContent).toBe('en')

    // The bounded retry succeeds and applies the persisted language.
    await waitFor(() => expect(screen.getByTestId('locale').textContent).toBe('zh'), { timeout: 5_000 })
    expect(getConfig).toHaveBeenCalledTimes(2)
  })

  it('stops retrying after the bounded retry budget is exhausted', async () => {
    vi.useFakeTimers()
    const getConfig = vi.fn().mockRejectedValue(new Error('backend unavailable'))

    const configClient: I18nConfigClient = {
      getConfig,
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLocale="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    // Flush the initial attempt: it fails and settles on English.
    await act(async () => {})
    expect(screen.getByTestId('locale').textContent).toBe('en')
    expect(getConfig).toHaveBeenCalledTimes(1)

    // Budget is 10 retries at 3s each; run the whole budget to completion.
    for (let i = 0; i < 10; i++) {
      await act(async () => {
        vi.advanceTimersByTime(3_000)
      })
    }
    expect(getConfig).toHaveBeenCalledTimes(11)

    // No timer is left after the budget is spent — nothing fires later.
    await act(async () => {
      vi.advanceTimersByTime(30_000)
    })
    expect(getConfig).toHaveBeenCalledTimes(11)

    vi.useRealTimers()
  })

  it('a late startup read never overrides a language the user picked mid-retry', async () => {
    vi.useFakeTimers()

    const getConfig = vi
      .fn()
      .mockRejectedValueOnce(new Error('backend not ready yet'))
      .mockResolvedValue({ display: { language: 'en' } })

    const configClient: I18nConfigClient = {
      getConfig,
      saveConfig: vi.fn().mockResolvedValue({ ok: true })
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe target="ja" />
      </I18nProvider>
    )

    await act(async () => {})
    expect(screen.getByTestId('locale').textContent).toBe('en')

    // User picks Japanese while the startup retry is still pending.
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'switch' }))
    })
    expect(screen.getByTestId('locale').textContent).toBe('ja')

    // The retry resolves with the stale on-disk value; the explicit pick wins.
    await act(async () => {
      vi.advanceTimersByTime(3_000)
    })
    expect(screen.getByTestId('locale').textContent).toBe('ja')

    vi.useRealTimers()
  })
})
