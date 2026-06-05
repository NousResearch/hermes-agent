import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { type I18nConfigClient, I18nProvider, useI18n, useTranslation } from './context'

function LanguageProbe() {
  const { isLoadingConfig, isSavingLanguage, language, saveError, setLanguage } = useI18n()
  const t = useTranslation()

  return (
    <div>
      <p data-testid="language">{language}</p>
      <p data-testid="label">{t('common.save')}</p>
      <p data-testid="loading">{String(isLoadingConfig)}</p>
      <p data-testid="saving">{String(isSavingLanguage)}</p>
      <p data-testid="save-error">{saveError?.message ?? ''}</p>
      <button onClick={() => void setLanguage('zh').catch(() => undefined)} type="button">
        switch
      </button>
    </div>
  )
}

function BrokenProbe() {
  useI18n()

  return null
}

describe('I18nProvider', () => {
  afterEach(() => {
    cleanup()
  })

  it('provides English translations by default', () => {
    render(
      <I18nProvider configClient={null}>
        <LanguageProbe />
      </I18nProvider>
    )

    expect(screen.getByTestId('language').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Save')
  })

  it('normalizes the initial language and re-renders translations after language changes', () => {
    render(
      <I18nProvider configClient={null} initialLanguage="zh-CN">
        <LanguageProbe />
      </I18nProvider>
    )

    expect(screen.getByTestId('language').textContent).toBe('zh')
    expect(screen.getByTestId('label').textContent).toBe('保存')

    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    expect(screen.getByTestId('language').textContent).toBe('zh')
    expect(screen.getByTestId('label').textContent).toBe('保存')
  })

  it('throws a clear error when the hook is used outside the provider', () => {
    expect(() => render(<BrokenProbe />)).toThrow('useI18n must be used within I18nProvider')
  })

  it('loads the initial language from display.language config', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'zh-CN' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await screen.findByText('zh')

    expect(screen.getByTestId('label').textContent).toBe('保存')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('keeps English usable when config loading fails', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockRejectedValue(new Error('config unavailable')),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLanguage="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('language').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Save')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('displays English for unsupported configured languages without overwriting config', async () => {
    const configClient: I18nConfigClient = {
      getConfig: vi.fn().mockResolvedValue({ display: { language: 'ja' } }),
      saveConfig: vi.fn()
    }

    render(
      <I18nProvider configClient={configClient} initialLanguage="zh">
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))

    expect(screen.getByTestId('language').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Save')
    expect(configClient.saveConfig).not.toHaveBeenCalled()
  })

  it('reads latest config before saving language and preserves unrelated values', async () => {
    const saveConfig = vi.fn().mockResolvedValue({ ok: true })

    const configClient: I18nConfigClient = {
      getConfig: vi
        .fn()
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'mono' }, terminal: { cwd: '/old' } })
        .mockResolvedValueOnce({ display: { language: 'en', skin: 'slate' }, terminal: { cwd: '/new' } }),
      saveConfig
    }

    render(
      <I18nProvider configClient={configClient}>
        <LanguageProbe />
      </I18nProvider>
    )

    await waitFor(() => expect(screen.getByTestId('loading').textContent).toBe('false'))
    fireEvent.click(screen.getByRole('button', { name: 'switch' }))

    await screen.findByText('zh')

    expect(saveConfig).toHaveBeenCalledWith({
      display: { language: 'zh', skin: 'slate' },
      terminal: { cwd: '/new' }
    })
  })

  it('rolls back the visible language when saving fails', async () => {
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

    await screen.findByText('save failed')

    expect(screen.getByTestId('language').textContent).toBe('en')
    expect(screen.getByTestId('label').textContent).toBe('Save')
  })
})
