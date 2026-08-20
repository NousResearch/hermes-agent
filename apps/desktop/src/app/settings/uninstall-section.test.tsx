import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { type I18nConfigClient, I18nProvider } from '@/i18n'

import { UninstallSection } from './uninstall-section'

describe('UninstallSection', () => {
  afterEach(() => {
    cleanup()
  })

  const enConfigClient: I18nConfigClient = {
    getConfig: async () => ({ display: { language: 'en', skin: 'slate' } }),
    saveConfig: async () => ({ ok: true }),
  }

  it('renders danger zone heading in English', () => {
    render(
      <I18nProvider configClient={enConfigClient}>
        <UninstallSection />
      </I18nProvider>
    )
    expect(screen.getByText('Danger zone')).toBeTruthy()
  })

  it('renders uninstall modes in English', () => {
    render(
      <I18nProvider configClient={enConfigClient}>
        <UninstallSection />
      </I18nProvider>
    )
    expect(screen.getByText('Uninstall Hermes')).toBeTruthy()
    expect(screen.getByText('Uninstall Chat GUI only')).toBeTruthy()
    expect(screen.getByText('Uninstall GUI + agent, keep my data')).toBeTruthy()
    expect(screen.getByText('Uninstall everything')).toBeTruthy()
  })

  it('renders danger zone heading in Chinese', () => {
    const zhClient: I18nConfigClient = {
      getConfig: async () => ({ display: { language: 'zh', skin: 'slate' } }),
      saveConfig: async () => ({ ok: true }),
    }
    render(
      <I18nProvider configClient={zhClient}>
        <UninstallSection />
      </I18nProvider>
    )
    expect(screen.getByText('危险区域')).toBeTruthy()
  })

  it('renders uninstall modes in Chinese', () => {
    const zhClient: I18nConfigClient = {
      getConfig: async () => ({ display: { language: 'zh', skin: 'slate' } }),
      saveConfig: async () => ({ ok: true }),
    }
    render(
      <I18nProvider configClient={zhClient}>
        <UninstallSection />
      </I18nProvider>
    )
    expect(screen.getByText('卸载 Hermes')).toBeTruthy()
    expect(screen.getByText('仅卸载聊天界面')).toBeTruthy()
    expect(screen.getByText('卸载界面与代理，保留数据')).toBeTruthy()
    expect(screen.getByText('全部卸载')).toBeTruthy()
  })
})
