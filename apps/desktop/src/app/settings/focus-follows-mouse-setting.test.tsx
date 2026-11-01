import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { en } from '@/i18n/en'
import { zh } from '@/i18n/zh'

import { AppearanceSettings } from './appearance-settings'
import { settingDefinition, settingSearchTargets } from './settings-manifest'

const KEY = 'hermes.desktop.focusFollowsMouse'

beforeEach(() => window.localStorage.removeItem(KEY))
afterEach(cleanup)

function showSettings(subpage: 'chat-display' | 'window-layout' = 'window-layout', locale = 'en') {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <I18nProvider configClient={null} initialLocale={locale}>
        <AppearanceSettings subpage={subpage} />
      </I18nProvider>
    </QueryClientProvider>
  )
}

describe('Focus follows mouse setting', () => {
  it('lives beside Floating Composer in Window & Layout and persists either choice', () => {
    showSettings()
    const toggle = screen.getByRole('switch', { name: 'Focus follows mouse' })
    const floating = screen.getByRole('switch', { name: 'Floating Composer' })

    expect(floating.compareDocumentPosition(toggle) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-checked')).toBe('true')
    expect(window.localStorage.getItem(KEY)).toBe('true')
    fireEvent.click(toggle)
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    expect(window.localStorage.getItem(KEY)).toBe('false')
  })

  it('is not hidden among chat-display settings', () => {
    showSettings('chat-display')
    expect(screen.queryByRole('switch', { name: 'Focus follows mouse' })).toBeNull()
  })

  it('has Chinese copy and a searchable deep link to the same page', () => {
    showSettings('window-layout', 'zh')
    expect(screen.getByRole('switch', { name: '鼠标经过时自动切换会话' })).toBeTruthy()
    const english = settingSearchTargets(en).find(row => row.id === 'appearance.focus-follows-mouse')
    const chinese = settingSearchTargets(zh).find(row => row.id === 'appearance.focus-follows-mouse')
    expect(english).toMatchObject({ label: 'Focus follows mouse', view: 'config:appearance' })
    expect(chinese?.label).toBe('鼠标经过时自动切换会话')
    expect(english?.keywords).toContain('hover')
    expect(settingDefinition('config:appearance', 'appearance.focus-follows-mouse')?.subpage).toBe('window-layout')
    const toggle = screen.getByRole('switch', { name: '鼠标经过时自动切换会话' })
    expect(toggle.closest('#setting-field-appearance\\.focus-follows-mouse')).toBeTruthy()
  })
})
