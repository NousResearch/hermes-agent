/**
 * The Sessions sidebar pane registers at module import, before `I18nProvider`
 * has fetched `display.language`, so its string `title` is whatever English
 * `translateNow` sampled then. The strip label must come from `tabTitle`,
 * which subscribes to the live locale.
 */
import type { ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { I18nProvider, setRuntimeI18nLocale } from '@/i18n'

const { registry } = await import('@/contrib/registry')

await import('./controller')

describe('the Sessions pane tab label', () => {
  it('follows the live locale instead of the register-time title', () => {
    const sessions = registry.getArea('panes').find(c => c.id === 'sessions')!
    const tabTitle = (sessions.data as { tabTitle: () => ReactNode }).tabTitle

    const inLocale = (locale: string) =>
      renderToStaticMarkup(
        <I18nProvider configClient={null} initialLocale={locale}>
          {tabTitle()}
        </I18nProvider>
      )

    expect(inLocale('ru')).not.toBe(inLocale('en'))
  })

  it('is joined by every other string-titled chrome pane, with a string twin for the zone menu / drag ghost', () => {
    const chrome = (id: string) =>
      registry.getArea('panes').find(c => c.id === id)!.data as {
        tabTitle: () => ReactNode
        tabTitleText: () => string
      }

    const inLocale = (id: string, locale: string) =>
      renderToStaticMarkup(
        <I18nProvider configClient={null} initialLocale={locale}>
          {chrome(id).tabTitle()}
        </I18nProvider>
      )

    for (const id of ['terminal', 'files']) {
      expect(inLocale(id, 'ru')).not.toBe(inLocale(id, 'en'))
    }

    const ruSessions = inLocale('sessions', 'ru')
    const ruFiles = inLocale('files', 'ru')

    // The non-React readers (zone menu Show/Hide rows, drag ghost chip) call
    // the string twin at menu-open / drag-start, so it follows the runtime
    // locale set after registration — not the English `title` sampled at import.
    setRuntimeI18nLocale('ru')

    try {
      expect(chrome('sessions').tabTitleText()).toBe(ruSessions)
      expect(chrome('files').tabTitleText()).toBe(ruFiles)
    } finally {
      setRuntimeI18nLocale('en')
    }
  })
})
