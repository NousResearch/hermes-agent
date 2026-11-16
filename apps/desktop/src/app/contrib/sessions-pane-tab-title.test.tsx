/**
 * The Sessions sidebar pane registers at module import, before `I18nProvider`
 * has fetched `display.language`, so its string `title` is whatever English
 * `translateNow` sampled then. The strip label must come from `tabTitle`,
 * which subscribes to the live locale.
 */
import { cleanup, render } from '@testing-library/react'
import type { ReactNode } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider, setRuntimeI18nLocale, TRANSLATIONS } from '@/i18n'
import { $selectedStoredSessionId, $sessions } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import type { SessionInfo } from '@/types/hermes'

const { registry } = await import('@/contrib/registry')

await import('./controller')

function untitledRow(id: string): SessionInfo {
  return {
    cwd: null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    last_active: 1,
    message_count: 1,
    model: null,
    output_tokens: 0,
    parent_session_id: null,
    preview: null,
    source: 'desktop',
    started_at: 1,
    title: null,
    tool_call_count: 0
  }
}

describe('the Sessions pane tab label', () => {
  afterEach(() => {
    cleanup()
    $selectedStoredSessionId.set(null)
    $sessions.set([])
    $sessionTiles.set([])
    setRuntimeI18nLocale('en')
  })

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

  it('re-titles the workspace pane and open session tiles when the runtime locale changes', () => {
    const pane = (id: string) => registry.getArea('panes').find(c => c.id === id)!

    // Fresh workspace draft: tabTitle follows I18nProvider, and title follows runtime locale.
    const workspaceTabTitle = (pane('workspace').data as { tabTitle?: () => ReactNode }).tabTitle
    expect(workspaceTabTitle).toBeDefined()

    const { container } = render(
      <I18nProvider configClient={null} initialLocale="es">
        {workspaceTabTitle?.()}
      </I18nProvider>
    )

    expect(container.textContent).toBe(TRANSLATIONS.es.sidebar.nav['new-session'])

    setRuntimeI18nLocale('es')
    expect(pane('workspace').title).toBe(TRANSLATIONS.es.sidebar.nav['new-session'])

    setRuntimeI18nLocale('en')
    expect(pane('workspace').title).toBe(TRANSLATIONS.en.sidebar.nav['new-session'])

    // Untitled stored session in workspace updates its registered title on locale change.
    $sessions.set([untitledRow('untitled-main')])
    $selectedStoredSessionId.set('untitled-main')
    expect(pane('workspace').title).toBe(TRANSLATIONS.en.sidebar.row.untitledPlaceholder)

    setRuntimeI18nLocale('es')
    expect(pane('workspace').title).toBe(TRANSLATIONS.es.sidebar.row.untitledPlaceholder)

    // Session tiles (draft + untitled stored) re-sync their registered titles on locale change.
    $sessions.set([untitledRow('untitled-main'), untitledRow('untitled-tile')])
    $sessionTiles.set([
      { dir: 'right', storedSessionId: 'draft-tile' },
      { dir: 'right', storedSessionId: 'untitled-tile' }
    ])

    expect(pane('session-tile:draft-tile').title).toBe(TRANSLATIONS.es.sidebar.nav['new-session'])
    expect(pane('session-tile:untitled-tile').title).toBe(TRANSLATIONS.es.sidebar.row.untitledPlaceholder)

    setRuntimeI18nLocale('en')
    expect(pane('session-tile:draft-tile').title).toBe(TRANSLATIONS.en.sidebar.nav['new-session'])
    expect(pane('session-tile:untitled-tile').title).toBe(TRANSLATIONS.en.sidebar.row.untitledPlaceholder)
  })
})
