import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

// eslint-disable-next-line no-restricted-imports
import { I18nProvider, registerPluginLocales, translatePlugin } from '@/i18n'

import { columnLabel, KANBAN_LOCALES, useKanban } from './i18n'

afterEach(cleanup)

it('renders Korean board states and help without translating task identifiers or command syntax', () => {
  const dispose = registerPluginLocales('kanban', KANBAN_LOCALES)

  function BoardLabels() {
    const k = useKanban()

    return <div>{[k.newTask, k.col.ready.label, k.col.ready.help, columnLabel(k, 'future-state')].join(' | ')}</div>
  }

  try {
    render(
      <I18nProvider configClient={null} initialLocale="ko">
        <BoardLabels />
      </I18nProvider>
    )
    expect(screen.getByText(/새 작업 \| 준비/)).toBeTruthy()
    expect(screen.getByText(/future-state/)).toBeTruthy()
    expect(translatePlugin('kanban', 'ko', 'copiedId', ['task-한글-17'])).toBe('task-한글-17 복사됨')
    expect(translatePlugin('kanban', 'ko', 'projectHintCmd', [])).toBe('hermes project')
    expect(translatePlugin('kanban', 'ko', 'notify.blockedTitle', [])).toBe('작업이 막힘 — 입력이 필요합니다')
  } finally {
    dispose()
  }
})
