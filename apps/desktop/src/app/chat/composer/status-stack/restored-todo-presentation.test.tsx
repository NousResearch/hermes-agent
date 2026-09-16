import { act, cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest'

import { I18nProvider, type Locale } from '@/i18n'
import { createClientSessionState } from '@/lib/chat-runtime'
import { $sessionStates } from '@/store/session-states'
import { $preservedTodosBySession, $todoContinuationsBySession, $todosBySession, setSessionTodos } from '@/store/todos'
import { LEGACY_TODOS_277757 } from '@/test/evidence-row-277757'

import { ComposerStatusStack } from './index'

const SID = 'session-restored'

function renderStack(locale: Locale = 'en') {
  return render(
    <MemoryRouter>
      <I18nProvider configClient={null} initialLocale={locale}>
        <ComposerStatusStack queue={null} sessionId={SID} />
      </I18nProvider>
    </MemoryRouter>
  )
}

function statusRowFor(title: string): HTMLElement {
  const titleNode = screen.getByText(title)
  const row = titleNode.parentElement?.parentElement

  expect(row).toBeInstanceOf(HTMLElement)

  return row as HTMLElement
}

describe('ComposerStatusStack restored Todo presentation', () => {
  beforeAll(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        disconnect() {}
        observe() {}
      }
    )
  })

  afterEach(() => {
    cleanup()
    $sessionStates.set({})
    $preservedTodosBySession.set({})
    $todoContinuationsBySession.set({})
    $todosBySession.set({})
  })

  it.each([
    ['en', 'Restored unfinished work — Tasks 0/11'],
    ['ja', '復元した未完了の作業 — タスク 0/11'],
    ['zh', '已恢复的未完成工作 — 任务 0/11'],
    ['zh-hant', '已還原的未完成工作 — 任務 0/11'],
    ['ar', 'عمل غير مكتمل تمت استعادته — المهام 0/11']
  ] as const)('labels exact restored history as unfinished, non-live work in %s', (locale, expectedLabel) => {
    setSessionTodos(SID, LEGACY_TODOS_277757, { preserved: true })

    renderStack(locale)

    expect(screen.getByRole('button', { name: expectedLabel })).toBeTruthy()
    expect(screen.queryByRole('button', { name: /^Tasks 0\/11$/ })).toBeNull()
  })

  it('renders the exact restored in-progress row with a static unfinished glyph', () => {
    setSessionTodos(SID, LEGACY_TODOS_277757, { preserved: true })

    renderStack()

    const rowTitle = screen.getByText('Audit top-level campaign-related sessions missed by parent-session provenance')
    const row = statusRowFor(rowTitle.textContent!)

    expect.soft(row.querySelector('.codicon-circle-large-outline')).not.toBeNull()
    expect.soft(row.querySelector('[role="status"]')).toBeNull()
    expect.soft(row.querySelector('[class*="bg-emerald"]')).toBeNull()
    expect(rowTitle.className).toContain('text-foreground/92')
    expect(rowTitle.className).not.toContain('text-muted-foreground/75')
  })

  it('retains the working in-progress spinner', () => {
    $sessionStates.set({ [SID]: { ...createClientSessionState(), busy: true, turnLive: true } })
    $todosBySession.set({ [SID]: [{ content: 'Working task', id: 'working', status: 'in_progress' }] })

    renderStack()

    expect(screen.getByText('Running — Tasks 0/1')).toBeTruthy()
    expect(statusRowFor('Working task').querySelector('[role="status"]')).not.toBeNull()
  })

  it('retains static continuing and paused in-progress glyphs', () => {
    $todosBySession.set({ [SID]: [{ content: 'Continuing task', id: 'continuing', status: 'in_progress' }] })
    $todoContinuationsBySession.set({ [SID]: { revision: 7, state: 'active' } })

    renderStack()

    expect(screen.getByText('Goal active — Tasks 0/1')).toBeTruthy()
    expect(statusRowFor('Continuing task').querySelector('.codicon-debug-continue')).not.toBeNull()
    expect(screen.queryByRole('status')).toBeNull()

    act(() => {
      $todoContinuationsBySession.set({ [SID]: { revision: 8, state: 'paused' } })
    })

    expect(screen.getByText('Goal paused — Tasks 0/1')).toBeTruthy()
    expect(statusRowFor('Continuing task').querySelector('.codicon-debug-pause')).not.toBeNull()
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('retains pending, completed, and cancelled row semantics', () => {
    $sessionStates.set({ [SID]: { ...createClientSessionState(), busy: true, turnLive: true } })
    $todosBySession.set({
      [SID]: [
        { content: 'Pending task', id: 'pending', status: 'pending' },
        { content: 'Completed task', id: 'completed', status: 'completed' },
        { content: 'Cancelled task', id: 'cancelled', status: 'cancelled' }
      ]
    })

    renderStack()

    const pendingTitle = screen.getByText('Pending task')
    const completedTitle = screen.getByText('Completed task')
    const cancelledTitle = screen.getByText('Cancelled task')

    expect(statusRowFor('Pending task').querySelector('[class*="border-dashed"]')).not.toBeNull()
    expect(statusRowFor('Completed task').querySelector('.codicon-pass-filled')).not.toBeNull()
    expect(statusRowFor('Cancelled task').querySelector('.codicon-circle-slash')).not.toBeNull()
    expect(pendingTitle.className).toContain('text-muted-foreground/75')
    expect(completedTitle.className).toContain('text-muted-foreground/75')
    expect(cancelledTitle.className).toContain('text-muted-foreground/75')
  })
})
