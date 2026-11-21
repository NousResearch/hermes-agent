import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { I18nProvider } from '@/i18n'
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { TaskDrawer } from './drawer'
import { en, KANBAN_LOCALES } from './i18n'
import { ru } from './i18n-ru'
import { COLUMN_META } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

// The real error text from a crashed managed worker (board lead-gen-pipeline, 2026-11-04).
const RAW_ERROR =
  "pid 24788 exited with code 1 [managed worker lost: managed_worker_failed] Worker's last output: 'Session: local-fc77 Kanban owner refused: Kanban execution unknown; retained by owner'"

const detail = {
  task: {
    id: 't_example',
    title: 'Copywriter step',
    body: 'Пользовательский текст задачи не переводится.',
    status: 'blocked',
    assignee: 'gza',
    consecutive_failures: 2,
    last_failure_error: RAW_ERROR,
    diagnostics: [
      {
        kind: 'repeated_crashes',
        severity: 'error',
        title: 'Agent crashed 2x: pid 24788 exited with code 1',
        detail: 'Original English detail from the dispatcher.',
        actions: [],
        count: 1,
        last_seen_at: 1,
        data: { consecutive_crashes: 2, last_error: RAW_ERROR }
      }
    ]
  },
  comments: [],
  events: [],
  links: { parents: [], children: [] },
  runs: [{ id: 55, profile: 'gza', status: 'done', outcome: 'crashed', error: RAW_ERROR, started_at: 1, ended_at: 2 }],
  attachments: []
}

const rest = vi.fn(async (path: string): Promise<unknown> => {
  if (path === '/tasks/t_example') {
    return detail
  }

  if (path.startsWith('/tasks/t_example/log?')) {
    return { exists: false, content: '', size_bytes: 0, truncated: false }
  }

  if (path === '/profiles') {
    return { profiles: [] }
  }

  if (path === '/orchestration') {
    return { default_assignee: '' }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

const keys = (o: object, prefix = ''): string[] =>
  Object.entries(o).flatMap(([k, v]) =>
    v && typeof v === 'object' ? keys(v, `${prefix}${k}.`) : [`${prefix}${k}`]
  )

const configFor = (language: string) => ({ getConfig: async () => ({ display: { language } }), saveConfig: async () => ({ ok: true }) })

function openDrawer(locale: 'en' | 'ru') {
  return render(
    <I18nProvider configClient={configFor(locale)} initialLocale={locale}>
      <QueryClientProvider client={client}>
        <TaskDrawer columns={['todo', 'ready', 'blocked', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
      </QueryClientProvider>
    </I18nProvider>
  )
}

describe('kanban ru bundle', () => {
  it('has exactly the same message keys as en (typed parity)', () => {
    expect(keys(ru).sort()).toEqual(keys(en).sort())
  })

  it('translates all eight board columns (plus archive) with hints, keeping the status ids', () => {
    const expected: Record<string, string> = {
      triage: 'Разбор',
      todo: 'К выполнению',
      scheduled: 'Запланировано',
      ready: 'Готово к запуску',
      running: 'Выполняется',
      blocked: 'Заблокировано',
      review: 'На проверке',
      done: 'Завершено'
    }

    for (const [id, label] of Object.entries(expected)) {
      expect(ru.col[id as keyof typeof ru.col].label).toBe(label)
      expect(ru.col[id as keyof typeof ru.col].help.length).toBeGreaterThan(10)
    }

    expect(ru.col.blocked.help).toBe('Задача не может продолжить выполнение. Откройте диагностику, чтобы узнать причину.')
    expect(Object.keys(ru.col).sort()).toEqual(Object.keys(en.col).sort())
    // The status ids the dispatcher uses are not display strings and must stay untouched.
    expect(Object.keys(COLUMN_META)).toEqual(expect.arrayContaining(['triage', 'todo', 'ready', 'running', 'blocked', 'done']))
  })

  it('translates the task card field names', () => {
    expect(ru.assignee).toBe('Исполнитель')
    expect(ru.priority).toBe('Приоритет')
    expect(ru.consecutiveFailures).toBe('Ошибок подряд')
    expect(ru.lastFailureError).toBe('Последняя ошибка')
    expect(ru.workspace).toBe('Рабочая папка')
    expect(ru.model).toBe('Модель')
    expect(ru.blockedBy).toBe('Зависит от задачи')
    expect(ru.metaCreatedBy).toBe('Создал')
    expect(ru.metaCreated).toBe('Создано')
    expect(ru.estimate).toBe('Оценка')
    expect(ru.estimateEffort).toBe('Оценить сложность')
    expect(ru.attachments(2)).toBe('Вложения · 2')
    expect(ru.diagnosticsN(1)).toBe('Диагностика · 1')
  })

  it('translates actions, dialogs and notifications', () => {
    expect([ru.createTask, ru.cancel, ru.save, ru.archive, ru.unassignAction, ru.requeueWithNote, ru.createBoard]).toEqual([
      'Создать задачу',
      'Отмена',
      'Сохранить',
      'В архив',
      'Снять исполнителя',
      'Перезапустить с заметкой',
      'Создать доску'
    ])
    expect(ru.deleteBoardTitle('Лиды')).toBe('Удалить «Лиды»?')
    expect(ru.notify.gaveUpTitle).toBe('Задача остановлена')
    expect(ru.moveTo('Завершено')).toBe('Переместить в «Завершено»')
  })

  it('explains known errors in Russian without touching the original text', () => {
    const before = RAW_ERROR
    const out = ru.errorExplain(RAW_ERROR)

    expect(out).toContain('Ошибка управляемого воркера (managed_worker_failed)')
    expect(out).toContain('Владелец сессии отклонил запрос')
    expect(out).toContain('Процесс завершился с кодом 1')
    expect(RAW_ERROR).toBe(before)
    expect(ru.errorExplain('Agent crash x2')).toBe('Агент аварийно завершился 2 раза')
    expect(ru.errorExplain('Circuit breaker triggered')).toBe('Автоматические повторы остановлены после нескольких ошибок')
    expect(ru.errorExplain('some brand new backend text')).toBe('')
    expect(en.errorExplain(RAW_ERROR)).toBe('')
  })

  it('formats counters with Russian plural forms and keeps dates/ids out of translation', () => {
    expect(ru.diagTitle('repeated_crashes', '', 1, { consecutive_crashes: 1 })).toBe('Агент аварийно завершился 1 раз')
    expect(ru.diagTitle('repeated_crashes', '', 1, { consecutive_crashes: 2 })).toBe('Агент аварийно завершился 2 раза')
    expect(ru.diagTitle('repeated_crashes', '', 1, { consecutive_crashes: 5 })).toBe('Агент аварийно завершился 5 раз')
    expect(ru.diagTitle('repeated_crashes', '', 1, { consecutive_crashes: 22 })).toBe('Агент аварийно завершился 22 раза')
    expect(ru.countTip(2, 3)).toBe('Канбан — выполняется: 2, готово к запуску: 3')
    expect(ru.nSelected(5)).toBe('Выбрано: 5')
    expect(ru.diagTitle('unknown_kind', 't', 1, {})).toBe('')
    expect(ru.runOutcome('weird_status')).toBe('weird_status')
    expect(ru.copiedId('t_6d87fe18')).toBe('Скопировано: t_6d87fe18')
  })

  it('keeps long Russian labels short enough for narrow columns', () => {
    for (const col of Object.values(ru.col)) {
      expect(col.label.length).toBeLessThanOrEqual(16)
    }

    for (const label of [ru.moveToShort, ru.assign, ru.archive, ru.save, ru.cancel, ru.filters, ru.priority, ru.assignee]) {
      expect(label.length).toBeLessThanOrEqual(18)
    }
  })
})

describe('task card in the real component', () => {
  it('shows Russian title, explanation, collapsed technical details and keeps ids/profiles', async () => {
    const { container } = openDrawer('ru')

    expect(await screen.findByText('Агент аварийно завершился 2 раза')).toBeTruthy()
    expect(screen.getAllByText('Технические подробности').length).toBeGreaterThan(0)
    // The original diagnostic text is kept verbatim inside the collapsible block.
    const raw = Array.from(container.ownerDocument.querySelectorAll('details pre')).map(n => n.textContent ?? '')
    expect(raw.some(t => t.includes('Agent crashed 2x') && t.includes('Original English detail'))).toBe(true)
    expect(raw.some(t => t === RAW_ERROR)).toBe(true)
    // User data and technical identifiers are not translated.
    expect(screen.getByText('Пользовательский текст задачи не переводится.')).toBeTruthy()
    expect(screen.getAllByText('gza').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByText('Запуски · 1'))
    expect((await screen.findAllByText('Аварийно завершён')).length).toBeGreaterThan(0)
  })

  it('switches back to English without leaking Russian', async () => {
    openDrawer('en')

    expect(await screen.findByText('Agent crashed 2x: pid 24788 exited with code 1')).toBeTruthy()
    expect(screen.queryByText('Технические подробности')).toBeNull()
    expect(screen.queryByText('Агент аварийно завершился 2 раза')).toBeNull()
  })
})
