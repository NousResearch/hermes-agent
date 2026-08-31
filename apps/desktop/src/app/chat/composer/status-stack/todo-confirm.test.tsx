import React, { act } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { TodoSnapshot } from '@/lib/todos'

const probe = vi.hoisted(() => ({
  authority: { generation: 2, revision: 2 } as { generation: number; revision: number } | null,
  confirmError: null as Error | null,
  snapshot: {
    generation: 2,
    revision: 2,
    session_id: 's1',
    todos: [{ content: 'Build tray', id: 'raw-id', status: 'in_progress' }]
  } as TodoSnapshot | null
}))

vi.mock('@nanostores/react', () => ({ useStore: () => false }))
vi.mock('@/app/chat/composer/focus', () => ({ blurComposerInput: vi.fn() }))
vi.mock('@/app/routes', () => ({ AGENTS_ROUTE: '/agents' }))
vi.mock('@/components/billing-banner', () => ({ BillingBanner: () => null }))
vi.mock('@/components/chat/composer-dock', () => ({ composerDockCard: () => '' }))
vi.mock('@/components/chat/status-section', () => ({
  StatusSection: ({
    accessory,
    children,
    label
  }: {
    accessory?: React.ReactNode
    children: React.ReactNode
    label: React.ReactNode
  }) => (
    <section>
      <h2>{label}</h2>
      {accessory}
      {children}
    </section>
  )
}))
vi.mock('@/components/ui/button', () => ({
  Button: (props: React.ButtonHTMLAttributes<HTMLButtonElement>) => <button {...props} />
}))
vi.mock('@/components/ui/codicon', () => ({ Codicon: () => null }))
vi.mock('@/components/ui/glyph-spinner', () => ({ GlyphSpinner: () => null }))
vi.mock('@/components/ui/tooltip', () => ({
  Tip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TipKeybindLabel: () => null
}))
vi.mock('@/components/ui/confirm-dialog', () => ({
  ConfirmDialog: ({
    open,
    onConfirm,
    title
  }: {
    open: boolean
    onConfirm: () => Promise<void>
    title: React.ReactNode
  }) =>
    open ? (
      <div>
        <span>{title}</span>
        <button
          aria-label="dialog-confirm"
          onClick={() => {
            void onConfirm().catch(error => {
              probe.confirmError = error instanceof Error ? error : new Error(String(error))
            })
          }}
        >
          Confirm
        </button>
      </div>
    ) : null
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      statusStack: {
        agents: 'Agents',
        background: (n: number) => `${n} Background`,
        goalActive: '',
        goalDone: '',
        goalPaused: '',
        goalWaiting: '',
        subagents: (n: number) => `${n} Subagents`,
        todos: (done: number, total: number) => `Tasks ${done}/${total}`,
        markDone: 'Mark done',
        reopen: 'Reopen',
        markingDone: '',
        reopening: '',
        markedDone: '',
        reopened: '',
        markDoneDescription: '',
        reopenDescription: '',
        markDoneTitle: 'Mark task done?',
        reopenTitle: 'Reopen task?',
        retryTaskSync: 'Retry task sync',
        taskChanged: 'changed',
        taskMissing: 'missing',
        taskSessionMissing: 'session',
        taskSyncFailed: 'Could not sync task status',
        taskUpdateFailed: 'failed',
        taskUpdatesUnavailable: 'Task updates unavailable',
        agentsLabel: '',
        running: 'Running'
      }
    }
  })
}))
vi.mock('@/lib/use-session-slice', () => ({
  useSessionSlice: (store: unknown) =>
    store === 'status-store'
      ? [
          {
            id: 'todo:raw-id',
            state: 'running',
            title: 'Build tray',
            todoItemId: 'raw-id',
            todoStatus: 'in_progress',
            type: 'todo'
          }
        ]
      : [],
  useStoreSelector: () => probe.authority
}))
vi.mock('@/lib/utils', () => ({ cn: (...values: unknown[]) => values.filter(Boolean).join(' ') }))
vi.mock('@/store/billing-block', () => ({ $billingBlock: 'billing' }))
vi.mock('@/store/composer-status', () => ({
  $statusItemsBySession: 'status-store',
  dismissBackgroundProcess: vi.fn(),
  groupStatusItems: (items: unknown[]) => [{ items, type: 'todo' }],
  refreshBackgroundProcesses: vi.fn(),
  resetBackgroundPollingGuard: vi.fn(),
  stopBackgroundProcess: vi.fn()
}))
vi.mock('@/store/goals', () => ({ refreshSessionGoal: vi.fn() }))
vi.mock('@/store/preview-status', () => ({ $previewStatusBySession: 'preview-store', dismissPreviewArtifact: vi.fn() }))
vi.mock('@/store/thread-scroll', () => ({ $threadScrolledUp: 'scroll' }))
vi.mock('@/store/windows', () => ({ openSessionInNewWindow: vi.fn() }))
vi.mock('@/store/todos', () => ({
  $sessionTodoSnapshots: 'authority-store',
  applyOptimisticTodoStatus: vi.fn(() => true),
  currentSessionTodoSnapshot: () => probe.snapshot,
  setSessionTodoSnapshot: vi.fn((next: TodoSnapshot) => {
    probe.snapshot = next
    probe.authority = { generation: next.generation, revision: next.revision }
  })
}))
vi.mock('./preview-row', () => ({ PreviewStatusRow: () => null }))
vi.mock('./status-row', () => ({
  StatusItemRow: ({
    item,
    onTodoAction,
    todoMutationEnabled,
    todoMutationUnavailableLabel
  }: {
    item: unknown
    onTodoAction: (item: unknown, button: HTMLButtonElement) => void
    todoMutationEnabled: boolean
    todoMutationUnavailableLabel: string
  }) => (
    <button
      aria-label={todoMutationEnabled ? 'todo-row-action' : todoMutationUnavailableLabel}
      disabled={!todoMutationEnabled}
      onClick={event => onTodoAction(item, event.currentTarget)}
    >
      Task action
    </button>
  )
}))

const { ComposerStatusStack } = await import('./index')

beforeAll(() => {
  ;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
})

beforeEach(() => {
  probe.authority = { generation: 2, revision: 2 }
  probe.confirmError = null
  probe.snapshot = {
    generation: 2,
    revision: 2,
    session_id: 's1',
    todos: [{ content: 'Build tray', id: 'raw-id', status: 'in_progress' }]
  }
})

afterEach(() => {
  globalThis.document.body.innerHTML = ''
})

describe('ComposerStatusStack confirmed todo mutation', () => {
  it('does not send update before confirmation and then sends the exact CAS payload', async () => {
    const request = vi.fn(async (method: string) =>
      method === 'todo.snapshot'
        ? probe.snapshot
        : {
            ...probe.snapshot!,
            generation: 3,
            revision: 3,
            todos: [{ ...probe.snapshot!.todos[0], status: 'completed' }]
          }
    )

    const container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ComposerStatusStack queue={null} requestGateway={request as never} sessionId="s1" />
        </MemoryRouter>
      )
    })
    await act(async () => {})
    expect(request.mock.calls.filter(([method]) => method === 'todo.update_status')).toHaveLength(0)

    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="todo-row-action"]')!.click())
    expect(container.textContent).toContain('Mark task done?')
    expect(request.mock.calls.filter(([method]) => method === 'todo.update_status')).toHaveLength(0)

    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="dialog-confirm"]')!.click())
    expect(request).toHaveBeenCalledWith('todo.update_status', {
      actor: 'user',
      expected_revision: 2,
      item_id: 'raw-id',
      session_id: 's1',
      status: 'completed'
    })
    await act(async () => root.unmount())
  })

  it('does not retarget an open confirmation after an authoritative update', async () => {
    const request = vi.fn(async (_method: string) => probe.snapshot)
    const container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ComposerStatusStack queue={null} requestGateway={request as never} sessionId="s1" />
        </MemoryRouter>
      )
    })
    await act(async () => {})
    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="todo-row-action"]')!.click())

    probe.authority = { generation: 3, revision: 3 }
    probe.snapshot = {
      generation: 3,
      revision: 3,
      session_id: 's1',
      todos: [{ content: 'Build tray', id: 'raw-id', status: 'in_progress' }]
    }

    await act(async () => container.querySelector<HTMLButtonElement>('[aria-label="dialog-confirm"]')!.click())
    await act(async () => {})

    expect(request.mock.calls.filter(([method]) => method === 'todo.update_status')).toHaveLength(0)
    expect(probe.confirmError?.message).toBe('changed')
    await act(async () => root.unmount())
  })

  it('distinguishes an unsupported gateway from an in-progress sync', async () => {
    probe.authority = null
    const request = vi.fn().mockRejectedValue(Object.assign(new Error('missing method'), { code: -32601 }))
    const container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ComposerStatusStack queue={null} requestGateway={request as never} sessionId="s1" />
        </MemoryRouter>
      )
    })
    await act(async () => {})

    expect(container.querySelector<HTMLButtonElement>('[aria-label="Task updates unavailable"]')?.disabled).toBe(true)
    expect(container.textContent).not.toContain('Retry task sync')
    await act(async () => root.unmount())
  })

  it('surfaces a transport failure and lets the user retry snapshot sync', async () => {
    probe.authority = null
    const request = vi.fn(async (_method: string) => probe.snapshot)
    request.mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(probe.snapshot)
    const container = globalThis.document.createElement('div')
    globalThis.document.body.append(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(
        <MemoryRouter>
          <ComposerStatusStack queue={null} requestGateway={request as never} sessionId="s1" />
        </MemoryRouter>
      )
    })
    await act(async () => {})

    expect(container.querySelector<HTMLButtonElement>('[aria-label="Could not sync task status"]')?.disabled).toBe(true)

    const retry = Array.from(container.querySelectorAll('button')).find(button =>
      button.textContent?.includes('Retry task sync')
    )

    expect(retry).toBeDefined()

    await act(async () => retry!.click())
    await act(async () => {})

    expect(request.mock.calls.filter(([method]) => method === 'todo.snapshot')).toHaveLength(2)
    expect(container.querySelector<HTMLButtonElement>('[aria-label="todo-row-action"]')?.disabled).toBe(false)
    await act(async () => root.unmount())
  })
})
