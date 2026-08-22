import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'

import '../../styles.css'
import './kanban.css'
import { bindApi } from './api'
import { KanbanBoardPage } from './board'
import { TaskDrawer } from './drawer'

const task = { assignee: 'canary-worker', attention: { reason: 'receipt', revision: 0, state: 'active', wake_at: null }, id: 'task-safe', status: 'running', title: 'Privacy-safe evidence task' }
const board = { assignees: ['canary-worker'], columns: [{ name: 'running', tasks: [task] }, { name: 'review', tasks: [{ ...task, id: 'task-review', status: 'review', title: 'Production review task' }] }], latest_event_id: 2, now: 1_800_000_000, tenants: [] }
const detail = { attachments: [], comments: [], events: [], links: { children: [], parents: [] }, runs: [], task }

const rest = async <T,>(path: string): Promise<T> => {
  if (path.startsWith('/tasks/task-safe/log')) return { content: '', exists: false, size_bytes: 0, truncated: false } as T
  if (path.startsWith('/tasks/task-safe')) return detail as T
  if (path.startsWith('/boards')) return { boards: [{ default_workspace_kind: 'scratch', slug: 'default' }], current: 'default' } as T
  if (path.startsWith('/board')) return board as T
  if (path.startsWith('/profiles')) return { profiles: [] } as T
  if (path.startsWith('/projects')) return { projects: [] } as T
  throw new Error(`Unexpected responsive evidence request: ${path}`)
}

bindApi(rest, { get: <T,>(_key: string, fallback: T) => fallback, remove: () => undefined, set: () => undefined }, () => () => undefined)

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
    <main className="relative h-screen w-screen overflow-hidden" data-production-surface="desktop">
      <KanbanBoardPage />
      <TaskDrawer columns={['running', 'review']} id="task-safe" onClose={() => undefined} onOpen={() => undefined} />
    </main>
  </QueryClientProvider>
)
