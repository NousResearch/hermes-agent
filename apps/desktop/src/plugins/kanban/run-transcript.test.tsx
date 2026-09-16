/**
 * Run history → agent session transcript.
 *
 * Every dispatcher-spawned worker is a real, resumable Hermes session whose id
 * the worker records as `worker_session_id` in its run metadata. The REST layer
 * already ships that metadata to the client; these tests pin that the drawer
 * surfaces it as a click-through instead of dropping it on the floor.
 */
import type { PluginRestOptions } from '@hermes/plugin-sdk'
import type * as pluginSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { TaskDrawer, workerSessionId } from './drawer'
import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanRun, KanbanTaskDetail } from './types'

// `vi.mock` is hoisted above every top-level binding, so the spy has to be
// created inside `vi.hoisted` or the factory closes over a TDZ reference.
const { openSession } = vi.hoisted(() => ({ openSession: vi.fn(async () => undefined) }))

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

// `host` is a live singleton on the SDK surface; spread the real module so
// every other export (Badge, Tip, useQuery, …) keeps working and only the
// navigation call is observable.
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const actual = await importOriginal<typeof pluginSdk>()

  return { ...actual, host: { ...actual.host, openSession } }
})

const SESSION = '20260916_112320_531100'

const run = (over: Partial<KanbanRun> = {}): KanbanRun => ({
  id: 8,
  status: 'done',
  outcome: 'completed',
  started_at: 1_789_000_000,
  ended_at: 1_789_000_600,
  ...over
})

const baseDetail: KanbanTaskDetail = {
  task: { id: 't_example', title: 'Example task', body: 'Body', status: 'review' },
  comments: [],
  events: [],
  links: { parents: [], children: [] },
  attachments: [],
  runs: []
}

let detail: object
let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => {
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

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
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

function openDrawer() {
  return render(
    <QueryClientProvider client={client}>
      <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
    </QueryClientProvider>
  )
}

describe('workerSessionId', () => {
  // SQLite stores run metadata as TEXT. Depending on the path it reaches the
  // client already parsed or still a JSON string — both must work, and that
  // asymmetry is the whole reason this helper exists.
  it('reads the id from an already-parsed metadata object', () => {
    expect(workerSessionId(run({ metadata: { worker_session_id: SESSION } }))).toBe(SESSION)
  })

  it('reads the id from a JSON metadata string', () => {
    expect(workerSessionId(run({ metadata: JSON.stringify({ worker_session_id: SESSION }) }))).toBe(SESSION)
  })

  it.each([
    ['absent metadata', undefined],
    ['null metadata', null],
    ['unparseable JSON', '{not json'],
    ['a JSON scalar rather than an object', '42'],
    ['metadata without the key', { branch: 'wt/t_example' }],
    ['a non-string id', { worker_session_id: 12345 }],
    ['a blank id', { worker_session_id: '   ' }]
  ])('returns null for %s', (_label, metadata) => {
    expect(workerSessionId(run({ metadata: metadata as KanbanRun['metadata'] }))).toBeNull()
  })
})

describe('run history transcript link', () => {
  it('opens the recorded session in a tab when the transcript button is clicked', async () => {
    detail = { ...baseDetail, runs: [run({ metadata: { worker_session_id: SESSION } })] }
    openDrawer()

    const button = await screen.findByRole('button', { name: en.openTranscript })
    fireEvent.click(button)

    await waitFor(() => expect(openSession).toHaveBeenCalledWith(SESSION, { intent: 'tab' }))
  })

  it('renders no transcript control for a run that never recorded a session', async () => {
    detail = { ...baseDetail, runs: [run({ metadata: { branch: 'wt/t_example' } })] }
    openDrawer()

    // The run itself still renders — only the link is absent.
    expect(await screen.findByText('completed')).toBeTruthy()
    expect(screen.queryByRole('button', { name: en.openTranscript })).toBeNull()
  })

  it('links each run independently when a task was dispatched more than once', async () => {
    detail = {
      ...baseDetail,
      runs: [
        run({ id: 8, metadata: { worker_session_id: SESSION } }),
        run({ id: 11, outcome: 'reclaimed', metadata: { prev_pid: 26355 } })
      ]
    }
    openDrawer()

    await waitFor(() => expect(screen.getAllByRole('button', { name: en.openTranscript })).toHaveLength(1))
  })
})
