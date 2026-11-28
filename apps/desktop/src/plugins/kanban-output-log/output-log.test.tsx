/**
 * The "Output log" drawer section: what it asks the backend for, how it renders the answer,
 * and when it stops asking.
 *
 * The SDK's `useQuery` is mocked (the real UI components are kept) so the tests can both
 * drive the section with each `state` the route can answer and read back the query options
 * it chose — in particular that polling is tied to `running`, never always-on.
 */
import { useQuery } from '@hermes/plugin-sdk'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as Sdk from '@hermes/plugin-sdk'

import { bindOutputLogRest, OutputLogSection } from './output-log'

vi.mock('@hermes/plugin-sdk', async importOriginal => ({
  ...(await importOriginal<typeof Sdk>()),
  useQuery: vi.fn()
}))

const queryMock = vi.mocked(useQuery)

interface QueryOptions {
  enabled: boolean
  queryFn: () => Promise<unknown>
  queryKey: readonly unknown[]
  refetchInterval: false | number
  refetchOnWindowFocus: boolean
}

let options: QueryOptions | undefined
let requested: string[] = []
const refetch = vi.fn()

const OK = {
  entries: [
    { id: 1, kind: 'user', role: 'user', seq: 0, text: 'work kanban task t_1', ts: 1000 },
    { id: 2, kind: 'thinking', role: 'assistant', seq: 0, text: 'thinking hard', ts: 1001 },
    { id: 2, kind: 'text', role: 'assistant', seq: 1, text: 'reading the file', ts: 1001 },
    {
      id: 2,
      kind: 'tool_call',
      role: 'assistant',
      seq: 2,
      text: '{"path": "a.txt"}',
      tool_name: 'read_file',
      ts: 1001
    },
    { id: 3, kind: 'tool_result', role: 'tool', seq: 0, text: 'file contents', tool_name: 'read_file', ts: 1002 },
    { id: 4, kind: 'text', role: 'assistant', seq: 1, text: 'all done', ts: 1003 }
  ],
  has_earlier: false,
  last_at: 1003,
  next_before: 1,
  profile: 'coder',
  returned: 6,
  session_id: 'local-abcdef0123456789',
  started_at: 1000,
  state: 'ok',
  task_id: 't_1',
  task_status: 'running',
  total: 4
}

/** Point the mocked query at one canned answer, recording the options it was given. */
function respond(data: unknown, { error, isFetching = false }: { error?: unknown; isFetching?: boolean } = {}) {
  queryMock.mockImplementation((given: unknown) => {
    options = given as QueryOptions

    return { data, error, isFetching, refetch } as unknown as ReturnType<typeof useQuery>
  })
}

const section = (props: Partial<Parameters<typeof OutputLogSection>[0]> = {}) => (
  <OutputLogSection board="hermes-maintenance" running status="running" taskId="t_1" {...props} />
)

const toggle = () => screen.getByRole('button', { name: /Output log/i })

/** Drive the disclosure to a known state, whatever a previous test left in the module memo. */
function ensureOpen(open: boolean) {
  if ((toggle().getAttribute('aria-expanded') === 'true') !== open) {
    fireEvent.click(toggle())
  }
}

beforeEach(() => {
  requested = []
  options = undefined
  localStorage.clear()
  bindOutputLogRest(async path => {
    requested.push(path)

    return OK as never
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('OutputLogSection', () => {
  it('keeps polling while the card is running and stops once it is not', () => {
    respond(OK)
    const { rerender } = render(section({ running: true }))

    expect(options?.refetchInterval).toBe(5000)

    rerender(section({ running: false, status: 'done' }))

    expect(options?.refetchInterval).toBe(false)
  })

  it('fetches the final transcript once on running-to-done, not on initial done mount', () => {
    respond(OK)
    const view = render(section({ running: true }))
    expect(refetch).not.toHaveBeenCalled()

    view.rerender(section({ running: false, status: 'done' }))
    expect(refetch).toHaveBeenCalledTimes(1)
    view.rerender(section({ running: false, status: 'done' }))
    expect(refetch).toHaveBeenCalledTimes(1)

    view.unmount()
    render(section({ running: false, status: 'done' }))
    expect(refetch).toHaveBeenCalledTimes(1)
  })

  it('scopes the query to the card and never refetches on window focus', () => {
    respond(OK)
    render(section())

    expect(options?.queryKey).toEqual(['kanban-output-log', 'hermes-maintenance', 't_1'])
    expect(options?.refetchOnWindowFocus).toBe(false)
    expect(options?.enabled).toBe(true)
  })

  it("asks this plugin's own namespace for the board and a bounded tail", async () => {
    respond(OK)
    render(section())

    await options?.queryFn()

    expect(requested).toEqual(['/tasks/t_1/output?tail=200&board=hermes-maintenance'])
  })

  it('omits the board parameter when the drawer has not resolved a board', async () => {
    respond(OK)
    render(section({ board: null }))

    await options?.queryFn()

    expect(requested).toEqual(['/tasks/t_1/output?tail=200'])
  })

  it('summarises a running card in the collapsed row', () => {
    respond(OK)
    render(section({ running: true }))
    ensureOpen(false)

    expect(screen.getByText('Output log')).toBeDefined()
    expect(screen.getByText('running · 6 lines · 4 messages')).toBeDefined()
    expect(toggle().getAttribute('aria-expanded')).toBe('false')
  })

  it('names the card status instead of "running" once the worker is finished', () => {
    respond({ ...OK, task_status: 'done' }, {})
    render(section({ running: false, status: 'done' }))
    ensureOpen(false)

    expect(screen.getByText('done · 6 lines · 4 messages')).toBeDefined()
  })

  it('renders one feed line per entry kind when expanded', () => {
    respond(OK)
    render(section())
    ensureOpen(true)

    expect(document.querySelector('[data-slot="kanban-output-log"]')?.getAttribute('data-open')).toBe('true')
    const kinds = [...document.querySelectorAll('[data-kind]')].map(node => node.getAttribute('data-kind'))
    expect(kinds).toEqual(['user', 'thinking', 'text', 'tool_call', 'tool_result', 'text'])
    // Both the call and its result name the tool.
    expect(screen.getAllByText('read_file')).toHaveLength(2)
    expect(screen.getByText('{"path": "a.txt"}')).toBeDefined()
    expect(screen.getByText('coder')).toBeDefined()
  })

  it('says so plainly when the card has no worker session', () => {
    respond({ ...OK, entries: [], returned: 0, state: 'no_session', total: 0 })
    render(section({ running: false, status: 'todo' }))
    ensureOpen(false)

    expect(screen.getByText('No output yet — no worker has run for this card')).toBeDefined()

    ensureOpen(true)
    expect(
      screen.getByText('No output yet — this card has no worker session (queued, or never started).')
    ).toBeDefined()
  })

  it('reports an unreadable profile store without inventing content', () => {
    respond({ ...OK, entries: [], returned: 0, state: 'no_db', total: 0 })
    render(section())
    ensureOpen(false)

    expect(screen.getByText('Profile state.db unavailable')).toBeDefined()
  })

  it('surfaces a transport failure instead of rendering a blank panel', () => {
    respond(undefined, { error: new Error('backend unavailable') })
    render(section())
    ensureOpen(false)

    expect(screen.getByText('Output unavailable')).toBeDefined()

    ensureOpen(true)
    expect(screen.getByText('Could not load the worker output.')).toBeDefined()
  })

  it('pages backwards from the oldest line it is holding', async () => {
    respond({ ...OK, has_earlier: true, next_before: 1 })
    render(section())
    ensureOpen(true)

    requested = []
    fireEvent.click(screen.getByRole('button', { name: /Load earlier/i }))

    await waitFor(() => {
      expect(requested).toEqual(['/tasks/t_1/output?tail=200&board=hermes-maintenance&before=1'])
    })
  })

  it('offers no pager when the whole transcript already fits', () => {
    respond(OK)
    render(section())
    ensureOpen(true)

    expect((screen.getByRole('button', { name: /Load earlier/i }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('carries the disclosure state across unmount and remount', () => {
    respond(OK)
    const view = render(section())

    ensureOpen(false)
    view.unmount()
    render(section())
    expect(toggle().getAttribute('aria-expanded')).toBe('false')

    ensureOpen(true)
    cleanup()
    render(section())
    expect(toggle().getAttribute('aria-expanded')).toBe('true')
  })
})
