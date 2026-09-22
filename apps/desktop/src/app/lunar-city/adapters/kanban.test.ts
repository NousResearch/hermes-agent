import { atom } from 'nanostores'
import { describe, expect, it, vi } from 'vitest'

import type { PluginSourceScope } from '@/api/plugins'
import type { DesktopAgentRoster } from '@/global'

import actualManifest from '../../../../public/lunar-city/v2/world-manifest.v2.json'
import { entityKey } from '../identity'
import { parseWorldManifest } from '../manifest'
import type { ProjectSlotManifestEntry, WorldManifestV2 } from '../model'

import {
  allocateProjectCompounds,
  compoundKey,
  createKanbanCitySource,
  createRegisteredKanbanCitySource,
  type KanbanCitySourceOptions,
  kanbanDetailKey,
  type KanbanRest,
  type KanbanSocket,
  projectSlotContractIssues
} from './kanban'

const scopeA: PluginSourceScope = { connectionId: 'source-a', profile: 'default' }
const scopeB: PluginSourceScope = { connectionId: 'source-b', profile: 'default' }
const scopeAResearch: PluginSourceScope = { connectionId: 'source-a', profile: 'research' }

function manifest(): WorldManifestV2 {
  return parseWorldManifest(structuredClone(actualManifest))
}

function httpError(status: number): Error & { status: number } {
  return Object.assign(new Error(`HTTP ${status}`), { status })
}

function restFrom(responses: Record<string, unknown | Error>): ReturnType<typeof vi.fn> & KanbanRest {
  return vi.fn(async (path: string, _options) => {
    const response = responses[path]

    if (response instanceof Error) {
      throw response
    }

    if (response === undefined) {
      throw new Error(`unexpected request ${path}`)
    }

    return response as never
  })
}

function boardFixtures(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    '/boards': {
      boards: [{ is_current: true, project_id: 'project-alpha', slug: 'main' }],
      current: 'main'
    },
    '/board?board=main': {
      columns: [
        {
          name: 'running',
          tasks: [
            {
              current_run_id: 7,
              id: 'task-1',
              project_id: 'project-alpha',
              status: 'running',
              title: 'Build approved city'
            }
          ]
        }
      ],
      latest_event_id: 8,
      now: 12_345
    },
    '/workers/active?board=main': {
      checked_at: 12_345,
      count: 1,
      workers: [
        {
          run_id: 7,
          task_id: 'task-1',
          task_assignee: 'display-assignee',
          task_status: 'running',
          task_title: 'Build approved city',
          worker_pid: 4242
        }
      ]
    },
    ...overrides
  }
}

describe('Kanban Lunar City optional source', () => {
  it('fails closed when the optional Kanban plugin API is unavailable', async () => {
    const rest = restFrom({ '/boards': httpError(404) })
    const source = createKanbanCitySource({ manifest: manifest(), now: () => 100, rest, scope: scopeA })

    const result = await source.read()

    expect(result.authoritative).toBe(false)
    expect(result.health).toBe('unavailable')
    expect(result.entities).toEqual([])
    expect(result.compounds).toEqual([])
    expect(result.sources).toEqual([
      { authority: 'unknown', error: 'Kanban plugin unavailable', observedAt: 100, source: 'kanban:source-a:default' }
    ])
  })

  it('does not turn a malformed bounded worker response into an authoritative empty board', async () => {
    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 101,
      rest: restFrom(boardFixtures({ '/workers/active?board=main': { checked_at: 101 } })),
      scope: scopeA
    })

    await expect(source.read()).resolves.toMatchObject({
      authoritative: false,
      entities: [],
      health: 'malformed',
      sources: [
        {
          authority: 'unknown',
          error: 'Kanban response malformed',
          observedAt: 101,
          source: 'kanban:source-a:default'
        }
      ]
    })
  })

  it.each([
    [
      'an object in place of the columns array',
      boardFixtures({ '/board?board=main': { columns: {}, latest_event_id: 8 } })
    ],
    [
      'more than the bounded number of columns',
      boardFixtures({
        '/board?board=main': {
          columns: Array.from({ length: 65 }, (_, index) => ({ name: `column-${index}`, tasks: [] })),
          latest_event_id: 8
        }
      })
    ],
    [
      'an object in place of a column task array',
      boardFixtures({ '/board?board=main': { columns: [{ name: 'running', tasks: {} }], latest_event_id: 8 } })
    ],
    [
      'a task with a non-string ID',
      boardFixtures({
        '/board?board=main': {
          columns: [{ name: 'running', tasks: [{ id: 7, status: 'running' }] }],
          latest_event_id: 8
        }
      })
    ],
    [
      'more than the bounded number of task rows',
      boardFixtures({
        '/board?board=main': {
          columns: [
            {
              name: 'running',
              tasks: Array.from({ length: 513 }, (_, index) => ({ id: `task-${index}`, status: 'running' }))
            }
          ],
          latest_event_id: 8
        }
      })
    ],
    [
      'a worker with a non-string task ID',
      boardFixtures({
        '/workers/active?board=main': {
          workers: [{ run_id: 7, task_id: 1, task_status: 'running', worker_pid: 4242 }]
        }
      })
    ],
    [
      'more than the bounded number of worker rows',
      boardFixtures({
        '/workers/active?board=main': {
          workers: Array.from({ length: 513 }, (_, index) => ({
            run_id: index,
            task_id: `task-${index}`,
            task_status: 'running',
            worker_pid: index + 1
          }))
        }
      })
    ]
  ])('fails closed for %s instead of authoritatively replacing prior Kanban rows', async (_label, responses) => {
    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 102,
      rest: restFrom(responses),
      scope: scopeA
    })

    await expect(source.read()).resolves.toMatchObject({
      authoritative: false,
      entities: [],
      health: 'malformed',
      sources: [expect.objectContaining({ authority: 'unknown', error: 'Kanban response malformed' })]
    })
  })

  it('performs only the bounded board and active-worker reads on initial load', async () => {
    const rest = restFrom(boardFixtures())
    const source = createKanbanCitySource({ manifest: manifest(), now: () => 200, rest, scope: scopeA })

    const result = await source.read()

    expect(rest.mock.calls.map(call => call[0])).toEqual(['/boards', '/board?board=main', '/workers/active?board=main'])
    expect(rest.mock.calls.every(call => call[1]?.method === 'GET')).toBe(true)
    expect(rest.mock.calls.every(call => call[1]?.scope.connectionId === 'source-a')).toBe(true)
    expect(rest.mock.calls.every(call => call[1]?.scope.profile === 'default')).toBe(true)
    expect(result.authoritative).toBe(true)
    expect(result.selectedBoard).toBe('main')
    expect(result.sources).toEqual([{ authority: 'authoritative', observedAt: 200, source: 'kanban:source-a:default' }])

    const task = result.entities.find(
      entity => entity.identity.kind === 'kanban' && entity.identity.taskId === 'task-1' && !entity.identity.workerId
    )

    const worker = result.entities.find(
      entity => entity.identity.kind === 'kanban' && entity.identity.taskId === 'task-1' && entity.identity.workerId
    )

    expect(task).toMatchObject({
      animation: 'work',
      authority: 'authoritative',
      destination: 'project',
      identity: { board: 'main', connectionId: 'source-a', runId: '7', taskId: 'task-1' },
      observedAt: 200,
      position: manifest().projectSlots[0]!.position,
      projectId: 'project-alpha'
    })
    expect(worker).toMatchObject({
      animation: 'work',
      authority: 'authoritative',
      destination: 'project',
      identity: {
        board: 'main',
        connectionId: 'source-a',
        runId: '7',
        taskId: 'task-1',
        workerId: 'pid:4242'
      },
      projectId: 'project-alpha'
    })
  })

  it('fetches selected task detail only for the selected task', async () => {
    const rest = restFrom({
      ...boardFixtures(),
      '/tasks/task-1?board=main': {
        comments: [{ body: 'bounded detail', id: 1 }],
        events: [{ id: 9, kind: 'claimed' }],
        task: { id: 'task-1' }
      }
    })

    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 300,
      rest,
      scope: scopeA,
      selectedTaskId: () => 'task-1'
    })

    const result = await source.read()

    expect(rest.mock.calls.map(call => call[0])).toEqual([
      '/boards',
      '/board?board=main',
      '/workers/active?board=main',
      '/tasks/task-1?board=main'
    ])
    expect(result.details.get('task-1')).toMatchObject({ task: { id: 'task-1' } })
  })

  it('retains authoritative board workers when selected detail becomes unavailable, with explicit degraded cache health', async () => {
    let detail: unknown = { comments: [], events: [], runs: [], task: { id: 'task-1' } }
    const responses = boardFixtures()

    const rest = vi.fn(async (path: string) => {
      const response = path === '/tasks/task-1?board=main' ? detail : responses[path]

      if (response instanceof Error) {
        throw response
      }

      if (response === undefined) {
        throw new Error(`unexpected request ${path}`)
      }

      return response as never
    }) as ReturnType<typeof vi.fn> & KanbanRest

    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 300,
      rest,
      scope: scopeA,
      selectedTaskId: () => 'task-1'
    })

    const initial = await source.read()
    detail = httpError(503)
    source.onFrame({ cursor: 9, events: [{ id: 9, task_id: 'task-1' }] })
    const recovered = await source.read()

    expect(initial.details.get('task-1')).toEqual({ comments: [], events: [], runs: [], task: { id: 'task-1' } })
    expect(recovered.authoritative).toBe(true)
    expect(recovered.health).toBe('authoritative')
    expect(recovered.entities).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          authority: 'authoritative',
          identity: expect.objectContaining({ taskId: 'task-1' })
        }),
        expect.objectContaining({
          authority: 'authoritative',
          identity: expect.objectContaining({ workerId: 'pid:4242' })
        })
      ])
    )
    expect(recovered.details.get('task-1')).toEqual(initial.details.get('task-1'))
    expect(recovered).toMatchObject({
      detailHealth: {
        cached: true,
        error: 'Kanban selected detail unavailable',
        health: 'degraded',
        taskId: 'task-1'
      }
    })
  })

  it.each([
    ['not bounded array data', {}],
    ['more than the bounded number of run rows', Array.from({ length: 513 }, (_, id) => ({ id }))]
  ])('fails closed to degraded selected-detail health when nested runs contain %s', async (_label, runs) => {
    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 303,
      rest: restFrom({ ...boardFixtures(), '/tasks/task-1?board=main': { runs, task: { id: 'task-1' } } }),
      scope: scopeA,
      selectedTaskId: () => 'task-1'
    })

    await expect(source.read()).resolves.toMatchObject({
      authoritative: true,
      detailHealth: {
        cached: false,
        error: 'Kanban selected detail unavailable',
        health: 'degraded',
        taskId: 'task-1'
      },
      health: 'authoritative'
    })
  })

  it('dedupes concurrent selected-task reconciliation so detail stays bounded', async () => {
    const rest = restFrom({
      ...boardFixtures(),
      '/tasks/task-1?board=main': { task: { id: 'task-1' } }
    })

    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 301,
      rest,
      scope: scopeA,
      selectedTaskId: () => 'task-1'
    })

    const [first, second] = await Promise.all([source.read(), source.read()])

    expect(first).toBe(second)
    expect(rest.mock.calls.map(call => call[0])).toEqual([
      '/boards',
      '/board?board=main',
      '/workers/active?board=main',
      '/tasks/task-1?board=main'
    ])
  })

  it('drops an in-flight REST result when the route disposes its source', async () => {
    let resolveBoards: ((value: unknown) => void) | undefined

    const rest = vi.fn((path: string) => {
      if (path === '/boards') {
        return new Promise<unknown>(resolve => {
          resolveBoards = resolve
        })
      }

      throw new Error(`unexpected request after disposal ${path}`)
    }) as ReturnType<typeof vi.fn> & KanbanRest

    const source = createKanbanCitySource({ manifest: manifest(), now: () => 301, rest, scope: scopeA })
    const pending = source.read()
    const stop = source.start(vi.fn())

    stop()
    resolveBoards?.({ boards: [{ is_current: true, slug: 'main' }], current: 'main' })

    await expect(pending).resolves.toMatchObject({
      authoritative: false,
      entities: [],
      health: 'unavailable',
      sources: [
        {
          error: 'Kanban source disposed',
          source: 'kanban:source-a:default'
        }
      ]
    })
    expect(rest).toHaveBeenCalledOnce()
  })

  it('retains selected detail until its task receives a socket invalidation', async () => {
    const rest = restFrom({
      ...boardFixtures(),
      '/tasks/task-1?board=main': { task: { id: 'task-1' } }
    })

    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 302,
      rest,
      scope: scopeA,
      selectedTaskId: () => 'task-1'
    })

    await source.read()
    await source.read()

    expect(rest.mock.calls.map(call => call[0])).toEqual([
      '/boards',
      '/board?board=main',
      '/workers/active?board=main',
      '/tasks/task-1?board=main',
      '/boards',
      '/board?board=main',
      '/workers/active?board=main'
    ])
  })

  it('keeps same task and run IDs separate across explicit connections', async () => {
    const first = await createKanbanCitySource({
      manifest: manifest(),
      now: () => 400,
      rest: restFrom(boardFixtures()),
      scope: scopeA
    }).read()

    const second = await createKanbanCitySource({
      manifest: manifest(),
      now: () => 400,
      rest: restFrom(boardFixtures()),
      scope: scopeB
    }).read()

    const keys = new Set([...first.entities, ...second.entities].map(entity => entity.key))

    expect(keys.size).toBe(first.entities.length + second.entities.length)
    expect([...keys]).toContain(
      entityKey({
        board: 'main',
        connectionId: 'source-a',
        kind: 'kanban',
        profile: 'default',
        runId: '7',
        taskId: 'task-1'
      })
    )
    expect([...keys]).toContain(
      entityKey({
        board: 'main',
        connectionId: 'source-b',
        kind: 'kanban',
        profile: 'default',
        runId: '7',
        taskId: 'task-1'
      })
    )
  })

  it('keeps same task and run IDs separate across explicit profile owners on one connection', async () => {
    const first = await createKanbanCitySource({
      manifest: manifest(),
      now: () => 400,
      rest: restFrom(boardFixtures()),
      scope: scopeA
    }).read()

    const second = await createKanbanCitySource({
      manifest: manifest(),
      now: () => 400,
      rest: restFrom(boardFixtures()),
      scope: scopeAResearch
    }).read()

    const keys = new Set([...first.entities, ...second.entities].map(entity => entity.key))

    expect(keys.size).toBe(first.entities.length + second.entities.length)
    expect(first.sources[0]?.source).not.toBe(second.sources[0]?.source)
  })

  it('keeps unmatched active workers partial instead of attaching by title or assignee', async () => {
    const rest = restFrom(
      boardFixtures({
        '/workers/active?board=main': {
          checked_at: 12_345,
          count: 1,
          workers: [
            {
              run_id: 99,
              task_assignee: 'display-assignee',
              task_id: 'missing-task',
              task_status: 'running',
              task_title: 'Build approved city',
              worker_pid: 777
            }
          ]
        }
      })
    )

    const result = await createKanbanCitySource({ manifest: manifest(), now: () => 500, rest, scope: scopeA }).read()

    const worker = result.entities.find(
      entity => entity.identity.kind === 'kanban' && entity.identity.workerId === 'pid:777'
    )

    expect(worker).toMatchObject({
      animation: 'unavailable',
      authority: 'partial',
      destination: 'unknown',
      identity: { board: 'main', runId: '99', taskId: 'missing-task', workerId: 'pid:777' }
    })
    expect(worker?.projectId).toBeUndefined()
  })

  it('treats socket frames as board-local invalidation hints and never as status authority', async () => {
    let socketListener: ((message: unknown) => void) | undefined
    const disposeSocket = vi.fn()

    const socket = vi.fn((path: string, onMessage: (message: unknown) => void) => {
      socketListener = onMessage

      return disposeSocket
    }) satisfies KanbanSocket

    const invalidated = vi.fn()

    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 600,
      rest: restFrom(boardFixtures()),
      socket,
      scope: scopeA
    })

    await source.read()
    const stop = source.start(invalidated)

    expect(socket).toHaveBeenCalledWith(
      '/events?board=main&since=8',
      expect.any(Function),
      expect.objectContaining({ onReconnect: expect.any(Function) })
    )
    expect(source.onFrame({ cursor: 8, events: [{ id: 8, kind: 'completed', task_id: 'task-1' }] })).toEqual({
      accepted: false,
      dirtyTaskIds: [],
      needsReconcile: false
    })
    expect(source.onFrame({ cursor: 9, events: [{ id: 9, kind: 'completed', task_id: 'task-1' }] })).toEqual({
      accepted: true,
      dirtyTaskIds: ['task-1'],
      needsReconcile: false
    })
    expect(source.onFrame({ cursor: 11, events: [{ id: 11, kind: 'completed', task_id: 'task-1' }] })).toEqual({
      accepted: true,
      dirtyTaskIds: ['task-1'],
      needsReconcile: true
    })

    socketListener?.({ cursor: 12, events: [{ id: 12, task_id: 'task-1' }] })
    stop()
    socketListener?.({ cursor: 13, events: [{ id: 13, task_id: 'task-1' }] })

    expect(invalidated).toHaveBeenCalledOnce()
    expect(disposeSocket).toHaveBeenCalledOnce()
  })

  it('never lowers a same-board event cursor from an older REST snapshot', async () => {
    const socket = vi.fn((_path: string, _onMessage: (message: unknown) => void) => vi.fn()) satisfies KanbanSocket
    const rest = restFrom(boardFixtures())
    const source = createKanbanCitySource({ manifest: manifest(), now: () => 600, rest, scope: scopeA, socket })

    await source.read()
    source.start(vi.fn())

    expect(source.onFrame({ cursor: 9, events: [{ id: 9, task_id: 'task-1' }] })).toEqual({
      accepted: true,
      dirtyTaskIds: ['task-1'],
      needsReconcile: false
    })

    await source.read()

    const readsBeforeDuplicate = rest.mock.calls.length
    expect(source.onFrame({ cursor: 9, events: [{ id: 9, task_id: 'task-1' }] })).toEqual({
      accepted: false,
      dirtyTaskIds: [],
      needsReconcile: false
    })
    expect(rest).toHaveBeenCalledTimes(readsBeforeDuplicate)
    expect(socket.mock.calls.map(([path]) => path)).toEqual([
      '/events?board=main&since=8',
      '/events?board=main&since=9'
    ])
  })

  it('turns a reconnected socket into one REST-rebaseline hint without accepting socket status', async () => {
    let reconnect: (() => void) | undefined

    const socket = vi.fn((...args: unknown[]) => {
      reconnect = (args[2] as { onReconnect?: () => void } | undefined)?.onReconnect

      return vi.fn()
    }) as unknown as KanbanSocket

    const invalidated = vi.fn()

    const source = createKanbanCitySource({
      manifest: manifest(),
      now: () => 600,
      rest: restFrom(boardFixtures()),
      socket,
      scope: scopeA
    })

    await source.read()
    source.start(invalidated)

    expect(reconnect).toEqual(expect.any(Function))
    reconnect?.()

    expect(invalidated).toHaveBeenCalledOnce()
    expect(invalidated).toHaveBeenCalledWith({ accepted: true, dirtyTaskIds: [], needsReconcile: true })
  })

  it('rebaselines a board switch and ignores a late frame from the disposed board socket', async () => {
    let board = 'main'

    const rest = vi.fn(async (path: string, _options) => {
      if (path === '/boards') {
        return {
          boards: [{ is_current: true, project_id: `project-${board}`, slug: board }],
          current: board
        } as never
      }

      if (path === '/board?board=main') {
        return { columns: [], latest_event_id: 8 } as never
      }

      if (path === '/workers/active?board=main') {
        return { workers: [] } as never
      }

      if (path === '/board?board=second') {
        return { columns: [], latest_event_id: 20 } as never
      }

      if (path === '/workers/active?board=second') {
        return { workers: [] } as never
      }

      throw new Error(`unexpected request ${path}`)
    }) as ReturnType<typeof vi.fn> & KanbanRest

    const listeners = new Map<string, (message: unknown) => void>()

    const socket = vi.fn((path: string, onMessage: (message: unknown) => void) => {
      listeners.set(path, onMessage)

      return vi.fn()
    }) satisfies KanbanSocket

    const invalidated = vi.fn()
    const source = createKanbanCitySource({ manifest: manifest(), now: () => 601, rest, scope: scopeA, socket })

    await source.read()
    source.start(invalidated)
    board = 'second'
    await source.read()

    const mainPath = '/events?board=main&since=8'
    const secondPath = '/events?board=second&since=20'
    expect(socket.mock.calls.map(call => call[0])).toEqual([mainPath, secondPath])

    listeners.get(mainPath)?.({ cursor: 900, events: [{ id: 900, task_id: 'stale-main' }] })
    listeners.get(secondPath)?.({ cursor: 21, events: [{ id: 21, task_id: 'live-second' }] })

    expect(invalidated).toHaveBeenCalledTimes(1)
    expect(invalidated).toHaveBeenCalledWith({ accepted: true, dirtyTaskIds: ['live-second'], needsReconcile: false })
  })

  it('uses delimiter-safe compound keys and deterministic finite project slots', () => {
    const slots = manifest().projectSlots

    const forward = allocateProjectCompounds(
      [
        { connectionId: 'source-a', projectId: 'project-z', taskCount: 1 },
        { connectionId: 'source-a', projectId: 'project-a', taskCount: 2 },
        { connectionId: 'source-a', projectId: 'project-overflow', taskCount: 3 },
        { connectionId: 'source-a', projectId: 'project-extra', taskCount: 4 }
      ],
      slots
    )

    const reverse = allocateProjectCompounds(
      [
        { connectionId: 'source-a', projectId: 'project-extra', taskCount: 4 },
        { connectionId: 'source-a', projectId: 'project-overflow', taskCount: 3 },
        { connectionId: 'source-a', projectId: 'project-a', taskCount: 2 },
        { connectionId: 'source-a', projectId: 'project-z', taskCount: 1 }
      ],
      slots
    )

    expect(compoundKey('a::b', 'c')).not.toBe(compoundKey('a', 'b::c'))
    expect(forward.map(placement => placement.slotId)).toEqual(reverse.map(placement => placement.slotId))
    expect(forward.map(placement => placement.projectId)).toEqual([
      'project-a',
      'project-extra',
      'project-overflow',
      'project-z'
    ])
    expect(forward.map(placement => placement.slotId)).toEqual([
      'compound-inner-1',
      'compound-inner-2',
      'compound-outer-1',
      undefined
    ])
    expect(forward.at(-1)).toMatchObject({ taskCount: 1, unplaced: true })
  })

  it('retains existing compound slots when an unrelated canonical project appears', () => {
    const slots = manifest().projectSlots

    const initial = allocateProjectCompounds(
      [
        { connectionId: 'source-a', projectId: 'project-b', taskCount: 1 },
        { connectionId: 'source-a', projectId: 'project-c', taskCount: 1 }
      ],
      slots
    )

    const updated = allocateProjectCompounds(
      [
        { connectionId: 'source-a', projectId: 'project-a', taskCount: 1 },
        { connectionId: 'source-a', projectId: 'project-b', taskCount: 1 },
        { connectionId: 'source-a', projectId: 'project-c', taskCount: 1 }
      ],
      slots,
      initial
    )

    expect(updated.find(placement => placement.projectId === 'project-b')?.slotId).toBe(
      initial.find(placement => placement.projectId === 'project-b')?.slotId
    )
    expect(updated.find(placement => placement.projectId === 'project-c')?.slotId).toBe(
      initial.find(placement => placement.projectId === 'project-c')?.slotId
    )
    expect(updated.find(placement => placement.projectId === 'project-a')?.slotId).toBe('compound-outer-1')
  })

  it('retains live compound placement across later bounded REST reconciliations', async () => {
    let includeEarlierProject = false

    const rest = vi.fn(async (path: string, _options) => {
      if (path === '/boards') {
        return { boards: [{ is_current: true, slug: 'main' }], current: 'main' } as never
      }

      if (path === '/board?board=main') {
        const projects = includeEarlierProject ? ['project-a', 'project-b', 'project-c'] : ['project-b', 'project-c']

        return {
          columns: [
            {
              name: 'running',
              tasks: projects.map((projectId, index) => ({
                id: `task-${index}`,
                project_id: projectId,
                status: 'running'
              }))
            }
          ],
          latest_event_id: includeEarlierProject ? 2 : 1
        } as never
      }

      if (path === '/workers/active?board=main') {
        return { workers: [] } as never
      }

      throw new Error(`unexpected request ${path}`)
    }) as ReturnType<typeof vi.fn> & KanbanRest

    const source = createKanbanCitySource({ manifest: manifest(), now: () => 603, rest, scope: scopeA })

    const initial = await source.read()
    includeEarlierProject = true
    const updated = await source.read()

    expect(updated.compounds.find(placement => placement.projectId === 'project-b')?.slotId).toBe(
      initial.compounds.find(placement => placement.projectId === 'project-b')?.slotId
    )
    expect(updated.compounds.find(placement => placement.projectId === 'project-c')?.slotId).toBe(
      initial.compounds.find(placement => placement.projectId === 'project-c')?.slotId
    )
  })

  it('proves declared project slots are inside the world, non-overlapping, and graph-linked', () => {
    const parsed = manifest()

    expect(projectSlotContractIssues(parsed)).toEqual([])

    const [first, second] = parsed.projectSlots

    const overlapping: ProjectSlotManifestEntry = {
      ...second!,
      id: 'compound-overlap',
      bounds: first!.bounds,
      position: first!.position
    }

    expect(projectSlotContractIssues({ ...parsed, projectSlots: [first!, overlapping] })).toContain(
      'projectSlots[compound-inner-1] overlaps projectSlots[compound-overlap]'
    )
  })
})

describe('registered Kanban Lunar City source', () => {
  it('merges colliding task identities from every exact registered profile owner', async () => {
    const roster = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    const childStops: ReturnType<typeof vi.fn>[] = []

    const createSource = vi.fn((options: { scope: PluginSourceScope }) => {
      const stop = vi.fn()
      childStops.push(stop)

      const identity = {
        board: 'main',
        connectionId: options.scope.connectionId,
        kind: 'kanban' as const,
        profile: options.scope.profile,
        taskId: 'shared-task'
      }

      return {
        onFrame: vi.fn(),
        read: vi.fn(async () => ({
          authoritative: true,
          compounds: [],
          details: new Map(),
          entities: [
            {
              animation: 'work' as const,
              authority: 'authoritative' as const,
              destination: 'project' as const,
              identity,
              key: entityKey(identity),
              observedAt: 42
            }
          ],
          health: 'authoritative' as const,
          sources: [
            {
              authority: 'authoritative' as const,
              observedAt: 42,
              source: `kanban:${options.scope.connectionId}:worker`
            }
          ]
        })),
        start: vi.fn(() => stop)
      }
    })

    const source = createRegisteredKanbanCitySource({
      createSource: createSource as never,
      manifest: manifest(),
      roster
    })

    const invalidate = vi.fn()
    const stop = source.start(invalidate)
    const result = await source.read()

    expect(createSource.mock.calls.map(call => call[0].scope)).toEqual([
      { connectionId: 'source-a', profile: 'worker' },
      { connectionId: 'source-b', profile: 'worker' }
    ])
    expect(result.authoritative).toBe(true)
    expect(result.entities.map(entity => entity.identity)).toEqual([
      { board: 'main', connectionId: 'source-a', kind: 'kanban', profile: 'worker', taskId: 'shared-task' },
      { board: 'main', connectionId: 'source-b', kind: 'kanban', profile: 'worker', taskId: 'shared-task' }
    ])
    expect(result.replacementSources).toEqual(['kanban:source-a:worker', 'kanban:source-b:worker'])

    stop()
    expect(childStops.every(childStop => childStop.mock.calls.length === 1)).toBe(true)
  })

  it('keeps a failed owner partial without rereading or replacing the healthy owner and bounds dynamic cleanup', async () => {
    const roster = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    const stops = new Map<string, ReturnType<typeof vi.fn>>()
    const reads = new Map<string, ReturnType<typeof vi.fn>>()
    let remoteFails = true

    const createSource = vi.fn((options: { scope: PluginSourceScope }) => {
      const stop = vi.fn()
      stops.set(options.scope.connectionId, stop)

      const read = vi.fn(async () => {
        if (options.scope.connectionId === 'source-b' && remoteFails) {
          return {
            authoritative: false,
            compounds: [],
            details: new Map(),
            entities: [],
            health: 'unavailable' as const,
            sources: [
              {
                authority: 'unknown' as const,
                error: 'Kanban plugin unavailable',
                observedAt: 42,
                source: 'kanban:source-b:worker'
              }
            ]
          }
        }

        return {
          authoritative: true,
          compounds: [],
          details: new Map(),
          entities: [],
          health: 'authoritative' as const,
          sources: [
            {
              authority: 'authoritative' as const,
              observedAt: 42,
              source: `kanban:${options.scope.connectionId}:worker`
            }
          ]
        }
      })

      reads.set(options.scope.connectionId, read)

      return {
        onFrame: vi.fn(),
        read,
        start: vi.fn(() => stop)
      }
    })

    const source = createRegisteredKanbanCitySource({
      createSource: createSource as never,
      manifest: manifest(),
      roster
    })

    const stop = source.start(vi.fn())

    await expect(source.read()).resolves.toMatchObject({
      authoritative: false,
      replacementSources: ['kanban:source-a:worker']
    })

    remoteFails = false
    roster.set({ agents: [...roster.get().agents], sources: [...roster.get().sources] })
    await expect(source.read()).resolves.toMatchObject({ authoritative: true })
    expect(reads.get('source-a')).toHaveBeenCalledOnce()
    expect(reads.get('source-b')).toHaveBeenCalledTimes(2)

    roster.set({
      agents: roster.get().agents.filter(agent => agent.connectionId === 'source-a'),
      sources: roster.get().sources.filter(item => item.connectionId === 'source-a')
    })
    expect(stops.get('source-b')).toHaveBeenCalledOnce()

    await expect(source.read()).resolves.toMatchObject({ authoritative: false })
    await expect(source.read()).resolves.toMatchObject({ authoritative: true })
    expect(createSource).toHaveBeenCalledTimes(2)
    stop()
    expect(stops.get('source-a')).toHaveBeenCalledOnce()
  })

  it('recovers one initial boards failure on a roster refresh without rereading healthy owners', async () => {
    const roster = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    const requests = new Map<string, string[]>()
    let remoteBoardsFails = true
    const fixtures = boardFixtures()

    const createSource = (options: KanbanCitySourceOptions) => {
      const rest: KanbanRest = async path => {
        requests.set(options.scope.connectionId, [...(requests.get(options.scope.connectionId) ?? []), path])

        if (options.scope.connectionId === 'source-b' && path === '/boards' && remoteBoardsFails) {
          throw new Error('gateway unavailable')
        }

        return fixtures[path] as never
      }

      return createKanbanCitySource({ ...options, rest, socket: () => () => undefined })
    }

    const source = createRegisteredKanbanCitySource({ createSource, roster })
    source.start(vi.fn())
    await expect(source.read()).resolves.toMatchObject({ authoritative: false })
    const healthyRequestCount = requests.get('source-a')!.length

    remoteBoardsFails = false
    roster.set({ agents: [...roster.get().agents], sources: [...roster.get().sources] })
    await expect(source.read()).resolves.toMatchObject({ authoritative: true })
    expect(requests.get('source-a')).toHaveLength(healthyRequestCount)
    expect(requests.get('source-b')!.filter(path => path === '/boards')).toHaveLength(2)
  })

  it('keeps an exact owner dirty when its socket invalidates during an in-flight snapshot read', async () => {
    const roster = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    const invalidators = new Map<string, () => void>()
    const reads = new Map<string, ReturnType<typeof vi.fn>>()
    let releaseRemote!: () => void

    const remoteGate = new Promise<void>(resolve => {
      releaseRemote = resolve
    })

    const createSource = (options: { scope: PluginSourceScope }) => {
      const read = vi.fn(async () => {
        if (options.scope.connectionId === 'source-b' && read.mock.calls.length === 1) {
          await remoteGate
        }

        return {
          authoritative: true,
          compounds: [],
          details: new Map(),
          entities: [],
          health: 'authoritative' as const,
          sources: [
            {
              authority: 'authoritative' as const,
              observedAt: 42,
              source: `kanban:${options.scope.connectionId}:worker`
            }
          ]
        }
      })

      reads.set(options.scope.connectionId, read)

      return {
        onFrame: vi.fn(),
        read,
        start: (listener: () => void) => {
          invalidators.set(options.scope.connectionId, listener)

          return () => undefined
        }
      }
    }

    const source = createRegisteredKanbanCitySource({ createSource: createSource as never, roster })
    source.start(vi.fn())
    const initialRead = source.read()
    await Promise.resolve()
    invalidators.get('source-b')!()
    releaseRemote()
    await initialRead
    await source.read()

    expect(reads.get('source-a')).toHaveBeenCalledOnce()
    expect(reads.get('source-b')).toHaveBeenCalledTimes(2)
  })

  it('bounds retired-owner churn and publishes each retained tombstone for only one read', async () => {
    const ownerRoster = (index: number): DesktopAgentRoster => ({
      agents: [
        {
          connectionId: `source-${index}`,
          connectionKind: 'local',
          connectionLabel: `source-${index}`,
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: `source-${index}`, kind: 'local', label: `source-${index}`, reachable: true }]
    })

    const roster = atom(ownerRoster(0))

    const createSource = (options: { scope: PluginSourceScope }) => ({
      onFrame: vi.fn(),
      read: async () => ({
        authoritative: true,
        compounds: [],
        details: new Map(),
        entities: [],
        health: 'authoritative' as const,
        sources: [
          {
            authority: 'authoritative' as const,
            observedAt: 42,
            source: `kanban:${options.scope.connectionId}:worker`
          }
        ]
      }),
      start: () => () => undefined
    })

    const source = createRegisteredKanbanCitySource({ createSource: createSource as never, roster })
    source.start(vi.fn())
    await source.read()

    for (let index = 1; index <= 300; index += 1) {
      roster.set(ownerRoster(index))
    }

    const removalRead = await source.read()
    expect(removalRead.authoritative).toBe(false)
    expect(removalRead.staleUnlistedSourcePrefixes).toEqual(['kanban:'])
    expect(removalRead.sources).toHaveLength(258)
    expect(removalRead.sources.some(item => item.source === 'kanban:source-1:worker')).toBe(false)
    expect(removalRead.sources).toContainEqual(
      expect.objectContaining({ source: 'kanban:source-299:worker', error: 'Registered Kanban owner removed' })
    )
    expect(removalRead.sources).toContainEqual({
      authority: 'partial',
      error: 'Registered Kanban removal tombstone limit exceeded',
      observedAt: expect.any(Number),
      source: 'kanban-registry:removal-overflow'
    })

    const acknowledged = await source.read()
    expect(acknowledged.authoritative).toBe(true)
    expect(acknowledged.sources).toEqual([
      { authority: 'authoritative', observedAt: 42, source: 'kanban:source-300:worker' }
    ])
  })

  it('distinguishes unavailable, authoritative empty, and oversized roster states', async () => {
    const roster = atom<DesktopAgentRoster | null>({
      agents: [
        {
          connectionId: 'source-a',
          connectionKind: 'local',
          connectionLabel: 'source a',
          handle: '@worker',
          profile: 'worker'
        }
      ],
      sources: [{ connectionId: 'source-a', kind: 'local', label: 'source a', reachable: true }]
    })

    const identity = {
      board: 'main',
      connectionId: 'source-a',
      kind: 'kanban' as const,
      profile: 'worker',
      taskId: 'task-a'
    }

    const createSource = vi.fn(() => ({
      onFrame: vi.fn(),
      read: async () => ({
        authoritative: true,
        compounds: [],
        details: new Map(),
        entities: [
          {
            animation: 'work' as const,
            authority: 'authoritative' as const,
            destination: 'project' as const,
            identity,
            key: entityKey(identity),
            observedAt: 42
          }
        ],
        health: 'authoritative' as const,
        sources: [{ authority: 'authoritative' as const, observedAt: 42, source: 'kanban:source-a:worker' }]
      }),
      start: () => () => undefined
    }))

    const source = createRegisteredKanbanCitySource({ createSource: createSource as never, roster })
    source.start(vi.fn())
    await expect(source.read()).resolves.toMatchObject({ authoritative: true })

    roster.set(null)
    const unavailable = await source.read()
    expect(unavailable.authoritative).toBe(false)
    expect(unavailable.entities).toEqual([expect.objectContaining({ authority: 'stale', key: entityKey(identity) })])
    expect(unavailable.sources).toContainEqual(
      expect.objectContaining({ source: 'kanban-registry:unavailable', authority: 'partial' })
    )

    const oversizedAgents = Array.from({ length: 257 }, (_, index) => ({
      connectionId: `oversized-${index}`,
      connectionKind: 'local' as const,
      connectionLabel: `oversized-${index}`,
      handle: '@worker',
      profile: 'worker'
    }))

    roster.set({
      agents: oversizedAgents,
      sources: oversizedAgents.map(agent => ({
        connectionId: agent.connectionId,
        kind: 'local' as const,
        label: agent.connectionLabel,
        reachable: true
      }))
    })
    const oversized = await source.read()
    expect(oversized.authoritative).toBe(false)
    expect(oversized.entities).toEqual([expect.objectContaining({ authority: 'stale', key: entityKey(identity) })])
    expect(oversized.sources).toContainEqual(
      expect.objectContaining({ source: 'kanban-registry:overflow', authority: 'partial' })
    )
    expect(createSource).toHaveBeenCalledOnce()

    roster.set({ agents: [], sources: [] })
    const removed = await source.read()
    expect(removed.authoritative).toBe(false)
    expect(removed.sources).toContainEqual(
      expect.objectContaining({ source: 'kanban:source-a:worker', error: 'Registered Kanban owner removed' })
    )
    const acknowledged = await source.read()
    expect(acknowledged).toMatchObject({ authoritative: true, entities: [], sources: [] })
  })

  it('fails closed for an initially oversized roster without creating child sources', async () => {
    const agents = Array.from({ length: 257 }, (_, index) => ({
      connectionId: `source-${index}`,
      connectionKind: 'local' as const,
      connectionLabel: `source-${index}`,
      handle: '@worker',
      profile: 'worker'
    }))

    const roster = atom<DesktopAgentRoster>({
      agents,
      sources: agents.map(agent => ({
        connectionId: agent.connectionId,
        kind: 'local' as const,
        label: agent.connectionLabel,
        reachable: true
      }))
    })

    const createSource = vi.fn()
    const source = createRegisteredKanbanCitySource({ createSource, roster })
    source.start(vi.fn())
    const result = await source.read()

    expect(createSource).not.toHaveBeenCalled()
    expect(result.authoritative).toBe(false)
    expect(result.entities).toEqual([])
    expect(result.sources).toEqual([
      {
        authority: 'partial',
        error: 'Registered Kanban owner limit exceeded',
        observedAt: expect.any(Number),
        source: 'kanban-registry:overflow'
      }
    ])
  })

  it('rereads only the exact dirty owner and retains collision-safe detail entries', async () => {
    const roster = atom<DesktopAgentRoster>({
      agents: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: ['source-a', 'source-b'].map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    const invalidators = new Map<string, () => void>()
    const reads = new Map<string, ReturnType<typeof vi.fn>>()

    const createSource = (options: { scope: PluginSourceScope }) => {
      const read = vi.fn(async () => ({
        authoritative: true,
        compounds: [],
        details: new Map([['shared-task', { owner: options.scope.connectionId }]]),
        entities: [],
        health: 'authoritative' as const,
        selectedBoard: 'main',
        sources: [
          {
            authority: 'authoritative' as const,
            observedAt: 42,
            source: `kanban:${options.scope.connectionId}:worker`
          }
        ]
      }))

      reads.set(options.scope.connectionId, read)

      return {
        onFrame: vi.fn(),
        read,
        start: (listener: () => void) => {
          invalidators.set(options.scope.connectionId, listener)

          return () => undefined
        }
      }
    }

    const source = createRegisteredKanbanCitySource({
      createSource: createSource as never,
      manifest: manifest(),
      roster
    })

    source.start(vi.fn())
    const initial = await source.read()
    expect(reads.get('source-a')).toHaveBeenCalledOnce()
    expect(reads.get('source-b')).toHaveBeenCalledOnce()
    expect(initial.details.get(kanbanDetailKey({ ...scopeA, profile: 'worker' }, 'main', 'shared-task'))).toEqual({
      owner: 'source-a'
    })
    expect(initial.details.get(kanbanDetailKey({ ...scopeB, profile: 'worker' }, 'main', 'shared-task'))).toEqual({
      owner: 'source-b'
    })

    invalidators.get('source-b')!()
    await source.read()
    expect(reads.get('source-a')).toHaveBeenCalledOnce()
    expect(reads.get('source-b')).toHaveBeenCalledTimes(2)
  })

  it('bounds initial registered-owner reads to four concurrent requests', async () => {
    const connectionIds = Array.from({ length: 12 }, (_, index) => `source-${index}`)

    const roster = atom<DesktopAgentRoster>({
      agents: connectionIds.map(connectionId => ({
        connectionId,
        connectionKind: 'local' as const,
        connectionLabel: connectionId,
        handle: `@worker-${connectionId}`,
        profile: 'worker'
      })),
      sources: connectionIds.map(connectionId => ({
        connectionId,
        kind: 'local' as const,
        label: connectionId,
        reachable: true
      }))
    })

    let active = 0
    let maxActive = 0

    const createSource = (options: { scope: PluginSourceScope }) => ({
      onFrame: vi.fn(),
      read: async () => {
        active += 1
        maxActive = Math.max(maxActive, active)
        await new Promise(resolve => setTimeout(resolve, 0))
        active -= 1

        return {
          authoritative: true,
          compounds: [],
          details: new Map(),
          entities: [],
          health: 'authoritative' as const,
          sources: [
            {
              authority: 'authoritative' as const,
              observedAt: 42,
              source: `kanban:${options.scope.connectionId}:worker`
            }
          ]
        }
      },
      start: () => () => undefined
    })

    const source = createRegisteredKanbanCitySource({
      createSource: createSource as never,
      manifest: manifest(),
      roster
    })

    source.start(vi.fn())
    await source.read()

    expect(maxActive).toBe(4)
  })
})
