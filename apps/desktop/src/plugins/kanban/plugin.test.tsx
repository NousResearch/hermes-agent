import { host, type PluginContext } from '@hermes/plugin-sdk'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import plugin from './plugin'

afterEach(() => {
  ;(host.state.profile as unknown as { set(value: string): void }).set('default')
})

describe('bundled Kanban plugin', () => {
  it('uses the canonical HermesPlugin registration contract', () => {
    const registrations: Array<Record<string, unknown>> = []

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest: vi.fn(),
      socket: vi.fn(() => vi.fn()),
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)

    expect(registrations).toEqual([
      expect.objectContaining({ area: 'routes', data: { path: '/kanban' }, id: 'route' }),
      expect.objectContaining({ area: 'sidebar.nav', data: expect.objectContaining({ path: '/kanban' }), id: 'nav' })
    ])
    expect(registrations[0]?.render).toEqual(expect.any(Function))
  })

  it('loads board data through the scoped REST and socket doors', async () => {
    const registrations: Array<Record<string, unknown>> = []

    const rest = vi.fn().mockResolvedValue({
      columns: [{ name: 'todo', tasks: [{ assignee: 'dollycode', id: 't_1', title: 'Repair plugin host' }] }],
      latest_event_id: 42
    })

    const socket = vi.fn(() => vi.fn())

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest,
      socket,
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)
    const routeRender = registrations[0]?.render

    if (typeof routeRender !== 'function') {
      throw new Error('Kanban route contribution did not register a renderer')
    }

    render(routeRender())

    expect(await screen.findByText('Repair plugin host')).toBeTruthy()
    expect(rest).toHaveBeenCalledWith('/board', { timeoutMs: 60_000 })
    expect(socket).toHaveBeenCalledWith('/events', expect.any(Function))
  })

  it('dedupes event bursts and defers automatic refreshes while a read is active', async () => {
    const registrations: Array<Record<string, unknown>> = []
    let resolveFirst: ((value: unknown) => void) | undefined
    let onSocketMessage: ((data: unknown) => void) | undefined
    const disposeSocket = vi.fn()

    const rest = vi.fn().mockImplementationOnce(
      () =>
        new Promise(resolve => {
          resolveFirst = resolve
        })
    )

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest,
      socket: vi.fn((_path: string, onMessage: (data: unknown) => void) => {
        onSocketMessage = onMessage

        return disposeSocket
      }),
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)
    const routeRender = registrations[0]?.render

    if (typeof routeRender !== 'function') {
      throw new Error('Kanban route contribution did not register a renderer')
    }

    const { unmount } = render(routeRender())

    await waitFor(() => expect(rest).toHaveBeenCalledTimes(1))

    act(() => {
      onSocketMessage?.({ id: 1 })
      onSocketMessage?.({ id: 2 })
      onSocketMessage?.({ id: 3 })
    })
    expect(rest).toHaveBeenCalledTimes(1)

    await act(async () => {
      resolveFirst?.({ columns: [], latest_event_id: 3 })
    })
    await waitFor(() => expect(screen.getByText('Kanban')).toBeTruthy())

    act(() => {
      onSocketMessage?.({ id: 4 })
      onSocketMessage?.({ id: 5 })
    })
    expect(rest).toHaveBeenCalledTimes(1)

    unmount()
    expect(disposeSocket).toHaveBeenCalledOnce()
  })

  it('shares in-flight reads across remounts and ignores results from an old profile', async () => {
    const registrations: Array<Record<string, unknown>> = []
    const resolvers: Array<(value: unknown) => void> = []

    const rest = vi.fn(
      () =>
        new Promise(resolve => {
          resolvers.push(resolve)
        })
    )

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest,
      socket: vi.fn(() => vi.fn()),
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)
    const routeRender = registrations[0]?.render

    if (typeof routeRender !== 'function') {
      throw new Error('Kanban route contribution did not register a renderer')
    }

    const strictMount = render(<StrictMode>{routeRender()}</StrictMode>)

    await waitFor(() => expect(rest).toHaveBeenCalledTimes(1))
    strictMount.unmount()

    const remount = render(routeRender())

    await waitFor(() => expect(rest).toHaveBeenCalledTimes(1))

    act(() => {
      ;(host.state.profile as unknown as { set(value: string): void }).set('dollycode')
    })
    await waitFor(() => expect(rest).toHaveBeenCalledTimes(2))

    await act(async () => {
      resolvers[0]?.({
        columns: [{ name: 'todo', tasks: [{ id: 'old', title: 'Old profile board' }] }]
      })
    })
    expect(screen.queryByText('Old profile board')).toBeNull()

    await act(async () => {
      resolvers[1]?.({
        columns: [{ name: 'todo', tasks: [{ id: 'new', title: 'New profile board' }] }]
      })
    })
    expect(await screen.findByText('New profile board')).toBeTruthy()

    remount.unmount()
  })

  it('opens a task button and displays the fetched task details', async () => {
    const registrations: Array<Record<string, unknown>> = []

    const rest = vi.fn((path: string) => {
      if (path === '/tasks/task%2F42') {
        return Promise.resolve({
          task: {
            assignee: 'dollycode',
            body: 'Keep the detail dialog concise.',
            id: 'task/42',
            latest_summary: 'The latest operator summary.',
            priority: 2,
            status: 'in_progress',
            title: 'Inspect the task details'
          }
        })
      }

      return Promise.resolve({
        columns: [{ name: 'todo', tasks: [{ id: 'task/42', title: 'Inspect the task details' }] }]
      })
    })

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest,
      socket: vi.fn(() => vi.fn()),
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)
    const routeRender = registrations[0]?.render

    if (typeof routeRender !== 'function') {
      throw new Error('Kanban route contribution did not register a renderer')
    }

    render(routeRender())

    const card = await screen.findByRole('button', { name: 'Open task Inspect the task details' })
    expect(card.getAttribute('type')).toBe('button')

    fireEvent.click(card)

    await waitFor(() =>
      expect(rest).toHaveBeenCalledWith('/tasks/task%2F42', {
        timeoutMs: 60_000
      })
    )

    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText('Inspect the task details')).toBeTruthy()
    expect(within(dialog).getByText('in_progress')).toBeTruthy()
    expect(within(dialog).getByText('Keep the detail dialog concise.')).toBeTruthy()
    expect(within(dialog).getByText('The latest operator summary.')).toBeTruthy()
  })

  it('validates and creates a task with a normalized payload, then refreshes the board', async () => {
    const registrations: Array<Record<string, unknown>> = []

    const rest = vi.fn((path: string) => {
      if (path === '/tasks') {
        return Promise.resolve({ task: { id: 'created', title: 'New task' } })
      }

      return Promise.resolve({ columns: [] })
    })

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest,
      socket: vi.fn(() => vi.fn()),
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)
    const routeRender = registrations[0]?.render

    if (typeof routeRender !== 'function') {
      throw new Error('Kanban route contribution did not register a renderer')
    }

    render(routeRender())
    await screen.findByText('Kanban')

    fireEvent.click(screen.getByRole('button', { name: 'New task' }))
    const dialog = await screen.findByRole('dialog')
    const form = within(dialog).getByRole('textbox', { name: 'Title' }).closest('form')

    if (!form) {
      throw new Error('New task form did not render')
    }

    fireEvent.submit(form)
    expect((await within(dialog).findByRole('alert')).textContent).toContain('Title is required.')
    expect(rest).not.toHaveBeenCalledWith('/tasks', expect.anything())

    fireEvent.change(within(dialog).getByRole('textbox', { name: 'Title' }), {
      target: { value: '  New operator task  ' }
    })
    fireEvent.change(within(dialog).getByRole('textbox', { name: /Description/ }), {
      target: { value: '  A useful description  ' }
    })
    fireEvent.change(within(dialog).getByRole('textbox', { name: /Assignee/ }), {
      target: { value: '  dollycode  ' }
    })
    fireEvent.change(within(dialog).getByRole('spinbutton', { name: /Priority/ }), {
      target: { value: '4' }
    })
    fireEvent.submit(form)

    await waitFor(() =>
      expect(rest).toHaveBeenCalledWith('/tasks', {
        body: { assignee: 'dollycode', body: 'A useful description', priority: 4, title: 'New operator task' },
        method: 'POST'
      })
    )
    await waitFor(() => expect(rest).toHaveBeenCalledTimes(3))
    expect(rest).toHaveBeenLastCalledWith('/board', { timeoutMs: 60_000 })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
  })

  it('keeps the board and cards contained by semantic layout contracts', async () => {
    const registrations: Array<Record<string, unknown>> = []

    const context = {
      register: vi.fn((contribution: Record<string, unknown>) => {
        registrations.push(contribution)

        return vi.fn()
      }),
      registerMany: vi.fn(),
      rest: vi.fn().mockResolvedValue({
        columns: [{ name: 'done', tasks: [{ id: 'task-1', title: 'Contained task' }] }]
      }),
      socket: vi.fn(() => vi.fn()),
      source: 'plugin:kanban',
      storage: { get: vi.fn(), remove: vi.fn(), set: vi.fn() }
    } as unknown as PluginContext

    plugin.register(context)
    const routeRender = registrations[0]?.render

    if (typeof routeRender !== 'function') {
      throw new Error('Kanban route contribution did not register a renderer')
    }

    render(routeRender())
    const card = await screen.findByRole('button', { name: 'Open task Contained task' })
    const route = globalThis.document.querySelector('.kanban-route')
    const scroller = globalThis.document.querySelector('.kanban-board-scroller')
    const column = globalThis.document.querySelector('.kanban-column')
    const taskList = globalThis.document.querySelector('.kanban-task-list')

    const expectClasses = (element: Element | null, ...classes: string[]) => {
      expect(element).not.toBeNull()

      for (const className of classes) {
        expect(element?.classList.contains(className)).toBe(true)
      }
    }

    expectClasses(route, 'min-w-0', 'max-w-full', 'overflow-hidden')
    expectClasses(scroller, 'min-w-0', 'max-w-full', 'overflow-x-auto', 'overflow-y-hidden')
    expectClasses(column, 'h-full', 'min-h-0', 'min-w-0', 'max-w-full', 'overflow-hidden')
    expectClasses(taskList, 'min-h-0', 'min-w-0', 'max-w-full', 'overflow-x-hidden', 'overflow-y-auto')
    expectClasses(card, 'w-full', 'min-w-0', 'max-w-full', 'whitespace-normal')
  })
})
