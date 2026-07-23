import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { createElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { KanbanTab, loadKanbanSnapshot } from './index'

afterEach(() => {
  cleanup()
})

function renderKanbanTab(requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false
      }
    }
  })

  return render(
    createElement(
      QueryClientProvider,
      {
        client: queryClient,
        children: createElement(I18nProvider, {
          configClient: null,
          children: createElement(KanbanTab, { requestGateway })
        })
      }
    )
  )
}

describe('loadKanbanSnapshot', () => {
  it('loads boards and tasks from cli.exec JSON output', async () => {
    const requestGatewayMock = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      expect(method).toBe('cli.exec')

      const argv = params?.argv

      if (Array.isArray(argv) && argv.join(' ') === 'kanban boards list --json') {
        return {
          blocked: false,
          code: 0,
          output: JSON.stringify([
            {
              slug: 'default',
              name: 'Default',
              is_current: true,
              total: 2
            }
          ])
        }
      }

      if (Array.isArray(argv) && argv.join(' ') === 'kanban list --json') {
        return {
          blocked: false,
          code: 0,
          output: JSON.stringify([
            {
              id: 'TASK-1',
              title: 'Ship desktop kanban',
              status: 'ready',
              assignee: 'idah',
              created_at: 1_717_840_000
            }
          ])
        }
      }

      throw new Error(`unexpected argv: ${JSON.stringify(argv)}`)
    })
    const requestGateway = (<T>(method: string, params?: Record<string, unknown>) =>
      requestGatewayMock(method, params) as Promise<T>)

    await expect(loadKanbanSnapshot(requestGateway)).resolves.toEqual({
      boards: [
        {
          slug: 'default',
          name: 'Default',
          is_current: true,
          total: 2
        }
      ],
      currentBoard: {
        slug: 'default',
        name: 'Default',
        is_current: true,
        total: 2
      },
      tasks: [
        {
          id: 'TASK-1',
          title: 'Ship desktop kanban',
          status: 'ready',
          assignee: 'idah',
          created_at: 1_717_840_000
        }
      ]
    })
  })

  it('surfaces cli failures as readable errors', async () => {
    const requestGatewayMock = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
      const argv = params?.argv

      if (Array.isArray(argv) && argv.join(' ') === 'kanban boards list --json') {
        return { blocked: false, code: 0, output: '[]' }
      }

      return {
        blocked: false,
        code: 2,
        output: 'kanban: database missing'
      }
    })
    const requestGateway = (<T>(method: string, params?: Record<string, unknown>) =>
      requestGatewayMock(method, params) as Promise<T>)

    await expect(loadKanbanSnapshot(requestGateway)).rejects.toThrow('kanban: database missing')
  })
})

describe('KanbanTab', () => {
  it('renders the empty-board command as inline code', async () => {
    const requestGatewayMock = vi.fn(async (_method: string, params?: Record<string, unknown>) => {
      const argv = params?.argv

      if (Array.isArray(argv) && argv.join(' ') === 'kanban boards list --json') {
        return {
          blocked: false,
          code: 0,
          output: '[]'
        }
      }

      if (Array.isArray(argv) && argv.join(' ') === 'kanban list --json') {
        return {
          blocked: false,
          code: 0,
          output: '[]'
        }
      }

      throw new Error(`unexpected argv: ${JSON.stringify(argv)}`)
    })
    const requestGateway = (<T>(method: string, params?: Record<string, unknown>) =>
      requestGatewayMock(method, params) as Promise<T>)

    const { container } = renderKanbanTab(requestGateway)

    await screen.findByText('No boards')

    const code = await waitFor(() => container.querySelector('code'))
    expect(code?.textContent).toBe('hermes kanban boards create <slug>')
    expect(container.textContent).toContain('Create a board with hermes kanban boards create <slug> to see it here.')
  })
})
