// @vitest-environment jsdom

import { act, fireEvent, render, screen } from '@testing-library/react'
import React, {
  type ComponentType,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

interface PassthroughProps {
  children?: ReactNode
}

const Box = ({ children }: PassthroughProps) => React.createElement('div', null, children)

const Button = ({ children, ...props }: PassthroughProps & React.ButtonHTMLAttributes<HTMLButtonElement>) =>
  React.createElement('button', props, children)

const Input = (props: React.InputHTMLAttributes<HTMLInputElement>) => React.createElement('input', props)

const Label = ({ children, ...props }: PassthroughProps & React.LabelHTMLAttributes<HTMLLabelElement>) =>
  React.createElement('label', props, children)

const task = {
  id: 'task-1',
  title: 'Live worker task',
  body: 'Exercise the worker log.',
  status: 'running',
  priority: 1,
  assignee: 'worker',
  tenant: 'default',
  created_at: '2026-09-01T00:00:00Z',
  updated_at: '2026-09-01T00:00:00Z',
}

let KanbanPage: ComponentType | null = null
let logReads = 0

const fetchJSON = vi.fn((url: string) => {
  const path = String(url).split('?')[0]

  if (path.endsWith('/tasks/task-1/log')) {
    logReads += 1

    return Promise.resolve({
      exists: true,
      size_bytes: logReads === 1 ? 0 : 14,
      content: logReads === 1 ? '' : 'worker output\n',
      path: '/tmp/task-1.log',
      truncated: false,
    })
  }

  if (path.endsWith('/tasks/task-1')) {
    return Promise.resolve({
      task,
      comments: [],
      events: [],
      attachments: [],
      links: { parents: [], children: [] },
      child_results: [],
      runs: [],
    })
  }

  if (path.endsWith('/home-channels')) {return Promise.resolve({ home_channels: [] })}

  if (path.endsWith('/config')) {return Promise.resolve({ render_markdown: false })}

  if (path.endsWith('/boards')) {
    return Promise.resolve({ boards: [{ slug: 'default', name: 'Default' }], current: 'default' })
  }

  if (path.endsWith('/board')) {
    return Promise.resolve({
      columns: [{ name: 'running', tasks: [task] }],
      assignees: ['worker'],
      tenants: ['default'],
      latest_event_id: 0,
    })
  }

  if (path.endsWith('/orchestration')) {return Promise.resolve({})}

  if (path.endsWith('/profiles')) {return Promise.resolve({ profiles: [] })}

  return Promise.reject(new Error(`Unexpected request: ${url}`))
})

class FakeWebSocket {
  onclose: null | ((event: { code: number }) => void) = null
  onmessage: null | ((event: { data: string }) => void) = null
  onopen: null | (() => void) = null

  close() {}
}

async function flushUpdates() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
    await Promise.resolve()
  })
}

describe('Kanban dashboard worker log refresh', () => {
  beforeEach(async () => {
    vi.useFakeTimers()
    logReads = 0
    fetchJSON.mockClear()
    KanbanPage = null

    Object.assign(window, {
      __HERMES_PLUGIN_SDK__: {
        React,
        hooks: { useCallback, useEffect, useMemo, useRef, useState },
        components: {
          Badge: Box,
          Button,
          Card: Box,
          CardContent: Box,
          Checkbox: Input,
          ConfirmDialog: () => null,
          Input,
          Label,
          Select: Box,
          SelectOption: Box,
        },
        utils: {
          cn: (...parts: Array<false | null | string | undefined>) => parts.filter(Boolean).join(' '),
          timeAgo: () => 'now',
        },
        useI18n: () => ({ t: { kanban: null }, locale: 'en' }),
        fetchJSON,
        buildWsUrl: () => Promise.resolve('ws://localhost/kanban'),
      },
      __HERMES_PLUGINS__: {
        register: (_name: string, page: ComponentType) => {
          KanbanPage = page
        },
      },
      WebSocket: FakeWebSocket,
    })

    vi.resetModules()
    // The dashboard plugin is a runtime IIFE that registers its page globally.
    // @ts-expect-error The plain JavaScript plugin intentionally has no TS declarations.
    await import('../plugins/kanban/dashboard/dist/index.js')
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  test('shows new output from a running worker without manual refresh', async () => {
    expect(KanbanPage).not.toBeNull()
    const page = KanbanPage as ComponentType
    const view = render(React.createElement(page))
    await flushUpdates()

    fireEvent.click(screen.getByRole('button', { name: /Live worker task/ }))
    await flushUpdates()

    expect(screen.getByText('(empty)')).toBeTruthy()
    expect(logReads).toBe(1)

    await act(async () => {
      vi.advanceTimersByTime(3000)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByText('worker output')).toBeTruthy()
    expect(logReads).toBe(2)
    view.unmount()
  })
})
