import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

type RegisteredPage = React.ComponentType

const reviewTask = {
  id: 't_review',
  title: 'Review task',
  status: 'review',
  workspace_kind: 'scratch'
}

const element = (tag: keyof React.JSX.IntrinsicElements) =>
  ({ children, ...props }: React.HTMLAttributes<HTMLElement>) => React.createElement(tag, props, children)

const select = ({ children, onValueChange: _onValueChange, ...props }: React.SelectHTMLAttributes<HTMLSelectElement> & {
  onValueChange?: (value: string) => void
}) => React.createElement('select', props, children)

describe('kanban dashboard task actions', () => {
  let registeredPage: RegisteredPage | null

  beforeEach(async () => {
    vi.resetModules()
    registeredPage = null

    class FakeWebSocket {
      close() {}
    }
    vi.stubGlobal('WebSocket', FakeWebSocket)

    const fetchJSON = vi.fn(async (url: string) => {
      if (url.includes('/config')) {
        return { render_markdown: false }
      }
      if (url.includes('/boards')) {
        return { boards: [{ slug: 'default', name: 'Default' }], current: 'default' }
      }
      if (url.includes('/home-channels')) {
        return { home_channels: [] }
      }
      if (url.includes('/tasks/t_review')) {
        return { task: reviewTask, comments: [], events: [], attachments: [], links: { parents: [], children: [] } }
      }
      if (url.includes('/board')) {
        return {
          columns: [
            { name: 'review', tasks: [reviewTask] },
            { name: 'done', tasks: [] }
          ],
          tenants: [],
          assignees: [],
          latest_event_id: 0
        }
      }
      return {}
    })

    Object.assign(window, {
      __HERMES_PLUGIN_SDK__: {
        React,
        components: {
          Card: element('div'),
          CardContent: element('div'),
          Badge: element('span'),
          Button: element('button'),
          Input: element('input'),
          Label: element('label'),
          Select: select,
          SelectOption: element('option')
        },
        hooks: {
          useState: React.useState,
          useEffect: React.useEffect,
          useCallback: React.useCallback,
          useMemo: React.useMemo,
          useRef: React.useRef
        },
        utils: {
          cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
          timeAgo: () => 'now'
        },
        fetchJSON,
        buildWsUrl: async () => 'ws://example.test',
        authedFetch: vi.fn()
      },
      __HERMES_PLUGINS__: {
        register: (_name: string, page: RegisteredPage) => {
          registeredPage = page
        }
      }
    })

    const pluginUrl = pathToFileURL(resolve(process.cwd(), '../../plugins/kanban/dashboard/dist/index.js')).href
    await import(/* @vite-ignore */ pluginUrl)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('offers Complete for a task in review', async () => {
    expect(registeredPage).not.toBeNull()
    render(React.createElement(registeredPage!))

    await screen.findByText('Review task')
    fireEvent.click(screen.getByText('Review task'))

    const complete = await screen.findByRole('button', { name: 'Complete' })
    expect(complete.hasAttribute('disabled')).toBe(false)
  })
})
