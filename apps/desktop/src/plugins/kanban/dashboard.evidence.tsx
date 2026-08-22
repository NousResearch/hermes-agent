import React from 'react'
import { createRoot } from 'react-dom/client'

const element = (tag: string) => ({ children, ...props }: Record<string, unknown>) => React.createElement(tag, props, children as React.ReactNode)
const Select = ({ children, onValueChange, value, ...props }: Record<string, unknown>) => React.createElement('select', { ...props, onChange: (event: React.ChangeEvent<HTMLSelectElement>) => (onValueChange as (value: string) => void)?.(event.target.value), value }, children as React.ReactNode)
const task = { assignee: 'canary-worker', attention: { reason: 'receipt', revision: 0, state: 'active', wake_at: null }, id: 'task-safe', status: 'running', title: 'Privacy-safe evidence task' }
const board = { assignees: ['canary-worker'], columns: [{ name: 'running', tasks: [task] }, { name: 'review', tasks: [{ ...task, id: 'task-review', status: 'review', title: 'Production review task' }] }], latest_event_id: 2, now: 1_800_000_000, tenants: [] }

const fetchJSON = async (url: string) => {
  if (url.includes('/config')) return { include_archived_by_default: false, lane_by_profile: false, render_markdown: true }
  if (url.includes('/boards')) return { boards: [{ name: 'Default', slug: 'default' }], current: 'default' }
  if (url.includes('/board')) return board
  throw new Error(`Unexpected responsive evidence request: ${url}`)
}

Object.assign(window, {
  __HERMES_PLUGIN_SDK__: {
    React,
    buildWsUrl: async () => 'ws://127.0.0.1/disabled',
    components: { Badge: element('span'), Button: element('button'), Card: element('section'), CardContent: element('div'), Input: element('input'), Label: element('label'), Select, SelectOption: element('option') },
    fetchJSON,
    hooks: { useCallback: React.useCallback, useEffect: React.useEffect, useMemo: React.useMemo, useRef: React.useRef, useState: React.useState },
    utils: { cn: (...parts: unknown[]) => parts.filter(Boolean).join(' '), timeAgo: () => 'now' }
  },
  __HERMES_PLUGINS__: {
    register: (_name: string, Component: React.ComponentType) => createRoot(document.getElementById('root')!).render(React.createElement(Component))
  }
})

const script = document.createElement('script')
script.src = '/dashboard-production/index.js'
document.body.append(script)
