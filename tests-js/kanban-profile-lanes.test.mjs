// @vitest-environment jsdom
/* global window, document */
import * as React from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, expect, it, vi } from 'vitest'

let root
let container

afterEach(async () => {
  if (root) await React.act(async () => root.unmount())
  container?.remove()
  root = null
  localStorage.clear()
  vi.unstubAllGlobals()
  delete window.__HERMES_PLUGIN_SDK__
  delete window.__HERMES_PLUGINS__
})

async function renderBoard(assignees, { laneByProfile = true, status = 'running' } = {}) {
  vi.resetModules()
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  const tasks = assignees.map((assignee, i) => ({ id: `task-${i}`, title: `Task ${i}`, assignee, status }))
  const board = { columns: [{ name: status, tasks }], tenants: [], assignees: [], latest_event_id: 0 }
  const fetchJSON = vi.fn(async url => {
    const path = new URL(url, 'http://localhost').pathname
    if (path.endsWith('/config')) return { lane_by_profile: laneByProfile }
    if (path.endsWith('/board')) return board
    if (path.endsWith('/boards')) return { boards: [], current: 'default' }
    if (path.endsWith('/orchestration')) return {}
    if (path.endsWith('/profiles')) return { profiles: [] }
    if (path.endsWith('/projects')) return { projects: [] }
    throw new Error(`Unexpected request ${url}`)
  })
  vi.stubGlobal('WebSocket', class { close() {} })
  window.__HERMES_PLUGIN_SDK__ = {
    React,
    hooks: React,
    components: Object.fromEntries(['Card', 'CardContent', 'Badge', 'Button', 'Input', 'Label', 'Select', 'SelectOption'].map(name => [name, name === 'Input' ? 'input' : 'div'])),
    utils: { cn: (...parts) => parts.filter(Boolean).join(' '), timeAgo: () => '' },
    fetchJSON,
    buildWsUrl: async () => 'ws://localhost/test',
  }
  let Page
  window.__HERMES_PLUGINS__ = { register: (name, component) => { expect(name).toBe('kanban'); Page = component } }
  await import('../plugins/kanban/dashboard/dist/index.js')
  expect(Page).toBeTypeOf('function')
  container = document.createElement('div')
  document.body.append(container)
  root = createRoot(container)
  await React.act(async () => root.render(React.createElement(Page)))
  return tasks
}

it('renders prototype-named assignees in their own sorted lanes without losing cards', async () => {
  const names = ['constructor', 'toString', 'hasOwnProperty', '__proto__', 'constructor', 'worker', null]
  const tasks = await renderBoard(names)
  expect(container.querySelectorAll('[data-task-id]')).toHaveLength(tasks.length)
  const lanes = [...container.querySelectorAll('.hermes-kanban-lane')]
  const expectedNames = [...new Set(names.map(name => name || '(unassigned)'))].sort()
  expect(lanes.map(lane => lane.querySelector('.hermes-kanban-lane-name').textContent)).toEqual(expectedNames)
  for (const [index, lane] of lanes.entries()) {
    const expectedTasks = tasks.filter(task => (task.assignee || '(unassigned)') === expectedNames[index])
    expect([...lane.querySelectorAll('[data-task-id]')].map(card => card.dataset.taskId)).toEqual(expectedTasks.map(task => task.id))
    expect(lane.querySelector('.hermes-kanban-lane-count').textContent).toBe(String(expectedTasks.length))
  }
})

it('keeps an empty running column renderable', async () => {
  await renderBoard([])
  expect(container.querySelector('[data-kanban-column="running"]')).not.toBeNull()
  expect(container.querySelectorAll('[data-task-id]')).toHaveLength(0)
})

it.each([{ laneByProfile: false }, { status: 'ready' }])('preserves flat columns with %j', async options => {
  await renderBoard(['constructor', '__proto__'], options)
  expect(container.querySelectorAll('[data-task-id]')).toHaveLength(2)
  expect(container.querySelectorAll('.hermes-kanban-lane')).toHaveLength(0)
})
