// @vitest-environment jsdom
import React, { act } from 'react'
import { createRoot } from 'react-dom/client'
import { afterEach, expect, test, vi } from 'vitest'

Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true })
const tick = () => new Promise(resolve => setTimeout(resolve, 0))

afterEach(() => {
  document.body.replaceChildren()
  localStorage.clear()
  vi.restoreAllMocks()
})

test('fleet refresh discovers an externally created board without navigation', async () => {
  localStorage.setItem('hermes.kanban.selectedBoard', '__all__')
  let boards = [{ slug: 'default', name: 'Default', total: 1 }]
  const calls: string[] = []

  const snapshot = {
    latest_event_id: 0,
    columns: [{ name: 'running', tasks: [{ id: 't_first', title: 'First', board_slug: 'default' }] }],
    attention: {},
  }

  const pluginWindow = window as typeof window & {
    __HERMES_PLUGIN_SDK__: unknown
    __HERMES_PLUGINS__: unknown
  }

  pluginWindow.__HERMES_PLUGIN_SDK__ = {
    React,
    hooks: React,
    components: Object.fromEntries(
      ['Card', 'CardContent', 'Badge', 'Button', 'Input', 'Label', 'Select', 'SelectOption'].map(name => [name,
        ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => React.createElement(
          ({ Button: 'button', Input: 'input', Select: 'select', SelectOption: 'option' })[name] || 'div',
          props, children,
        ),
      ]),
    ),
    utils: { cn: (...parts: string[]) => parts.filter(Boolean).join(' '), timeAgo: () => '' },
    fetchJSON: async (url: string) => {
      calls.push(url)

      if (url.includes('/boards')) { return { boards, current: 'default' } }

      if (url.includes('/config')) { return {} }

      if (url.includes('/board/all')) { return snapshot }
      throw new Error(`Unexpected request: ${url}`)
    },
  }
  let Page: React.ComponentType = () => null
  pluginWindow.__HERMES_PLUGINS__ = { register: (_name: string, component: React.ComponentType) => { Page = component } }
  // This shipped IIFE registers against the SDK; it does not export a TS module.
  // @ts-expect-error no declaration for the plain dashboard bundle
  await import('../plugins/kanban/dashboard/dist/index.js')
  const root = createRoot(document.body)
  await act(async () => { root.render(React.createElement(Page)); await tick() })
  const source = document.querySelector<HTMLSelectElement>('[aria-label="Filter source board"]')
  expect(source).not.toBeNull()
  expect([...source!.options].map(option => option.value)).not.toContain('external')

  boards = [...boards, { slug: 'external', name: 'External', total: 1 }]
  snapshot.columns[0].tasks.push({ id: 't_external', title: 'External task', board_slug: 'external' })
  await act(async () => {
    [...document.querySelectorAll('button')].find(button => button.textContent === 'Refresh')!.click()
    await tick()
  })
  expect([...source!.options].map(option => option.value)).toContain('external')
  expect([...document.querySelector<HTMLSelectElement>('[aria-label="Switch kanban board"]')!.options]
    .map(option => option.value)).toContain('external')
  expect(calls.filter(url => url.includes('/boards')).length).toBeGreaterThan(1)
  await act(async () => root.unmount())
})
