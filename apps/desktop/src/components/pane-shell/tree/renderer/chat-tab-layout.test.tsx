import { act, cleanup, render } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, expect, it, vi } from 'vitest'

import { registry } from '@/contrib/registry'
import { stubResizeObserver } from '@/test/jsdom'
import type { GroupNode } from '../model'
import { TreeGroup } from './tree-group'

const disposers: (() => void)[] = []
afterEach(() => { cleanup(); disposers.splice(0).forEach(dispose => dispose()); vi.unstubAllGlobals() })

it('takes hidden chats out of layout without discarding their component or draft', () => {
  vi.stubGlobal('CSS', { escape: (value: string) => value })
  stubResizeObserver()
  const tile = 'session-tile:stored-chat'
  for (const id of ['workspace', tile, 'other']) {
    disposers.push(registry.register({ area: 'panes', id, title: id, data: { placement: 'main' },
      render: () => <textarea data-probe={id} defaultValue="draft" /> }))
  }
  const group: GroupNode = { type: 'group', id: 'chat-layout-test', active: 'workspace',
    panes: ['workspace', tile, 'other'], tabStrip: 'always' }
  const pane = (active: string) => <MemoryRouter><TreeGroup node={{...group, active}} /></MemoryRouter>
  const view = render(pane('workspace'))
  const input = view.container.querySelector<HTMLTextAreaElement>('[data-probe="workspace"]')!
  input.value = 'unsent draft remains'
  view.rerender(pane(tile))
  const hidden = input.closest('[data-pane-hidden]')!
  expect(hidden).toBeTruthy()
  expect(hidden.classList.contains('hidden')).toBe(true)
  expect(hidden.hasAttribute('inert')).toBe(true)
  view.rerender(pane('workspace'))
  expect(view.container.querySelector('[data-probe="workspace"]')).toBe(input)
  expect(input.value).toBe('unsent draft remains')
  expect(input.closest('[data-pane-hidden]')).toBeNull()
  const hiddenTile = view.container.querySelector(`[data-probe="${tile}"]`)!.closest('[data-pane-hidden]')!
  expect(hiddenTile.classList.contains('hidden')).toBe(true)
  view.rerender(pane('other'))
  view.rerender(pane('workspace'))
  const other = view.container.querySelector('[data-probe="other"]')!.closest('[data-pane-hidden]')!
  expect(other.classList.contains('hidden')).toBe(false)
  act(() => view.unmount())
})
