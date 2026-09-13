import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { createPluginContext } from '@/contrib/plugin'
import { SESSION_AREAS } from '@/contrib/session'
import { $sessionTiles } from '@/store/session-states'
import { $todosBySession } from '@/store/todos'

import { ComposerStatusStack } from './index'

/**
 * A plugin's status row is the one block in the stack with no chrome of its
 * own — no caret, no icon, no label — so under another section it reads as
 * that section's footnote. It gets a hairline for an edge, but ONLY when
 * something precedes it: a lone plugin row must not carry a stray line, and
 * sections that already have headers must not be divided by lines they don't
 * need.
 */
const disposers: Array<() => void> = []

beforeAll(() => {
  vi.stubGlobal(
    'ResizeObserver',
    class {
      disconnect() {}
      observe() {}
    }
  )
})

afterEach(() => {
  cleanup()
  disposers.splice(0).forEach(d => d())
  $sessionTiles.set([])
  $todosBySession.set({})
})

function registerPluginRow() {
  const ctx = createPluginContext('seam-fixture', d => disposers.push(d))
  ctx.register({
    area: SESSION_AREAS.statusStack,
    data: { render: () => <span>Realm mode</span> },
    id: 'row'
  })
}

const renderStack = () =>
  render(
    <MemoryRouter>
      <ComposerStatusStack queue={null} sessionId="runtime-a" />
    </MemoryRouter>
  )

const pluginSlot = () => screen.getByText('Realm mode').closest('[data-session-contribution-content]')

it('gives a plugin row an edge when it follows another section', () => {
  $sessionTiles.set([
    { ownerRoute: { connectionId: 'a', profile: 'a' }, runtimeId: 'runtime-a', storedSessionId: 'stored-a' }
  ])
  $todosBySession.set({ 'runtime-a': [{ content: 'Ship it', id: '1', status: 'in_progress' }] })
  registerPluginRow()

  renderStack()

  expect(pluginSlot()?.className).toContain('border-t')
})

it('leaves a lone plugin row unframed — nothing precedes it to separate from', () => {
  $sessionTiles.set([
    { ownerRoute: { connectionId: 'a', profile: 'a' }, runtimeId: 'runtime-a', storedSessionId: 'stored-a' }
  ])
  registerPluginRow()

  renderStack()

  expect(pluginSlot()?.className).not.toContain('border-t')
})
