import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { TreeGroup } from '@/components/pane-shell/tree/renderer/tree-group'
import { $layoutTree, $narrowViewport } from '@/components/pane-shell/tree/store'
import { registry } from '@/contrib/registry'
import { $sessions } from '@/store/session'
import { $sessionTiles, type SessionTileDelegate, setSessionTileDelegate } from '@/store/session-states'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'
import type { SessionInfo } from '@/types/hermes'

import { watchSessionTiles } from './session-tile'

// The transport: the REST rename the dialog falls back to with no live runtime.
const renameSession = vi.hoisted(() => vi.fn())

vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal<object>()), renameSession }))

const row = (id: string, title: string): SessionInfo => ({
  cwd: null,
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 2,
  model: null,
  output_tokens: 0,
  parent_session_id: null,
  preview: null,
  source: 'desktop',
  started_at: 1,
  title,
  tool_call_count: 0
})

const panes = ['workspace', 'session-tile:s2', 'session-tile:s3']
const delegate = { archiveSession: vi.fn(async () => {}), deleteSession: vi.fn(async () => {}) }

let disposeWorkspace = () => {}

function Host() {
  const node = useStore($layoutTree)

  return <MemoryRouter>{node?.type === 'group' && <TreeGroup node={node} />}</MemoryRouter>
}

/** Opens the phone picker, then the given tab's own menu from its row. */
function openTabMenu(title: string) {
  fireEvent.pointerDown(screen.getByRole('button', { name: /^3 tabs/ }), { button: 0, ctrlKey: false })
  fireEvent.click(screen.getByRole('menuitem', { name: `Actions for ${title}` }))
}

beforeAll(() => {
  // Real tile mirror: the tabs register with the same chrome a Webapp gets.
  $sessionTiles.set([{ storedSessionId: 's2' }, { storedSessionId: 's3' }])
  watchSessionTiles()
})

beforeEach(() => {
  stubResizeObserver()
  stubMenuDomApis()
  vi.stubGlobal('CSS', { escape: (value: string) => value })
  vi.stubGlobal(
    'matchMedia',
    vi.fn().mockReturnValue({ matches: true, addEventListener: vi.fn(), removeEventListener: vi.fn() })
  )
  globalThis.document.documentElement.dataset.hermesDesktopHost = 'browser'
  disposeWorkspace = registry.register({
    area: 'panes',
    id: 'workspace',
    title: 'Main chat',
    data: { placement: 'main' },
    render: () => <p>main chat</p>
  })
  $sessions.set([row('s2', 'Trip plan'), row('s3', 'Old draft')])
  setSessionTileDelegate(delegate as unknown as SessionTileDelegate)
  renameSession.mockResolvedValue({ title: 'Trip plan v2' })
  $layoutTree.set({ ...group(panes), active: 'workspace', tabStrip: 'always' })
  $narrowViewport.set(true)
})

afterEach(() => {
  cleanup()
  disposeWorkspace()
  $layoutTree.set(null)
  $narrowViewport.set(false)
  delete globalThis.document.documentElement.dataset.hermesDesktopHost
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

it('archives and deletes a background tab from the phone picker, delete behind its confirm', async () => {
  render(<Host />)

  openTabMenu('Trip plan')
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Archive' }))
  expect(delegate.archiveSession).toHaveBeenCalledWith('s2')

  openTabMenu('Old draft')
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }))
  expect(delegate.deleteSession).not.toHaveBeenCalled()
  fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Delete' }))
  await waitFor(() => expect(delegate.deleteSession).toHaveBeenCalledWith('s3'))

  // Neither verb touched the tab the user was on.
  expect($layoutTree.get()).toMatchObject({ active: 'workspace' })
})

it('renames a background tab from the phone picker', async () => {
  render(<Host />)

  openTabMenu('Trip plan')
  fireEvent.click(await screen.findByRole('menuitem', { name: 'Rename…' }))
  const input = await screen.findByRole('textbox')
  fireEvent.change(input, { target: { value: 'Trip plan v2' } })
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Save' })))

  expect(renameSession).toHaveBeenCalledWith('s2', 'Trip plan v2', undefined)
  await waitFor(() => expect($sessions.get().find(s => s.id === 's2')?.title).toBe('Trip plan v2'))
})
