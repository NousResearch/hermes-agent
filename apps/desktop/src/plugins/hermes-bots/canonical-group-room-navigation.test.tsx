import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { type ReactNode, useSyncExternalStore } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

// A confirmed rename or end updates every visible projection of the room: the gateway group
// list, each open tab's label and the bindings. The tabs are what `host.openWorkspace` receives,
// labelled the way the tab strip labels a pane (its live `tabTitle`, else `title`); the SDK test
// checks the host hands that label to the strip. Stubbed: the gateway (`requestProfile`), the
// main-window door, the workspace scope, the pane-visibility probe and i18n.
interface Tab { id: string; options: { title?: string; tabTitle?: () => ReactNode; render: () => ReactNode; onClose?: () => void } }

const { request, workspace } = vi.hoisted(() => {
  let tabs: Tab[] = []
  const listeners = new Set<() => void>()

  const set = (next: Tab[]) => {
    tabs = next
    listeners.forEach(listener => listener())
  }

  return {
    request: vi.fn(),
    workspace: {
      tabs: () => tabs,
      subscribe: (listener: () => void) => {
        listeners.add(listener)

        return () => { listeners.delete(listener) }
      },
      // As the host does: the same id is refreshed in place, and closing fires onClose.
      open: vi.fn((id: string, options: Tab['options']) => {
        set(tabs.some(tab => tab.id === id) ? tabs.map(tab => tab.id === id ? { id, options } : tab) : [...tabs, { id, options }])

        return () => {
          set(tabs.filter(tab => tab.id !== id))
          options.onClose?.()
        }
      })
    }
  }
})

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => 1,
    host: {
      ...sdk.host,
      requestProfile: request,
      openWorkspace: workspace.open,
      setWorkspaceScope: vi.fn(),
      // No visibility probe: an open room counts as on screen, so it polls.
      paneVisibility: undefined,
      state: { ...sdk.host.state, connectionId: sdk.atom<string | null>('local'), profile: sdk.atom('team'), gateway: sdk.atom('open') }
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots
  }
})
import { $canonicalGroupBindings, canonicalGroupAliasKey, CanonicalGroupList, canonicalGroupTitle, registerCanonicalGroup } from './canonical-group-registry'
import { openGroupChat } from './group-chat-view'
import { closeGroupChatMainTab, groupChatMainTabs } from './group-panes'
import { translateBots } from './i18n-test-helper'

interface Room { room_id: string; name: string; members: [] }
let rooms: Room[]
let ends: Array<'refused' | 'unconfirmed' | 'tombstone'>
let unansweredRenames: number
let heldList: null | (() => void)

// Room ids are never reused, and an ended one stays ended for the session: a fresh id per test.
let planId = ''

beforeEach(() => {
  planId = crypto.randomUUID()
  rooms = [{ room_id: planId, name: 'Planning', members: [] }]
  ends = ['tombstone']
  unansweredRenames = 0
  heldList = null
  $canonicalGroupBindings.set({})
  Element.prototype.scrollIntoView = vi.fn()
  request.mockImplementation(async (_route, method: string, params: Record<string, unknown> = {}) => {
    const room = rooms.find(r => r.room_id === params.room_id)

    if (method === 'groups.capabilities') {return { driver: true, persistent_process: true }}

    if (method === 'groups.list') {
      const answer = { rooms: rooms.map(r => ({ ...r })), next_offset: null }

      return heldList ? new Promise(resolve => { heldList = () => resolve(answer) }) : answer
    }

    if (method === 'groups.state' && room) {return { room: { ...room, latest_seq: 0 }, driver_status: {} }}

    if (method === 'groups.log') {return { events: [] }}

    if (method === 'groups.rename' && room) {
      if (unansweredRenames-- > 0) {throw new Error('request timed out: groups.rename')}
      room.name = String(params.name)

      return { room }
    }

    if (method === 'groups.disband' && room) {
      const end = ends.shift()

      if (end === 'refused') {throw new Error('room work is still stopping; retry deletion after Stop completes')}

      if (end !== 'tombstone') {return {}}
      rooms = rooms.filter(r => r !== room)

      return { tombstone: { room_id: room.room_id, disbanded_at: 1_792_152_000 } }
    }

    throw new Error(`Unexpected RPC: ${method} ${JSON.stringify(params)}`)
  })
})
afterEach(() => {
  vi.useRealTimers()
  act(() => { for (const group of [...groupChatMainTabs.keys()]) {closeGroupChatMainTab(group)} })
  cleanup()
  request.mockReset()
  workspace.open.mockClear()
  localStorage.clear()
})

const groupPanes = () => workspace.tabs()

const TabView = ({ view }: { view: () => ReactNode }) => view()

/** The open room tabs as the tab strip shows them: each one's live label, else its opening title. */
function RoomTabs() {
  const tabs = useSyncExternalStore(workspace.subscribe, workspace.tabs)

  return <>{tabs.map(({ id, options }) => <section aria-label={id} key={id}>
    <div role="tab">{options.tabTitle ? <TabView view={options.tabTitle} /> : options.title}</div>
    <div role="tabpanel"><TabView view={options.render} /></div>
  </section>)}</>
}

// A dialog hides the rest of the page from the accessibility tree, so these look past it.
const tabLabels = () => screen.queryAllByRole('tab', { hidden: true }).map(tab => tab.textContent)

const listed = () => within(screen.getByRole('navigation', { hidden: true }))
  .queryAllByRole('button', { hidden: true }).map(button => button.textContent)

const panel = (match: (paneId: string) => boolean) =>
  within(screen.getAllByRole('region', { hidden: true }).find(region => match(region.getAttribute('aria-label') ?? ''))!)

async function showRooms() {
  render(<>
    <nav aria-label="Gateway groups"><CanonicalGroupList onOpen={openGroupChat} /></nav>
    <RoomTabs />
  </>)
  await within(screen.getByRole('navigation')).findByRole('button', { name: 'Planning' })
}

async function openFromList(name: string) {
  fireEvent.click(within(screen.getByRole('navigation')).getByRole('button', { name }))
  await within(await screen.findByRole('tabpanel')).findByRole('heading', { name })
}

async function rename(view: ReturnType<typeof within>, name: string) {
  fireEvent.click(await view.findByRole('button', { name: 'Rename' }))
  const input = await screen.findByRole('textbox', { name: 'Room name' }) as HTMLInputElement
  expect(input.value).toBe('Planning')
  // Create's limit: a rename never gives a room a name Create could not.
  expect(input.maxLength).toBe(64)
  fireEvent.change(input, { target: { value: name } })

  // An unanswered Save keeps the dialog open with its error, and Save sends the same rename again.
  for (let unanswered = unansweredRenames; unanswered > 0; unanswered--) {
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))
    await screen.findByText('request timed out: groups.rename')
  }

  fireEvent.click(screen.getByRole('button', { name: 'Save' }))
  await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Room name' })).toBeNull())
}

const sent = (method: string) => request.mock.calls.filter(call => call[1] === method).map(call => call[2])

async function typeDraft(view: ReturnType<typeof within>, text: string) {
  const composer = view.getByRole('textbox', { name: 'Group message' }) as HTMLTextAreaElement
  await waitFor(() => expect(composer.disabled).toBe(false))
  fireEvent.change(composer, { target: { value: text } })

  return composer
}

it('retitles the open tab and the group list in place when the gateway confirms a rename', async () => {
  unansweredRenames = 1
  await showRooms()
  await openFromList('Planning')
  expect(tabLabels()).toEqual(['Planning'])
  const opened = groupPanes()[0]
  const composer = await typeDraft(within(screen.getByRole('tabpanel')), 'Keep this draft')

  await rename(within(screen.getByRole('tabpanel')), 'Launch')

  await waitFor(() => expect(tabLabels()).toEqual(['Launch']))
  expect(listed()).toEqual(['Refresh gateway groups', 'Launch'])
  // The label follows on its own: the tab is neither opened again nor brought to the front.
  expect(workspace.open).toHaveBeenCalledOnce()
  expect(groupPanes()).toEqual([opened])
  expect(composer.isConnected && composer.value).toBe('Keep this draft')
  // One event id per intended name.
  expect(sent('groups.rename')).toEqual([expect.objectContaining({ room_id: planId, name: 'Launch' }), sent('groups.rename')[0]])
})

it('brings a rename made on another surface to the open tab and the list on the next poll', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ['setTimeout', 'clearTimeout'] })
  await showRooms()
  await openFromList('Planning')
  rooms[0].name = 'Renamed elsewhere'

  await act(() => vi.advanceTimersByTimeAsync(2_000))
  await waitFor(() => expect(tabLabels()).toEqual(['Renamed elsewhere']))
  expect(listed()).toEqual(['Refresh gateway groups', 'Renamed elsewhere'])
})

it('keeps a view open by the old name on its room, and opens a name only on the room that has it now', async () => {
  await showRooms()
  act(() => openGroupChat('Planning', 'name'))
  const byName = within(await screen.findByRole('tabpanel'))
  await byName.findByRole('heading', { name: 'Planning' })
  const composer = await typeDraft(byName, 'Keep this draft')

  await rename(byName, 'Launch')
  await waitFor(() => expect(tabLabels()).toEqual(['Launch']))
  expect(byName.getByRole('heading', { name: 'Launch' })).toBeDefined()
  expect(composer.isConnected && composer.value).toBe('Keep this draft')

  // A newer room now has the old name: opening that name reaches it, not the renamed room.
  rooms.push({ room_id: 'newer', name: 'Planning', members: [] })
  act(() => openGroupChat('Planning', 'name'))
  await waitFor(() => expect(tabLabels()).toEqual(['Planning']))
  await within(screen.getByRole('tabpanel')).findByRole('heading', { name: 'Planning' })
  expect(request.mock.calls.filter(call => call[1] === 'groups.state').at(-1)?.[2]).toMatchObject({ room_id: 'newer' })

  // And the new name opens the renamed room.
  act(() => openGroupChat('Launch', 'name'))
  await waitFor(() => expect(tabLabels()).toEqual(['Planning', 'Launch']))
  await panel(id => id.endsWith(':launch')).findByRole('heading', { name: 'Launch' })
  expect(request.mock.calls.filter(call => call[1] === 'groups.state').at(-1)?.[2]).toMatchObject({ room_id: planId })
})

it('keeps the room listed until the gateway returns its tombstone, then drops it from the list at once', async () => {
  ends = ['refused', 'unconfirmed', 'tombstone']
  await showRooms()
  await openFromList('Planning')
  fireEvent.click(screen.getByRole('button', { name: 'End group chat' }))
  await screen.findByText(/End this group chat permanently\?/)
  const confirm = async () => fireEvent.click((await screen.findAllByRole('button', { name: 'End group chat' })).at(-1)!)

  for (const attempt of [1, 2]) {
    await confirm()
    await waitFor(() => expect(sent('groups.disband')).toHaveLength(attempt))
    await screen.findByText('We couldn’t confirm that the group chat ended. Check its status before trying again.')
    expect(listed()).toEqual(['Refresh gateway groups', 'Planning'])
    expect(groupPanes()).toHaveLength(1)
  }

  await confirm()
  await waitFor(() => expect(groupPanes()).toEqual([]))
  expect(listed()).toEqual(['Refresh gateway groups'])
  expect($canonicalGroupBindings.get()).toEqual({})
  // One cancellation id per End, across its attempts.
  expect(new Set(sent('groups.disband').map(params => params.cancel_id)).size).toBe(1)
})

it('closes every tab of an ended room, by name or by key, and a list already in flight never brings it back', async () => {
  await showRooms()
  act(() => openGroupChat('Planning', 'name'))
  await within(await screen.findByRole('tabpanel')).findByRole('heading', { name: 'Planning' })
  fireEvent.click(within(screen.getByRole('navigation')).getByRole('button', { name: 'Planning' }))
  await waitFor(() => expect(groupPanes()).toHaveLength(2))
  const byKey = panel(id => id.includes(':group:canonical'))
  await byKey.findByRole('heading', { name: 'Planning' })

  // A Refresh whose answer still lists the room arrives after the end.
  heldList = () => undefined
  const lists = request.mock.calls.filter(call => call[1] === 'groups.list').length
  fireEvent.click(within(screen.getByRole('navigation')).getByRole('button', { name: 'Refresh gateway groups' }))
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.list')).toHaveLength(lists + 1))
  fireEvent.click(byKey.getByRole('button', { name: 'End group chat' }))
  fireEvent.click((await screen.findAllByRole('button', { name: 'End group chat' })).at(-1)!)

  await waitFor(() => expect(groupPanes()).toEqual([]))
  expect(groupChatMainTabs.size).toBe(0)
  await act(async () => { heldList?.() })
  expect(listed()).toEqual(['Refresh gateway groups'])
  expect($canonicalGroupBindings.get()).toEqual({})
})

it('names every alias of a room after any read of it, a list read included', () => {
  const here = { connectionId: 'local', profile: 'team' }
  const alias = canonicalGroupAliasKey(here, 'Planning')
  $canonicalGroupBindings.set({ [alias]: { ...here, roomId: planId, roomName: 'Planning' } })
  registerCanonicalGroup(here, { room_id: planId, name: 'Launch', members: [] })
  expect($canonicalGroupBindings.get()[alias].roomName).toBe('Launch')
})

it('names a tab from the binding of its own source only', () => {
  const here = { connectionId: 'local', profile: 'team' }
  const binding = { ...here, roomId: 'plan', roomName: 'Launch' }
  const bindings = { [canonicalGroupAliasKey(here, 'Planning')]: binding }

  expect(canonicalGroupTitle(bindings, here, 'Planning', 'name')).toBe('Launch')
  expect(canonicalGroupTitle(bindings, { ...here, profile: 'other' }, 'Planning', 'name')).toBe('Planning')
  // A name that spells a key is still a name: it never borrows the keyed room's title.
  expect(canonicalGroupTitle({ 'canonical:local:team:plan': binding }, here, 'canonical:local:team:plan', 'name')).toBe('canonical:local:team:plan')
  expect(canonicalGroupTitle({ 'canonical:local:team:plan': binding }, here, 'canonical:local:team:plan', 'key')).toBe('Launch')
})
