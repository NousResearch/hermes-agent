/**
 * A section the user just made has no member yet, and the roster must draw it
 * anyway: its heading is where the next bot gets dropped. With one gateway the
 * flat list always did. With several, sections nest inside each gateway's
 * bucket and a bucket draws only the sections it has rows for, so a section
 * with no row anywhere was drawn nowhere: New section closed its dialog and
 * nothing appeared (#155196). Such a section is drawn once, at the top level
 * below the gateway buckets, until a member seats it inside one.
 *
 * Seated, it is still missing from every other gateway's bucket, which is
 * where a bot of that gateway is listed and dragged from. So while a drag is
 * in flight those buckets draw it too, empty, as the drop target that lets one
 * section gather bots of several gateways.
 *
 * The real Bots pane, with only its gateway transport mocked.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'

const { agents, notify, request, requestProfile } = vi.hoisted(() => ({
  agents: vi.fn(),
  notify: vi.fn(),
  request: vi.fn(),
  requestProfile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: {
      ...sdk.host,
      agents,
      notify,
      request,
      requestProfile,
      state: {
        ...sdk.host.state,
        connectionId: atom<null | string>('local'),
        gateway: atom('open'),
        profile: atom('default')
      }
    },
    usePluginI18n: () => translateBotsIn('en')
  }
})

import { $botMeta, $lastRoster } from './data'
import { botsText } from './i18n'
import { BotsPane } from './roster-pane'
import { $lastSources } from './roster-pane-lifecycle'
import { $botSections, $draggingBot, BOT_DRAG_MIME } from './user-sections'

const en = botsText()

/** A registered connection and the one bot it serves. */
const THIS_DEVICE = { bot: 'default', connectionId: 'local', kind: 'local', label: 'This device' }
const VPS = { bot: 'scout', connectionId: 'vps', kind: 'ssh', label: 'VPS' }

type Gateway = typeof THIS_DEVICE

/** The mounted pane's own node (menus and dialogs portal out of it). */
let pane: HTMLElement

/** The Bots pane over `gateways`: this device is the active one and answers
 *  `profiles.list`; the union roster lists every connection's bot; and each
 *  bot's own gateway takes the profile write that files it into a section. */
async function mountPane(gateways: Gateway[]) {
  request.mockImplementation(async (method: string) => {
    if (method === 'profiles.list') {
      return { profiles: [{ name: THIS_DEVICE.bot }] }
    }

    throw new Error(`unexpected ${method}`)
  })
  requestProfile.mockImplementation(async (route: { profile: string }, method: string) => {
    if (method === 'groups.list') {
      return { rooms: [], next_offset: null }
    }

    if (method === 'profiles.list') {
      return { profiles: [{ name: route.profile }] }
    }

    if (method === 'profiles.configure') {
      return { applied: { ui_meta: true } }
    }

    throw new Error(`unexpected ${method}`)
  })
  agents.mockResolvedValue({
    agents: gateways.map(gateway => ({
      connectionId: gateway.connectionId,
      connectionKind: gateway.kind,
      connectionLabel: gateway.label,
      handle: gateway.bot,
      profile: gateway.bot
    })),
    primaryConnectionId: THIS_DEVICE.connectionId,
    sources: gateways.map(({ connectionId, kind, label }) => ({ connectionId, kind, label, reachable: true }))
  })

  pane = render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <BotsPane />
    </QueryClientProvider>
  ).container
  await waitFor(() => gateways.forEach(gateway => expect(botRow(gateway)).toBeTruthy()))
}

const botRow = (gateway: Gateway) =>
  pane.querySelector<HTMLElement>(`[data-roster-key="${gateway.connectionId}::${gateway.bot}"]`)

/** The fold toggles labelled `label`: a gateway bucket's, or a section's. */
const headings = (label: string) =>
  screen.queryAllByText(label, { selector: 'button[aria-expanded] span' }).map(node => node.closest('button')!)

/** The user sections drawn, top to bottom, as their headings read: the name,
 *  then the member count. */
const sections = () =>
  [...pane.querySelectorAll('[data-slot="bots-section-heading"] button:not([aria-haspopup])')].map(toggle =>
    [...toggle.children]
      .map(part => part.textContent)
      .filter(Boolean)
      .join(' ')
  )

const isBelow = (node: Element, other: Element) =>
  Boolean(other.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)

/** New section, through the pane's + menu and the name dialog. */
async function createSection(name: string) {
  fireEvent.keyDown(screen.getByRole('button', { name: en.roster.newBotOrGroup }), { key: 'Enter' })
  fireEvent.click(await screen.findByRole('menuitem', { name: en.sections.newSection }))
  fireEvent.change(await screen.findByRole('textbox', { name: en.sections.nameLabel }), { target: { value: name } })
  fireEvent.click(screen.getByRole('button', { name: en.sections.create }))
}

/** Pick `item` from a section heading's ⋯ menu. */
async function pickFromSectionMenu(section: string, item: string) {
  fireEvent.keyDown(screen.getByRole('button', { name: en.sections.options(section) }), { key: 'Enter' })
  fireEvent.click(await screen.findByRole('menuitem', { name: item }))
}

/** Just enough DataTransfer for the roster's private-MIME drag. */
function dragPayload() {
  const data = new Map<string, string>()

  return {
    dropEffect: 'none',
    effectAllowed: 'all',
    getData: (type: string) => data.get(type) ?? '',
    setData: (type: string, value: string) => void data.set(type, value),
    get types() {
      return [...data.keys()]
    }
  }
}

/** The blocks drawn for section `name`, top to bottom: each one a drop target. */
const sectionBlocks = (name: string) =>
  headings(name).map(heading => heading.closest<HTMLElement>('[data-slot="bots-section"]')!)

/** Start dragging `gateway`'s bot. The gesture is in flight until it is dropped. */
function drag(gateway: Gateway) {
  const dataTransfer = dragPayload()

  fireEvent.dragStart(botRow(gateway)!, { dataTransfer })

  return dataTransfer
}

function dropOn(target: Element, dataTransfer: ReturnType<typeof dragPayload>) {
  fireEvent.dragOver(target, { dataTransfer })
  fireEvent.drop(target, { dataTransfer })
}

/** The sections `gateway` was told, by profile write, to file its bot under. */
const filedOn = (gateway: Gateway) =>
  requestProfile.mock.calls
    .filter(([route, method]) => route.connectionId === gateway.connectionId && method === 'profiles.configure')
    .map(([, , params]) => params.ui_meta['hermes-bots'].sectionName)

// Radix (Dialog, menus) calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  $botSections.set([])
  $botMeta.set({})
  $lastRoster.set([])
  $lastSources.set([])
  $draggingBot.set(null)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('a section with no member', () => {
  it('is drawn once, below the gateway buckets, when the roster shows several gateways', async () => {
    await mountPane([THIS_DEVICE, VPS])
    await createSection('Clients')

    await waitFor(() => expect(sections()).toEqual(['Clients 0']))
    expect(screen.getAllByText(en.sections.emptyHint)).toHaveLength(1)

    const [clients] = headings('Clients')
    const [thisDevice] = headings(THIS_DEVICE.label)
    const [vps] = headings(VPS.label)

    expect(isBelow(clients!, thisDevice!)).toBe(true)
    expect(isBelow(clients!, vps!)).toBe(true)

    // It belongs to no gateway: folding every bucket leaves it where it is.
    fireEvent.click(thisDevice!)
    fireEvent.click(vps!)
    expect(botRow(THIS_DEVICE)).toBeNull()
    expect(botRow(VPS)).toBeNull()
    expect(sections()).toEqual(['Clients 0'])
  })

  it('is drawn in the flat list of a single gateway, as before', async () => {
    await mountPane([THIS_DEVICE])
    await createSection('Clients')

    await waitFor(() => expect(sections()).toEqual(['Clients 0']))
    expect(screen.getAllByText(en.sections.emptyHint)).toHaveLength(1)
    // One gateway: no bucket heading, the bot sits in the flat list.
    expect(headings(THIS_DEVICE.label)).toHaveLength(0)
    expect(botRow(THIS_DEVICE)).toBeTruthy()
  })

  it('moves into a gateway bucket once a bot of that gateway is dropped on it', async () => {
    await mountPane([THIS_DEVICE, VPS])
    await createSection('Clients')
    await waitFor(() => expect(sections()).toEqual(['Clients 0']))

    const dataTransfer = dragPayload()

    fireEvent.dragStart(botRow(THIS_DEVICE)!, { dataTransfer })
    expect(dataTransfer.types).toEqual([BOT_DRAG_MIME])
    // The drag changes nothing on screen: still that one heading to aim for.
    expect(sections()).toEqual(['Clients 0'])

    const target = headings('Clients')[0]!.closest('[data-slot="bots-section"]')!

    fireEvent.dragOver(target, { dataTransfer })
    fireEvent.drop(target, { dataTransfer })

    // Filed: the one Clients heading now holds the bot, between its gateway's
    // heading and the next gateway's...
    await waitFor(() => expect(sections()).toEqual(['Clients 1']))
    expect(screen.queryByText(en.sections.emptyHint)).toBeNull()

    const [clients] = headings('Clients')
    const [thisDevice] = headings(THIS_DEVICE.label)

    expect(isBelow(clients!, thisDevice!)).toBe(true)
    expect(isBelow(headings(VPS.label)[0]!, clients!)).toBe(true)

    // ...so it folds away with that gateway: no top-level copy is left.
    fireEvent.click(thisDevice!)
    expect(sections()).toEqual([])
  })

  it('keeps its rename, reorder and delete menu at the top level', async () => {
    await mountPane([THIS_DEVICE, VPS])
    await createSection('Clients')
    await waitFor(() => expect(sections()).toEqual(['Clients 0']))
    await createSection('Team')
    await waitFor(() => expect(sections()).toEqual(['Clients 0', 'Team 0']))

    await pickFromSectionMenu('Team', en.sections.moveUp)
    await waitFor(() => expect(sections()).toEqual(['Team 0', 'Clients 0']))

    await pickFromSectionMenu('Clients', en.sections.rename)

    const name = await screen.findByRole('textbox', { name: en.sections.nameLabel })

    fireEvent.change(name, { target: { value: 'Customers' } })
    fireEvent.keyDown(name, { key: 'Enter' })
    await waitFor(() => expect(sections()).toEqual(['Team 0', 'Customers 0']))

    await pickFromSectionMenu('Team', 'Delete')
    await waitFor(() => expect(sections()).toEqual(['Customers 0']))
  })
})

describe('a section with a member on one gateway only', () => {
  /** Clients holds this device's bot, so only that gateway's bucket lists it.
   *  Team holds nothing, so it is listed below the buckets. */
  async function mountWithClientsOnThisDevice() {
    await mountPane([THIS_DEVICE, VPS])
    await createSection('Clients')
    await waitFor(() => expect(sections()).toEqual(['Clients 0']))
    await createSection('Team')
    await waitFor(() => expect(sections()).toEqual(['Clients 0', 'Team 0']))
    dropOn(sectionBlocks('Clients')[0]!, drag(THIS_DEVICE))
    await waitFor(() => expect(sections()).toEqual(['Clients 1', 'Team 0']))
  }

  /** Clients under both gateways, each copy holding that gateway's own bot. */
  async function expectClientsOnBothGateways() {
    await waitFor(() => expect(sections()).toEqual(['Clients 1', 'Clients 1', 'Team 0']))

    const [onThisDevice, onVps] = sectionBlocks('Clients')

    expect(onThisDevice!.contains(botRow(THIS_DEVICE))).toBe(true)
    expect(onVps!.contains(botRow(VPS))).toBe(true)
    // Membership is a field of the bot's profile: its own gateway took the write.
    await waitFor(() => expect(filedOn(VPS)).toEqual(['Clients']))
  }

  it('is offered in every gateway bucket during a drag, and takes a bot of another gateway there', async () => {
    await mountWithClientsOnThisDevice()

    const dataTransfer = drag(VPS)

    // Not in the tick of `dragstart`: a block appearing above the row then
    // moves it from under the pointer, and Chromium cancels such a drag.
    expect(sections()).toEqual(['Clients 1', 'Team 0'])
    // Right after: a second Clients, empty, between the VPS heading and the
    // VPS bot, so inside that bucket. Team, with no member anywhere, is not
    // repeated: its one block below the buckets is already the target.
    await waitFor(() => expect(sections()).toEqual(['Clients 1', 'Clients 0', 'Team 0']))

    const [, offered] = sectionBlocks('Clients')

    expect(isBelow(offered!, headings(VPS.label)[0]!)).toBe(true)
    expect(isBelow(botRow(VPS)!, offered!)).toBe(true)

    dropOn(offered!, dataTransfer)
    await expectClientsOnBothGateways()
  })

  it('takes a bot of another gateway dropped on the block that holds its member', async () => {
    await mountWithClientsOnThisDevice()

    dropOn(sectionBlocks('Clients')[0]!, drag(VPS))
    await expectClientsOnBothGateways()
  })
})
