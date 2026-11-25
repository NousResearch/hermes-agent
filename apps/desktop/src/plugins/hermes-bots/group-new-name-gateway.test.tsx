/**
 * A name with no room record is a new group chat, and every new group chat runs on a gateway:
 * Manage groups' Create & join goes through New group chat's gateway create, and the workspace
 * never opens such a name Desktop-driven (nor saves `engine: 'desktop'` on an edit), whether the
 * gateway is current or too old to run group chats.
 */

import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $canonicalGroupBindings } from './canonical-group-registry'
import { GroupDialog } from './create-dialog'
import { CROSS_GATEWAY_GROUP_LOCALES } from './cross-gateway-locales'
import { $botMeta, $lastRoster } from './data'
import { $groupChats, $groupChatWorkspace } from './group-chat'
import type * as GroupChatModule from './group-chat'
import type * as GroupChatParts from './group-chat-parts'
import { GroupChatWorkspace } from './group-chat-view'
import { translateBots } from './i18n-test-helper'
import type { RosterRow } from './types'
import { moveGroupChatsToSection } from './user-sections'

const { request, notify, openWorkspace } = vi.hoisted(() => ({
  request: vi.fn(),
  notify: vi.fn(),
  openWorkspace: vi.fn()
}))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()
  const { en } = await import('@/i18n/en')

  return {
    ...sdk,
    gatewayActivationEpoch: () => 1,
    useGatewayActivationEpoch: () => 1,
    host: {
      ...sdk.host,
      requestProfile: request,
      notify,
      openWorkspace,
      request: (method: string, params?: Record<string, unknown>) => request(null, method, params),
      connections: vi.fn(async () => []),
      state: {
        ...sdk.host.state,
        connectionId: sdk.atom<string | null>('local'),
        profile: sdk.atom('default'),
        gateway: sdk.atom('open')
      }
    },
    useI18n: () => ({ locale: 'en', t: en }),
    usePluginI18n: () => translateBots
  }
})
vi.mock('./group-chat', async importOriginal => {
  const actual = await importOriginal<typeof GroupChatModule>()

  return {
    ...actual,
    updateGroupChat: vi.fn((group, mutate) => actual.updateGroupChat(group, mutate, { sync: false }))
  }
})
vi.mock('./group-chat-parts', async importOriginal => ({
  ...(await importOriginal<typeof GroupChatParts>()),
  GroupImageControls: () => null
}))

const alpha: RosterRow = { name: 'alpha', connectionId: 'local', connectionLabel: 'This device' }
const beta: RosterRow = { name: 'beta', connectionId: 'local', connectionLabel: 'This device' }
const remote: RosterRow = { name: 'scout', connectionId: 'other-box', connectionLabel: 'Other box', remoteSource: true }

function gateway(capabilities: unknown) {
  request.mockReset().mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {
      return capabilities
    }

    if (method === 'groups.list') {
      return { rooms: [], next_offset: null }
    }

    if (method === 'groups.create') {
      return { room: { room_id: params.room_id, name: params.name, members: params.members } }
    }

    return {}
  })
}

beforeEach(() => {
  gateway({ driver: true, persistent_process: true, seatable_profiles: ['alpha', 'beta'] })
  $canonicalGroupBindings.set({})
  $groupChats.set({})
  $groupChatWorkspace.set(null)
  $botMeta.set({})
  $lastRoster.set([alpha, beta, remote])
  notify.mockReset()
  openWorkspace.mockReset().mockReturnValue(() => undefined)
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  $groupChats.set({})
  localStorage.clear()
})

const desktopRecords = () => Object.values($groupChats.get()).filter(room => room.engine === 'desktop')

it("Create & join from a bot's Manage groups makes a gateway-run group chat, never a Desktop-run one", async () => {
  const onCreated = vi.fn()
  render(<GroupDialog bot={alpha} onClose={vi.fn()} onCreated={onCreated} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Research' } })
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Create & join' }))
  })

  // New group chat opens with this bot picked and the name filled in; the person adds the others.
  const dialog = await screen.findByRole('dialog', { name: 'New group chat' })
  expect((within(dialog).getByRole('textbox', { name: 'Group name' }) as HTMLInputElement).value).toBe('Research')
  await waitFor(() => expect(dialog.querySelector('[data-bot-face="beta"]')).not.toBeNull())
  fireEvent.click(dialog.querySelector('[data-bot-face="beta"]')!.closest('label')!.querySelector('[role="checkbox"]')!)
  const create = within(dialog).getByRole('button', { name: 'Create group (2)' }) as HTMLButtonElement
  await waitFor(() => expect(create.disabled).toBe(false))
  await act(async () => {
    fireEvent.click(create)
  })

  await waitFor(() => expect(onCreated).toHaveBeenCalledOnce())
  const created = request.mock.calls.filter(call => call[1] === 'groups.create')
  expect(created).toHaveLength(1)
  expect(created[0][2]).toMatchObject({ name: 'Research', members: [{ profile: 'alpha' }, { profile: 'beta' }] })
  expect($groupChats.get().Research.engine).toMatchObject({ kind: 'gateway', connectionId: 'local' })
  expect(desktopRecords()).toEqual([])
})

it('a name with no room record never opens Desktop-run, even when its bots span computers', async () => {
  // A group known only from Bots' memberships (what Create & join used to leave behind).
  await act(async () => {
    render(<GroupChatWorkspace group="Research" members={[alpha, remote]} />)
  })

  await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.list'))
  expect(screen.queryByRole('textbox', { name: 'Message Research' })).toBeNull()
  expect(desktopRecords()).toEqual([])
})

it('a new name on a gateway too old to run group chats says to update it, as New group chat does', async () => {
  gateway({ driver: false, persistent_process: false })
  await act(async () => {
    render(<GroupChatWorkspace group="Research" members={[alpha, beta]} />)
  })

  expect(await screen.findByText(CROSS_GATEWAY_GROUP_LOCALES.en.ownerUpdateRequired('This device'))).toBeTruthy()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
  expect(screen.queryByRole('textbox')).toBeNull()
  expect($groupChats.get()).toEqual({})
})

it.each(['section', 'order'])(
  'a membership-only name filed by a layout write (%s) still never opens Desktop-run',
  async write => {
    // Filing a group into a section or reordering it writes an empty record for the name.
    if (write === 'section') {
      moveGroupChatsToSection(['Research'], 'work')
    } else {
      $groupChats.set({ Research: { log: [], watermarks: {}, epoch: 0, running: false, rosterOrder: 0 } })
    }

    expect($groupChats.get().Research).toBeTruthy()
    await act(async () => {
      render(<GroupChatWorkspace group="Research" members={[alpha, remote]} />)
    })

    await waitFor(() => expect(request.mock.calls.map(call => call[1])).toContain('groups.list'))
    expect(screen.queryByRole('textbox', { name: 'Message Research' })).toBeNull()
    expect(desktopRecords()).toEqual([])
  }
)
