/** Real route/capability/read components, with only the transport boundary replaced. */
import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { GroupAttachmentDownload } from '../plugins/hermes-bots/group-attachment-download'
import { $groupChats } from '../plugins/hermes-bots/group-chat'
import {
  deferred,
  FILE_ROOM,
  fileItem,
  filePage,
  parsedFilePage
} from '../plugins/hermes-bots/group-files-test-fixtures'
import { SharedFilesControl } from '../plugins/hermes-bots/group-files-view'
import { $hostedRoomCapabilities } from '../plugins/hermes-bots/hosted-room-capability-state'
import { classifyHostedRoomCapability } from '../plugins/hermes-bots/hosted-room-client'
import { hostedRouteForRoom, stopHostedRoomRuntime } from '../plugins/hermes-bots/hosted-room-runtime'
import { translateBots } from '../plugins/hermes-bots/i18n-test-helper'
import type { GroupChat, GroupMessage } from '../plugins/hermes-bots/types'

import { HermesGateway } from './client'

const mocks = vi.hoisted(() => ({ notify: vi.fn(), requestProfile: vi.fn(), profileRoutes: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, ...mocks }, usePluginI18n: () => translateBots }
})
const route = { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' }

const capability = {
  authority_gateway_id: 'install:home',
  driver: true,
  persistent_process: true,
  features: ['attachment_metadata_catalog'],
  methods: ['groups.attachment.list', 'groups.attachment.read']
}

const lists = () => mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.attachment.list')
const reads = () => mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.attachment.read')
const saved: Array<{ name: string; href: string }> = []

beforeEach(() => {
  vi.clearAllMocks()
  saved.length = 0
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ name: this.download, href: this.href })
  })
  $groupChats.set({ Core: FILE_ROOM })
  $hostedRoomCapabilities.set({ 'gateway-a': classifyHostedRoomCapability(capability, { connectionId: 'gateway-a' }) })
  mocks.profileRoutes.mockResolvedValue([route])
  mocks.requestProfile.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.capabilities') {
      return capability
    }

    if (method === 'groups.attachment.list') {
      return params.cursor ? filePage([fileItem(19)]) : filePage([fileItem(20)], true)
    }

    if (method === 'groups.attachment.read') {
      return { attachment: { ...fileItem(20) }, content_base64: 'YQ==' }
    }

    throw new Error(`Unexpected RPC ${method}`)
  })
})
afterEach(() => {
  cleanup()
  stopHostedRoomRuntime()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

async function open(room = FILE_ROOM) {
  const view = render(<SharedFilesControl group="Core" room={room} />)
  fireEvent.click(screen.getByRole('button', { name: 'Files' }))

  return view
}

async function miss() {
  await act(async () =>
    $hostedRoomCapabilities.set({
      'gateway-a': classifyHostedRoomCapability(
        { ok: false, error: new Error('offline') },
        { connectionId: 'gateway-a' }
      )
    })
  )
}

describe('actual capability path: preserved counterexamples and C7 repair', () => {
  it.each([
    ['feature', false],
    ['feature', true],
    ['authority', false],
    ['authority', true]
  ] as const)('recovers usable downloads after %s loss with older=%s', async (fault, older) => {
    let currentCapability: typeof capability = capability
    const original = mocks.requestProfile.getMockImplementation()!
    mocks.requestProfile.mockImplementation(async (...args) =>
      args[1] === 'groups.capabilities' ? currentCapability : original(...args)
    )
    await open()
    await screen.findByText('file-20.txt')

    if (older) {
      fireEvent.click(screen.getByRole('button', { name: 'Older' }))
      await screen.findByText('file-19.txt')
    }

    currentCapability =
      fault === 'feature' ? { ...capability, features: [] } : { ...capability, authority_gateway_id: 'install:other' }
    await act(async () => {
      await hostedRouteForRoom(FILE_ROOM, 'read')
    })
    expect(screen.queryByRole('listitem')).toBeNull()
    currentCapability = capability
    await act(async () => {
      await hostedRouteForRoom(FILE_ROOM, 'read')
    })
    const name = older ? 'file-19.txt' : 'file-20.txt'
    await screen.findByText(name)
    await waitFor(() => expect(lists()).toHaveLength(older ? 3 : 2))
    const response = { attachment: fileItem(older ? 19 : 20), content_base64: 'YQ==' }
    const beforeRead = mocks.requestProfile.getMockImplementation()!
    mocks.requestProfile.mockImplementation(async (...args) =>
      args[1] === 'groups.attachment.read' ? response : beforeRead(...args)
    )
    fireEvent.click(screen.getByRole('button', { name: `Download ${name}` }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(lists().at(-1)?.[2].cursor).toBe(older ? 'cursor-after-20' : undefined)
  })

  it('settled list timeout cancels late route work without requiring Escape', async () => {
    vi.useFakeTimers()
    const pending = deferred<(typeof route)[]>()
    mocks.profileRoutes.mockReturnValue(pending.promise)
    await open()
    await act(async () => vi.advanceTimersByTimeAsync(10_001))
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy()
    await act(async () => {
      pending.resolve([route])
      await Promise.resolve()
    })
    expect(mocks.requestProfile).not.toHaveBeenCalled()
  })

  it('one miss/recovery keeps an older page and focus without a fresh list request', async () => {
    await open()
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await screen.findByText('file-19.txt')
    const row = screen.getByRole('listitem')
    row.focus()
    await miss()
    expect(screen.queryByText('Files are temporarily unavailable.')).toBeNull()
    expect(screen.getByText('file-19.txt')).toBeTruthy()
    await act(async () =>
      $hostedRoomCapabilities.set({
        'gateway-a': classifyHostedRoomCapability(capability, { connectionId: 'gateway-a' })
      })
    )
    expect(lists()).toHaveLength(2)
    expect(globalThis.document.activeElement).toBe(row)
  })

  it('Retry itself re-probes and fetches the held page, with no redundant first-page read', async () => {
    await open()
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Older' }))
    await screen.findByText('file-19.txt')
    await miss()
    await miss()
    expect(screen.getByText('Files are temporarily unavailable.')).toBeTruthy()
    expect(screen.getByText('file-19.txt')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('Reconnected')
    expect(screen.getByText('file-19.txt')).toBeTruthy()
    expect(screen.queryByText('file-20.txt')).toBeNull()
    expect(lists().map(call => call[2].cursor)).toEqual([undefined, 'cursor-after-20', 'cursor-after-20'])
    expect($hostedRoomCapabilities.get()['gateway-a'].limits.attachmentList).toBe(true)
  })

  it('the tighter 10s Files budget settles before the existing 30s gateway timeout', async () => {
    vi.useFakeTimers()
    const client = new HermesGateway()
    Reflect.set(client, 'socket', { readyState: WebSocket.OPEN, send: vi.fn() })
    mocks.requestProfile.mockImplementation(async (_route, method, params) =>
      method === 'groups.capabilities' ? capability : client.request(method, params)
    )
    render(
      <GroupAttachmentDownload
        attachment={parsedFilePage().items[0].attachment}
        group="Core"
        message={{ eventId: 'event-20', roomId: 'room-1' } as GroupMessage}
      />
    )
    const button = screen.getByRole('button', { name: 'Download file-20.txt' }) as HTMLButtonElement
    fireEvent.click(button)
    await act(async () => vi.advanceTimersByTimeAsync(9999))
    expect(button.disabled).toBe(true)
    await act(async () => vi.advanceTimersByTimeAsync(1))
    expect(button.disabled).toBe(false)
    expect(mocks.notify).toHaveBeenCalledWith({ kind: 'error', message: 'The download timed out.' })
    await act(async () => vi.advanceTimersByTimeAsync(20_000))
    expect(saved).toHaveLength(0)
  })
})

describe('Files v2 concrete UI paths', () => {
  it.each([
    { driver: false, persistent_process: true },
    { driver: true, persistent_process: false },
    { driver: false, persistent_process: false }
  ])('reads Files without admitting control on a readable gateway: %j', async flags => {
    const readable = { ...capability, ...flags }
    const original = mocks.requestProfile.getMockImplementation()!
    mocks.requestProfile.mockImplementation(async (...args) =>
      args[1] === 'groups.capabilities' ? readable : original(...args)
    )
    $hostedRoomCapabilities.set({
      'gateway-a': classifyHostedRoomCapability(readable, { connectionId: 'gateway-a' })
    })
    await open()
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(saved[0].href).toBe('data:text/plain;base64,YQ==')
    expect(await hostedRouteForRoom(FILE_ROOM)).toBeNull()
    expect(reads()).toHaveLength(1)
    expect(
      mocks.requestProfile.mock.calls.every(call =>
        ['groups.capabilities', 'groups.attachment.list', 'groups.attachment.read'].includes(call[1])
      )
    ).toBe(true)
  })

  it('an older host keeps Files discoverable with its own unavailable state and no list RPC', async () => {
    $hostedRoomCapabilities.set({
      'gateway-a': classifyHostedRoomCapability({ ...capability, features: [] }, { connectionId: 'gateway-a' })
    })
    await open()
    expect(screen.getByText("File browsing isn't available for this Group Chat yet.")).toBeTruthy()
    expect(screen.queryByText(translateBots('group.filesClassicDescription'))).toBeNull()
    expect(mocks.requestProfile).not.toHaveBeenCalled()
  })

  it('a confirmed list access denial clears rows and ignores a pending row download', async () => {
    const held = deferred<unknown>()
    const original = mocks.requestProfile.getMockImplementation()!
    let denied = false
    mocks.requestProfile.mockImplementation(async (...args) => {
      if (args[1] === 'groups.attachment.read') {
        return held.promise
      }

      if (args[1] === 'groups.attachment.list' && denied) {
        throw Object.assign(new Error('room quarantined'), { code: 4142 })
      }

      return original(...args)
    })
    await open()
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
    await waitFor(() => expect(reads()).toHaveLength(1))
    await miss()
    await miss()
    denied = true
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('Files are unavailable for this Group Chat.')
    expect(screen.queryByRole('listitem')).toBeNull()
    await act(async () => held.resolve({ attachment: fileItem(), content_base64: 'YQ==' }))
    expect(saved).toHaveLength(0)
  })

  it('classic files work offline, including search/row arrows/Enter and bidi isolation', async () => {
    const names = ['مرحبا\u202ereport.pdf', 'Résumé.pdf']

    const log = names.map((name, index): GroupMessage => ({
      id: `classic-${index}`,
      at: 1700000000000 + index,
      from: { kind: 'member', name: 'José' },
      text: '',
      images: [{ kind: 'file', name, data: 'data:text/plain;base64,YQ==' }]
    }))

    const room: GroupChat = { ...FILE_ROOM, hosted: null, continuityMode: 'desktop', log }
    $groupChats.set({ Core: room })
    const view = await open(room)
    await screen.findByText('Résumé.pdf')
    expect(screen.getByText(translateBots('group.filesClassicDescription'))).toBeTruthy()
    const input = screen.getByRole('textbox', { name: 'Search files' })
    expect(globalThis.document.activeElement).toBe(input)
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    const rows = screen.getAllByRole('listitem')
    expect(globalThis.document.activeElement).toBe(rows[0])
    fireEvent.keyDown(rows[0], { key: 'ArrowDown' })
    expect(globalThis.document.activeElement).toBe(rows[1])
    fireEvent.keyDown(rows[1], { key: 'Enter' })
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(saved[0].name).toBe(names[0])
    expect(screen.getByText(names[0]).tagName).toBe('BDI')
    fireEvent.keyDown(rows[1], { key: '/' })
    expect(globalThis.document.activeElement).toBe(input)
    fireEvent.change(input, { target: { value: 'jose' } })
    await waitFor(() => expect(screen.getByRole('list').getAttribute('aria-busy')).toBe('false'))
    expect(screen.getAllByRole('listitem')).toHaveLength(2)
    fireEvent.change(input, { target: { value: 'missing' } })
    await screen.findByText('No matching files.')
    const clearActions = screen.getAllByRole('button', { name: 'Clear search' })
    expect(clearActions).toHaveLength(2)
    fireEvent.click(clearActions.find(button => button.textContent === 'Clear search')!)
    await screen.findByText('Résumé.pdf')
    const textRoom = { ...room, log: [...log, { ...log[1], id: 'text-only', images: [], text: 'hello' }] }
    $groupChats.set({ Core: textRoom })
    view.rerender(<SharedFilesControl group="Core" room={textRoom} />)
    expect(screen.queryByRole('button', { name: 'Show latest' })).toBeNull()

    const withFile = {
      ...textRoom,
      log: [...textRoom.log, { ...log[1], id: 'new-file', images: [{ ...log[1].images![0], name: 'new.txt' }] }]
    }

    $groupChats.set({ Core: withFile })
    view.rerender(<SharedFilesControl group="Core" room={withFile} />)
    fireEvent.click(screen.getByRole('button', { name: 'Show latest' }))
    await screen.findByText('new.txt')
    expect(mocks.requestProfile).not.toHaveBeenCalled()
    expect(mocks.profileRoutes).not.toHaveBeenCalled()
  })

  it('verification refusal is distinct and retries the exact selection', async () => {
    const original = mocks.requestProfile.getMockImplementation()!
    let mismatch = true
    mocks.requestProfile.mockImplementation(async (...args) =>
      args[1] === 'groups.attachment.read'
        ? { attachment: { ...fileItem(), mime: mismatch ? 'text/html' : 'text/plain' }, content_base64: 'YQ==' }
        : original(...args)
    )
    await open()
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
    await screen.findByText("This file couldn't be verified. Nothing was downloaded.")
    expect(saved).toHaveLength(0)
    mismatch = false
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(saved).toHaveLength(1))
    expect(reads().map(call => call[2])).toEqual([reads()[0][2], reads()[0][2]])
  })

  it('an expired file offers Refresh list, not a blind download retry', async () => {
    const original = mocks.requestProfile.getMockImplementation()!
    let gone = false
    mocks.requestProfile.mockImplementation(async (...args) => {
      if (args[1] === 'groups.attachment.read') {
        gone = true
        throw Object.assign(new Error('attachment has expired'), { code: 4141 })
      }

      if (gone && args[1] === 'groups.attachment.list') {
        return filePage([])
      }

      return original(...args)
    })
    await open()
    await screen.findByText('file-20.txt')
    fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
    await screen.findByText('This file is no longer available.')
    expect((screen.getByRole('button', { name: 'Download file-20.txt' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Refresh list' }))
    await screen.findByText('No files shared yet.')
    expect(reads()).toHaveLength(1)
    expect(saved).toHaveLength(0)
  })
})
