import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { GroupAttachmentDownload } from './group-attachment-download'
import { $groupChats } from './group-chat'
import { GroupChatWorkspace } from './group-chat-view'
import { deferred, FILE_ROOM, fileItem, parsedFilePage } from './group-files-test-fixtures'
import { SharedFilesDialog } from './group-files-view'
import { $hostedRoomCapabilities } from './hosted-room-capability-state'
import { readHostedGroupChatAttachment, stopHostedRoomRuntime } from './hosted-room-runtime'
import { translateBots } from './i18n-test-helper'
import type { GroupMessage } from './types'

const mocks = vi.hoisted(() => ({ notify: vi.fn(), requestProfile: vi.fn(), profileRoutes: vi.fn() }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return { ...sdk, host: { ...sdk.host, ...mocks }, usePluginI18n: () => translateBots }
})

vi.mock('./group-chat-parts', () => ({
  GroupClarifyCard: () => null,
  GroupImageControls: () => null,
  GroupMentionInput: () => null
}))

vi.mock('./group-rounds', () => ({ sendToGroupChatDurably: vi.fn(), stopGroupThread: vi.fn() }))

const ROUTE = { connectionId: 'gateway-a', mode: 'remote', profile: 'default', targetProfile: 'default' }
const item = fileItem(20, 'report.txt')
const attachment = parsedFilePage([item]).items[0].attachment

const message: GroupMessage = {
  at: 1_700_000_000_000,
  eventId: item.event_id,
  from: { kind: 'user', name: 'You' },
  id: item.event_id,
  images: [attachment],
  roomId: 'room-1',
  text: ''
}

const receipt = (override: Record<string, unknown> = {}) => ({
  attachment: {
    attachment_id: item.attachment_id,
    kind: 'file',
    mime: 'text/plain',
    name: 'report.txt',
    size: 1,
    ...override
  },
  content_base64: 'YQ=='
})

const reads = () => mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.attachment.read')
const downloads: Array<{ href: string; name: string }> = []

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

beforeEach(() => {
  vi.clearAllMocks()
  downloads.length = 0
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.href, name: this.download })
  })
  $groupChats.set({ Core: { ...FILE_ROOM, log: [message] } })
  $hostedRoomCapabilities.set({})
  mocks.profileRoutes.mockResolvedValue([ROUTE])
  mocks.requestProfile.mockImplementation(async (_route, method) => {
    if (method === 'groups.capabilities') {
      return {
        authority_gateway_id: 'install:home',
        driver: true,
        persistent_process: true,
        methods: ['groups.attachment.read', 'groups.attachment.put']
      }
    }

    if (method === 'groups.attachment.read') {
      return receipt()
    }

    throw new Error(`Unexpected RPC: ${method}`)
  })
})

afterEach(() => {
  cleanup()
  stopHostedRoomRuntime()
  vi.restoreAllMocks()
})

describe('the shared verified download action', () => {
  it('reads exact room/event/attachment IDs before creating the download', async () => {
    render(<GroupAttachmentDownload attachment={attachment} group="Core" message={message} presentation="icon" />)
    fireEvent.click(screen.getByRole('button', { name: 'Download report.txt' }))
    await waitFor(() => expect(downloads).toHaveLength(1))
    expect(reads()).toEqual([
      [
        ROUTE,
        'groups.attachment.read',
        {
          attachment_id: item.attachment_id,
          event_id: item.event_id,
          purpose: 'viewer',
          room_id: 'room-1'
        }
      ]
    ])
    expect(downloads).toEqual([{ href: 'data:text/plain;base64,YQ==', name: 'report.txt' }])
  })

  it('selects the exact same-name version from a Files row, not a filename lookup', async () => {
    const older = fileItem(19, 'report.txt')
    const loadPage = vi.fn(async () => parsedFilePage([item, older]))
    mocks.requestProfile.mockImplementation(async (_route, method, params) => {
      if (method === 'groups.capabilities') {
        return {
          authority_gateway_id: 'install:home',
          driver: true,
          persistent_process: true,
          features: ['attachment_metadata_catalog']
        }
      }

      expect(params.attachment_id).toBe(older.attachment_id)
      expect(params.event_id).toBe(older.event_id)

      return receipt({ attachment_id: older.attachment_id })
    })
    render(
      <SharedFilesDialog
        availability="available"
        group="Core"
        latestSeq={20}
        loadPage={loadPage}
        onClose={vi.fn()}
        open
        roomId="room-1"
      />
    )
    await waitFor(() => expect(screen.getAllByRole('button', { name: 'Download report.txt' })).toHaveLength(2))
    expect(reads()).toHaveLength(0)
    fireEvent.click(screen.getAllByRole('button', { name: 'Download report.txt' })[1])
    await waitFor(() => expect(downloads).toHaveLength(1))
    expect(reads()[0][2]).toMatchObject({ event_id: older.event_id, attachment_id: older.attachment_id })
  })

  it.each([
    { attachment_id: fileItem(19).attachment_id },
    { name: 'different.txt' },
    { mime: 'text/html' },
    { size: 2 },
    { name: undefined },
    { mime: undefined },
    { size: undefined },
    { size: '1' }
  ])('refuses a mismatched or incomplete server receipt %j', async override => {
    mocks.requestProfile.mockImplementation(async (_route, method) =>
      method === 'groups.capabilities'
        ? { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
        : receipt(override)
    )
    render(<GroupAttachmentDownload attachment={attachment} group="Core" message={message} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download report.txt' }))
    await waitFor(() =>
      expect(mocks.notify).toHaveBeenCalledWith({
        kind: 'error',
        message: "This file couldn't be verified. Nothing was downloaded."
      })
    )
    expect(downloads).toHaveLength(0)
    expect((screen.getByRole('button') as HTMLButtonElement).disabled).toBe(false)
  })

  it('suppresses duplicate clicks and permits an explicit retry after failure', async () => {
    const held = deferred<ReturnType<typeof receipt>>()
    mocks.requestProfile.mockImplementation(async (_route, method) =>
      method === 'groups.capabilities'
        ? { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
        : held.promise
    )
    render(<GroupAttachmentDownload attachment={attachment} group="Core" message={message} />)
    const button = screen.getByRole('button', { name: 'Download report.txt' })
    fireEvent.click(button)
    fireEvent.click(button)
    await waitFor(() => expect(reads()).toHaveLength(1))
    expect((button as HTMLButtonElement).disabled).toBe(true)
    await act(async () => held.reject(new Error('offline')))
    expect((button as HTMLButtonElement).disabled).toBe(false)
    mocks.requestProfile.mockImplementation(async (_route, method) =>
      method === 'groups.capabilities'
        ? { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
        : receipt()
    )
    fireEvent.click(button)
    await waitFor(() => expect(downloads).toHaveLength(1))
    expect(reads()).toHaveLength(2)
  })

  it('does not download or notify after the row unmounts during a read', async () => {
    const held = deferred<ReturnType<typeof receipt>>()
    mocks.requestProfile.mockImplementation(async (_route, method) =>
      method === 'groups.capabilities'
        ? { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
        : held.promise
    )
    const view = render(<GroupAttachmentDownload attachment={attachment} group="Core" message={message} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download report.txt' }))
    await waitFor(() => expect(reads()).toHaveLength(1))
    view.unmount()
    await act(async () => held.resolve(receipt()))
    expect(downloads).toHaveLength(0)
    expect(mocks.notify).not.toHaveBeenCalled()
  })

  it('refuses a stale row after the room name has been reused for another room', async () => {
    $groupChats.set({ Core: { ...FILE_ROOM, roomId: 'room-replacement' } })
    await expect(readHostedGroupChatAttachment('Core', message, attachment)).rejects.toThrow('unavailable')
    expect(reads()).toHaveLength(0)
  })

  it('refuses a room replacement while its authority route is being resolved', async () => {
    const routes = deferred<(typeof ROUTE)[]>()
    mocks.profileRoutes.mockReturnValue(routes.promise)
    const pending = readHostedGroupChatAttachment('Core', message, attachment)
    $groupChats.set({ Core: { ...FILE_ROOM, roomId: 'replacement' } })
    routes.resolve([ROUTE])
    await expect(pending).rejects.toThrow('unavailable')
    expect(reads()).toHaveLength(0)
  })

  it('refuses bytes if authority scope changes after the read starts', async () => {
    const held = deferred<ReturnType<typeof receipt>>()
    mocks.requestProfile.mockImplementation(async (_route, method) =>
      method === 'groups.capabilities'
        ? { authority_gateway_id: 'install:home', driver: true, persistent_process: true }
        : held.promise
    )
    render(<GroupAttachmentDownload attachment={attachment} group="Core" message={message} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download report.txt' }))
    await waitFor(() => expect(reads()).toHaveLength(1))
    $groupChats.set({ Core: { ...FILE_ROOM, hostedEpoch: 2 } })
    await act(async () => held.resolve(receipt()))
    expect(downloads).toHaveLength(0)
    expect(mocks.notify).toHaveBeenCalledTimes(1)
  })
})

describe('existing transcript attachment regression', () => {
  it('downloads a hosted transcript chip even when its gateway lacks the catalog', async () => {
    render(<GroupChatWorkspace group="Core" members={[]} />)
    expect(screen.getByRole('button', { name: 'Files' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Download report.txt' }))
    await waitFor(() => expect(downloads).toHaveLength(1))
    expect(reads()[0][2]).toMatchObject({ event_id: item.event_id, attachment_id: item.attachment_id })
  })

  it('keeps classic data-URL chip downloads local without inventing a catalog/read route', async () => {
    $groupChats.set({
      Core: {
        ...FILE_ROOM,
        hosted: null,
        continuityMode: 'desktop',
        log: [{ ...message, images: [{ kind: 'file', name: 'local.txt', data: 'data:text/plain;base64,Yg==' }] }]
      }
    })
    render(<GroupChatWorkspace group="Core" members={[]} />)
    fireEvent.click(screen.getByRole('button', { name: 'Download local.txt' }))
    await waitFor(() => expect(downloads).toEqual([{ href: 'data:text/plain;base64,Yg==', name: 'local.txt' }]))
    expect(mocks.requestProfile).not.toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Files' })).toBeTruthy()
  })
})
