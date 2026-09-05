/** Access/cancellation contracts through real Files components, route and receipt readers. */
import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { downloadGroupChatAttachment } from '../plugins/hermes-bots/group-attachment-download'
import { $groupChats } from '../plugins/hermes-bots/group-chat'
import { groupFileFailure } from '../plugins/hermes-bots/group-file-errors'
import { listHostedGroupFiles } from '../plugins/hermes-bots/group-files-client'
import { deferred, FILE_ROOM, fileItem, filePage } from '../plugins/hermes-bots/group-files-test-fixtures'
import { SharedFilesControl } from '../plugins/hermes-bots/group-files-view'
import { $hostedRoomCapabilities } from '../plugins/hermes-bots/hosted-room-capability-state'
import { classifyHostedRoomCapability } from '../plugins/hermes-bots/hosted-room-client'
import { stopHostedRoomRuntime } from '../plugins/hermes-bots/hosted-room-runtime'
import { translateBots } from '../plugins/hermes-bots/i18n-test-helper'
import type { GroupMessage } from '../plugins/hermes-bots/types'

const mocks = vi.hoisted(() => ({ requestProfile: vi.fn(), profileRoutes: vi.fn(), notify: vi.fn() }))
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

const receipt = (seq = 19) => ({ attachment: fileItem(seq), content_base64: 'YQ==' })
const error = (code: number, message: string) => Object.assign(new Error(message), { code })
const reads = () => mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.attachment.read')
const lists = () => mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.attachment.list')
const saved: string[] = []

beforeEach(() => {
  vi.clearAllMocks()
  saved.length = 0
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    saved.push(this.download)
  })
  $groupChats.set({ Core: FILE_ROOM })
  $hostedRoomCapabilities.set({ 'gateway-a': classifyHostedRoomCapability(capability, { connectionId: 'gateway-a' }) })
  mocks.profileRoutes.mockResolvedValue([route])
  mocks.requestProfile.mockImplementation((_route, method) =>
    Promise.resolve(
      method === 'groups.capabilities'
        ? capability
        : method === 'groups.attachment.list'
          ? filePage([fileItem(20), fileItem(19)])
          : receipt()
    )
  )
})
afterEach(() => {
  cleanup()
  stopHostedRoomRuntime()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

async function open() {
  render(<SharedFilesControl group="Core" room={FILE_ROOM} />)
  fireEvent.click(screen.getByRole('button', { name: 'Files' }))
  await screen.findByText('file-19.txt')
}

async function offline() {
  for (let index = 0; index < 2; index += 1) {
    await act(async () =>
      $hostedRoomCapabilities.set({
        'gateway-a': classifyHostedRoomCapability(
          { ok: false, error: new Error('offline') },
          { connectionId: 'gateway-a' }
        )
      })
    )
  }
}

describe('observed room denial fences sibling delivery before React unmount', () => {
  it.each([
    'This Group Chat is managed by another gateway.',
    'hosted room not found',
    'Group Chat history expired; room_id remains permanently retired',
    'room quarantined'
  ])('A1/B3 list denial %s wins over a sibling success in the same tick', async message => {
    const pendingRead = deferred<unknown>()
    const pendingList = deferred<unknown>()
    const original = mocks.requestProfile.getMockImplementation()!
    let retry = false
    mocks.requestProfile.mockImplementation((...args) =>
      args[1] === 'groups.attachment.read'
        ? pendingRead.promise
        : args[1] === 'groups.attachment.list' && retry
          ? pendingList.promise
          : original(...args)
    )
    await open()
    fireEvent.click(screen.getByRole('button', { name: 'Download file-19.txt' }))
    await waitFor(() => expect(reads()).toHaveLength(1))
    await offline()
    retry = true
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(lists()).toHaveLength(2))
    await act(async () => {
      pendingList.reject(error(4142, message))
      pendingRead.resolve(receipt())
    })
    expect(saved).toHaveLength(0)
    expect(screen.queryAllByRole('listitem')).toHaveLength(0)
    expect(screen.getByText('Files are unavailable for this Group Chat.')).toBeTruthy()
  })

  it.each([true, false])(
    'A2 read denial cancels sibling success without a render gap, denial first=%s',
    async denialFirst => {
      const denied = deferred<unknown>()
      const sibling = deferred<unknown>()
      const original = mocks.requestProfile.getMockImplementation()!
      mocks.requestProfile.mockImplementation((...args) =>
        args[1] === 'groups.attachment.read'
          ? args[2].attachment_id === fileItem(20).attachment_id
            ? denied.promise
            : sibling.promise
          : original(...args)
      )
      await open()
      fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
      await waitFor(() => expect(reads()).toHaveLength(1))
      fireEvent.click(screen.getByRole('button', { name: 'Download file-19.txt' }))
      await waitFor(() => expect(reads()).toHaveLength(2))
      await act(async () => {
        if (denialFirst) {
          denied.reject(error(4141, 'room quarantined'))
        }

        sibling.resolve(receipt())

        if (!denialFirst) {
          denied.reject(error(4141, 'room quarantined'))
        }
      })
      expect(saved).toHaveLength(0)
      expect(screen.queryAllByRole('listitem')).toHaveLength(0)
    }
  )

  it('requires a fresh successful catalog before new delivery and never revives an old lease', async () => {
    const sibling = deferred<unknown>()
    const original = mocks.requestProfile.getMockImplementation()!
    let deny = true
    let catalogFailure: 'offline' | 'malformed' | null = null
    mocks.requestProfile.mockImplementation((...args) =>
      args[1] === 'groups.attachment.read'
        ? args[2].attachment_id === fileItem(19).attachment_id
          ? sibling.promise
          : deny
            ? Promise.reject(error(4141, 'room quarantined'))
            : Promise.resolve(receipt(20))
        : args[1] === 'groups.attachment.list' && catalogFailure
          ? catalogFailure === 'offline'
            ? Promise.reject(new Error('connection lost'))
            : Promise.resolve({ ...filePage(), items: 'invalid' })
          : original(...args)
    )
    await open()
    fireEvent.click(screen.getByRole('button', { name: 'Download file-19.txt' }))
    await waitFor(() => expect(reads()).toHaveLength(1))
    fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
    await waitFor(() => expect(screen.queryAllByRole('listitem')).toHaveLength(0))
    const selected = fileItem(20)

    const attemptDownload = () =>
      downloadGroupChatAttachment('Core', { eventId: selected.event_id, roomId: FILE_ROOM.roomId } as GroupMessage, {
        attachmentId: selected.attachment_id,
        kind: 'file',
        name: selected.name,
        mime: selected.mime,
        size: selected.size
      })

    await expect(attemptDownload()).rejects.toThrow()
    expect(reads()).toHaveLength(2)

    for (const failure of ['offline', 'malformed'] as const) {
      catalogFailure = failure
      await act(async () => {
        await expect(listHostedGroupFiles('Core')).rejects.toThrow()
      })
      await expect(attemptDownload()).rejects.toThrow()
      expect(reads()).toHaveLength(2)
    }

    catalogFailure = null
    deny = false
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await screen.findByText('file-20.txt')
    expect(lists()).toHaveLength(4)
    fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
    await waitFor(() => expect(saved).toEqual(['file-20.txt']))
    await act(async () => sibling.resolve(receipt()))
    expect(saved).toEqual(['file-20.txt'])
  })

  it.each(['attachment has expired', 'attachment is unavailable to Group Chat viewers', 'verification'])(
    'row-local %s does not revoke sibling delivery',
    async message => {
      const first = deferred<unknown>()
      const second = deferred<unknown>()
      const original = mocks.requestProfile.getMockImplementation()!
      mocks.requestProfile.mockImplementation((...args) =>
        args[1] === 'groups.attachment.read'
          ? args[2].attachment_id === fileItem(20).attachment_id
            ? first.promise
            : second.promise
          : original(...args)
      )
      await open()
      fireEvent.click(screen.getByRole('button', { name: 'Download file-20.txt' }))
      await waitFor(() => expect(reads()).toHaveLength(1))
      fireEvent.click(screen.getByRole('button', { name: 'Download file-19.txt' }))
      await waitFor(() => expect(reads()).toHaveLength(2))
      await act(async () => {
        if (message === 'verification') {
          first.resolve({ ...receipt(20), attachment: { ...fileItem(20), mime: 'text/html' } })
        } else {
          first.reject(error(4141, message))
        }

        second.resolve(receipt())
      })
      expect(saved).toEqual(['file-19.txt'])
      expect(screen.getAllByRole('listitem')).toHaveLength(2)
    }
  )
})

describe('closing intent reaches the route-to-RPC transition', () => {
  it('does not dispatch a timed-out read after its dialog closes', async () => {
    await open()
    const routes = deferred<(typeof route)[]>()
    mocks.profileRoutes.mockClear().mockReturnValue(routes.promise)
    vi.useFakeTimers()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Download file-19.txt' })))
    expect(mocks.profileRoutes).toHaveBeenCalledTimes(1)
    await act(async () => vi.advanceTimersByTimeAsync(10_001))
    expect(screen.getByRole('alert').textContent).toContain('The download timed out.')
    await act(async () => {
      fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
      routes.resolve([route])
    })
    expect(reads()).toHaveLength(0)
    expect(saved).toHaveLength(0)
  })

  it.each(['list', 'read'].flatMap(kind => ['routes', 'capability'].map(phase => ({ kind, phase }))))(
    'B2 does not dispatch a new $kind after closing during $phase discovery',
    async ({ kind, phase }) => {
      const routes = deferred<(typeof route)[]>()
      const probe = deferred<unknown>()

      if (kind === 'read') {
        await open()
      }

      mocks.profileRoutes.mockClear()
      const original = mocks.requestProfile.getMockImplementation()!
      const initialProbeCount = mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.capabilities').length

      if (phase === 'routes') {
        mocks.profileRoutes.mockReturnValue(routes.promise)
      } else {
        mocks.requestProfile.mockImplementation((...args) =>
          args[1] === 'groups.capabilities' ? probe.promise : original(...args)
        )
      }

      if (kind === 'list') {
        render(<SharedFilesControl group="Core" room={FILE_ROOM} />)
        fireEvent.click(screen.getByRole('button', { name: 'Files' }))
      } else {
        fireEvent.click(screen.getByRole('button', { name: 'Download file-19.txt' }))
      }

      await waitFor(() => expect(mocks.profileRoutes).toHaveBeenCalledTimes(1))

      if (phase === 'capability') {
        await waitFor(() =>
          expect(mocks.requestProfile.mock.calls.filter(call => call[1] === 'groups.capabilities')).toHaveLength(
            initialProbeCount + 1
          )
        )
      }

      const previousCapabilities = $hostedRoomCapabilities.get()
      await act(async () => {
        fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
        routes.resolve([route])
        probe.resolve(capability)
      })
      expect($hostedRoomCapabilities.get()).toBe(previousCapabilities)
      expect(lists()).toHaveLength(kind === 'list' ? 0 : 1)
      expect(reads()).toHaveLength(0)
      expect(saved).toHaveLength(0)
    }
  )
})

describe('room denial classification is not a blanket RPC-code rule', () => {
  it.each([
    'hosted room not found',
    'This Group Chat is managed by another gateway.',
    'Group Chat history expired; room_id remains permanently retired',
    'hosted room is being disbanded',
    'attachment catalogue is unavailable for this room authority',
    'Group Chat is unavailable to viewers',
    'Group Chat viewer authority changed',
    'This Group Chat has an unverified authority takeover and is read-only until its history is reconciled (test).'
  ])('recognizes declared room denial %s', message => {
    expect(groupFileFailure(error(4142, message))).toBe('access')
  })
  it.each([
    'authority_epoch must be a positive integer',
    'attachment catalogue room metadata is invalid',
    'query must be a string',
    'database is locked'
  ])('keeps non-revocation 4142 %s generic', message => {
    expect(groupFileFailure(error(4142, message))).toBe('error')
  })
})
