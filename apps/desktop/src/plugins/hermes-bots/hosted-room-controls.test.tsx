import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import type * as Client from './hosted-room-client'
import type * as Controls from './hosted-room-controls'
import { hostedPendingActions, hostedRoomKey } from './hosted-room-protocol'
import { translateBots } from './i18n-test-helper'
import type { Attachment } from './types'

const { host } = vi.hoisted(() => ({ host: {} as Record<string, unknown> }))
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const { useStore } = await import('@nanostores/react')

  return {
    ...await pluginSdkMock(host),
    useValue: useStore,
    usePluginI18n: () => translateBots,
    ConfirmDialog: (await import('../../components/ui/confirm-dialog')).ConfirmDialog,
    Button: ({ size: _size, variant: _variant, ...props }: ComponentProps<'button'> & { size?: string, variant?: string }) => <button type="button" {...props} />,
    RowButton: (props: ComponentProps<'button'>) => <button type="button" {...props} />,
    Codicon: () => null,
    GlyphSpinner: () => null,
    ErrorState: ({ description }: { description: string }) => <p>{description}</p>
  }
})

// DOM integration of the production controls and parser. RPC effects are explicit mocks,
// not live gateway/model evidence. Files here contain only synthetic test bytes.
const identity = { connectionId: 'mock-owner', authorityGatewayId: 'mock-authority', roomId: 'mock-room' }
const key = hostedRoomKey(identity)

const approval = {
  kind: 'approval', task_id: 'task-a', member_id: 'member-a', execution_generation: 3, request_id: 'request-a',
  approval: { command: 'printf "<review me>"', reason: 'Shell command', choices: ['once', 'deny', 'always'] }
}

const attachment: Attachment = {
  kind: 'file', name: 'test.txt', data: '', mime: 'text/plain', size: 4, attachmentId: `att_${'a'.repeat(32)}`
}

let client: typeof Client
let controls: typeof Controls

beforeEach(async () => {
  vi.resetModules()
  const { atom } = await import('nanostores')
  host.state = { gateway: atom('open') }
  client = await import('./hosted-room-client')
  controls = await import('./hosted-room-controls')
  client.$hostedRooms.set({ [key]: {
    identity, name: 'Mock room', cursor: 0, events: [], loading: false, busy: false,
    room: { room_id: identity.roomId, name: 'Mock room', authority_gateway_id: identity.authorityGatewayId,
      authority_epoch: 1, latest_seq: 0, members: [{ member_id: 'member-a', profile: 'helper', handle: 'helper', display_name: 'Helper' }] },
    capabilities: { authorityGatewayId: identity.authorityGatewayId, logLimit: 100, driver: true,
      persistentProcess: true, persistentHolds: true, attachments: true },
    driverStatus: { running: true, pending_actions: [approval, { kind: 'retry', task_id: 'task-b' }] }
  } })
  vi.spyOn(client, 'approveHostedTask').mockResolvedValue(undefined)
  vi.spyOn(client, 'retryHostedTask').mockResolvedValue(undefined)
  vi.spyOn(client, 'refreshHostedRoom').mockResolvedValue(undefined)
  vi.spyOn(client, 'fetchHostedAttachment').mockResolvedValue({ ...attachment, data: 'data:text/plain;base64,dGVzdA==' })
})

afterEach(() => { cleanup(); vi.useRealTimers(); vi.restoreAllMocks() })

it.each(['retry', 'confirmed-delivered'] as const)('confirms Telegram %s with frozen exact part and external-readback warning', async decision => {
  const delivery = { event_id: 'output', chunk_index: 1, attempt: 2, profile: 'helper', thread_id: 'thread', status: 'uncertain' as const }
  const resolve = vi.spyOn(client, 'resolveHostedTelegramDelivery').mockResolvedValue(undefined)
  client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key],
    capabilities: { ...client.$hostedRooms.get()[key].capabilities!, telegramRecovery: true },
    telegramStatus: { room_id: identity.roomId, chat_id: -999, blocked: true, attention: delivery }
  } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  expect(screen.getByText(/Telegram delivery blocked/)).toBeTruthy()
  expect(screen.getByText(/Bot API cannot read chat history/)).toBeTruthy()
  const mark = screen.getByRole('button', { name: 'Mark delivered' }) as HTMLButtonElement
  expect(mark.disabled).toBe(true)
  const field = screen.getByRole('textbox', { name: /Telegram message ID/ })
  fireEvent.change(field, { target: { value: '1.5' } })
  expect(mark.disabled).toBe(true)
  fireEvent.change(field, { target: { value: '777' } })
  expect(mark.disabled).toBe(false)
  const launch = decision === 'retry' ? 'Authorize delivery retry' : 'Mark delivered'
  fireEvent.click(screen.getByRole('button', { name: launch }))
  expect(resolve).not.toHaveBeenCalled()
  expect(screen.getByRole('dialog').textContent).toContain('Telegram chat -999, helper: output, part 2, attempt 2')
  expect(screen.getByRole('dialog').textContent).toMatch(decision === 'retry' ? /external readback.*duplicate/ : /message 777.*Bot API cannot verify/)
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(resolve).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: launch }))
  act(() => client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key], telegramStatus: {
    room_id: identity.roomId, chat_id: -999, blocked: true, attention: { ...delivery, attempt: 3 }
  } } }))
  fireEvent.click(screen.getByRole('button', { name: decision === 'retry' ? 'Retry this part' : 'Confirm delivered' }))
  await waitFor(() => expect(resolve).toHaveBeenCalledExactlyOnceWith(key, delivery, decision, decision === 'retry' ? undefined : 777))
  expect(client.retryHostedTask).not.toHaveBeenCalled()
})

it('keeps Telegram confirmation errors visible without claiming success', async () => {
  const delivery = { event_id: 'output', chunk_index: 1, attempt: 2, profile: 'helper', thread_id: 'thread', status: 'uncertain' as const }
  vi.spyOn(client, 'resolveHostedTelegramDelivery').mockRejectedValue(new Error('currently sending'))
  client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key],
    capabilities: { ...client.$hostedRooms.get()[key].capabilities!, telegramRecovery: true },
    telegramStatus: { room_id: identity.roomId, chat_id: -999, blocked: true, attention: delivery }
  } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  fireEvent.click(screen.getByRole('button', { name: 'Authorize delivery retry' }))
  fireEvent.click(screen.getByRole('button', { name: 'Retry this part' }))
  await screen.findByText('currently sending')
  expect(screen.getByRole('dialog')).toBeTruthy()
  expect(screen.queryByText('Done')).toBeNull()
})

it.each(['busy', 'sending', 'unsupported'] as const)('does not offer actionable recovery while %s', mode => {
  const resolve = vi.spyOn(client, 'resolveHostedTelegramDelivery').mockResolvedValue(undefined)
  client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key], busy: mode === 'busy',
    capabilities: { ...client.$hostedRooms.get()[key].capabilities!, telegramRecovery: mode !== 'unsupported' },
    telegramStatus: { room_id: identity.roomId, chat_id: -999, blocked: mode !== 'sending', attention: {
      event_id: 'output', chunk_index: 1, attempt: 2, profile: 'helper', thread_id: 'thread', status: mode === 'sending' ? 'sending' : 'uncertain'
    } }
  } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  const button = screen.queryByRole('button', { name: 'Authorize delivery retry' }) as HTMLButtonElement | null

  if (mode === 'busy') {
    expect(button?.disabled).toBe(true)
    fireEvent.click(button!)
  } else {
    expect(button).toBeNull()
  }

  expect(resolve).not.toHaveBeenCalled()
})

it('stops fetching on error and resumes polling after explicit successful refresh', async () => {
  vi.useFakeTimers()
  vi.mocked(client.refreshHostedRoom).mockImplementationOnce(async () => {
    client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key], error: 'mock outage' } })
  }).mockImplementation(async () => {
    client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key], error: undefined } })
  })
  const view = render(<controls.HostedRoomStatus roomKey={key} visible />)
  await act(async () => { await vi.advanceTimersByTimeAsync(6000) })
  expect(client.refreshHostedRoom).toHaveBeenCalledTimes(1)
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Refresh room' })) })
  expect(client.refreshHostedRoom).toHaveBeenCalledTimes(2)
  await act(async () => { await vi.advanceTimersByTimeAsync(4000) })
  expect(client.refreshHostedRoom).toHaveBeenCalledTimes(4)
  view.unmount()
  await act(async () => { await vi.advanceTimersByTimeAsync(4000) })
  expect(client.refreshHostedRoom).toHaveBeenCalledTimes(4)
})

it('requires confirmation to discard only the identified local input and warns backend work may still complete', async () => {
  const discard = vi.fn().mockResolvedValue(undefined)
  vi.spyOn(client, 'discardHostedInput').mockImplementation(discard)
  client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key],
    pending: { eventId: 'saved-id', text: 'saved text', threadId: 'saved-thread' } } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  fireEvent.click(screen.getByRole('button', { name: 'Discard saved input' }))
  expect(discard).not.toHaveBeenCalled()
  expect(screen.getByRole('dialog').textContent).toMatch(/does not cancel.*may still complete/i)
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(discard).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole('button', { name: 'Discard saved input' }))
  act(() => client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key],
    pending: { eventId: 'newer-id', text: 'newer text', threadId: 'newer-thread' } } }))
  fireEvent.click(screen.getByRole('button', { name: 'Discard local input' }))
  await waitFor(() => expect(discard).toHaveBeenCalledExactlyOnceWith(key, 'saved-id'))
})

it('shows the actual command as inert text and sends the exact request with once/deny only', () => {
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  const preview = screen.getByText('printf "<review me>"')
  expect(preview.tagName).toBe('PRE')
  expect(preview.querySelector('review')).toBeNull()
  expect(screen.getByText('Shell command')).toBeTruthy()
  expect(screen.queryByRole('button', { name: /always/i })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Allow once' }))
  expect(client.approveHostedTask).toHaveBeenCalledExactlyOnceWith(key, hostedPendingActions({ pending_actions: [approval] })[0], 'once')
  fireEvent.click(screen.getByRole('button', { name: 'Retry task' }))
  expect(client.retryHostedTask).toHaveBeenCalledExactlyOnceWith(key, 'task-b')
  expect(client.refreshHostedRoom).not.toHaveBeenCalled()
})

it('does not allow a command that has no inspectable preview', () => {
  const cache = client.$hostedRooms.get()[key]
  client.$hostedRooms.set({ [key]: { ...cache, driverStatus: {
    running: true, pending_actions: [{ ...approval, approval: { choices: ['once', 'deny'] } }]
  } } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  expect(screen.queryByRole('button', { name: 'Allow once' })).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Deny' }))
  expect(client.approveHostedTask).toHaveBeenCalledWith(key, expect.objectContaining({ requestId: 'request-a', executionGeneration: 3 }), 'deny')
})

it('disables task controls while another room operation owns admission', () => {
  client.$hostedRooms.set({ [key]: { ...client.$hostedRooms.get()[key], busy: true } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)

  for (const name of ['Allow once', 'Deny', 'Retry task']) {
    const button = screen.getByRole('button', { name }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
    fireEvent.click(button)
  }

  expect(client.approveHostedTask).not.toHaveBeenCalled()
  expect(client.retryHostedTask).not.toHaveBeenCalled()
})

it('reports unsupported pending actions and old-gateway attachment capability truthfully', () => {
  const cache = client.$hostedRooms.get()[key]
  client.$hostedRooms.set({ [key]: { ...cache, capabilities: { ...cache.capabilities!, attachments: false },
    driverStatus: { running: true, pending_actions: [{ kind: 'unknown', task_id: 'task-c' }] }
  } })
  render(<controls.HostedRoomStatus roomKey={key} visible={false} />)
  expect(screen.getByText(/1 pending actions cannot be safely identified/)).toBeTruthy()
  expect(screen.getByText(/Gateway-hosted · Text only/)).toBeTruthy()
})

it('fetches committed bytes only after a click and offers documents as downloads, not active frames', async () => {
  const view = render(<controls.HostedRoomAttachment attachment={attachment} eventId="event-a" roomKey={key} />)
  expect(client.fetchHostedAttachment).not.toHaveBeenCalled()
  expect(screen.queryByRole('link')).toBeNull()
  fireEvent.click(screen.getByRole('button', { name: 'Open attachment: test.txt' }))
  const link = await screen.findByRole('link', { name: 'Download test.txt' })
  expect(link.getAttribute('href')).toBe('data:text/plain;base64,dGVzdA==')
  expect(link.getAttribute('download')).toBe('test.txt')
  expect(client.fetchHostedAttachment).toHaveBeenCalledExactlyOnceWith(key, 'event-a', attachment)
  expect(view.container.querySelector('iframe,object,embed')).toBeNull()
})

it('shows fetch failures without dropping the retry gesture', async () => {
  vi.mocked(client.fetchHostedAttachment).mockRejectedValueOnce(new Error('receipt mismatch'))
  render(<controls.HostedRoomAttachment attachment={attachment} eventId="event-a" roomKey={key} />)
  fireEvent.click(screen.getByRole('button', { name: 'Open attachment: test.txt' }))
  expect((await screen.findByRole('alert')).textContent).toContain('receipt mismatch')
  fireEvent.click(screen.getByRole('button', { name: 'Open attachment: test.txt' }))
  await screen.findByRole('link', { name: 'Download test.txt' })
  expect(screen.queryByRole('alert')).toBeNull()
})

it.each([
  ['image', 'image/png', 'img'], ['file', 'audio/ogg', 'audio'], ['file', 'video/mp4', 'video']
] as const)('renders fetched %s %s bytes inline', async (kind, mime, tag) => {
  const media = { ...attachment, kind, mime }
  vi.mocked(client.fetchHostedAttachment).mockResolvedValue({ ...media, data: `data:${mime};base64,dGVzdA==` })
  const view = render(<controls.HostedRoomAttachment attachment={media} eventId="event-a" roomKey={key} />)
  fireEvent.click(screen.getByRole('button', { name: 'Open attachment: test.txt' }))
  await waitFor(() => expect(view.container.querySelector(tag)?.getAttribute('src')).toBe(`data:${mime};base64,dGVzdA==`))
})
