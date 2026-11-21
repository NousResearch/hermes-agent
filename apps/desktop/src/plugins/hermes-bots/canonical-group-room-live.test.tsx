import { useStore } from '@nanostores/react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const request = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock, createGroupGateway } = await import('./group-test-utils')
  const gateway = createGroupGateway()
  const { en } = await import('@/i18n/en')
  const { CANONICAL_GROUP_LOCALES } = await import('./canonical-group-locales')
  const { ConfirmDialog } = await import('@/components/ui/confirm-dialog')

  return {
    ...(await pluginSdkMock(gateway.host)),
    atom,
    useValue: useStore,
    ConfirmDialog,
    useI18n: () => ({ t: en }),
    usePluginI18n: () => (key: string) =>
      CANONICAL_GROUP_LOCALES.en[key.replace('canonical.', '') as keyof typeof CANONICAL_GROUP_LOCALES.en] ?? key,
    Button: (p: ComponentProps<'button'>) => <button {...p} />,
    Input: (p: ComponentProps<'input'>) => <input {...p} />,
    RowButton: (p: ComponentProps<'button'>) => <button type="button" {...p} />,
    host: { ...gateway.host, requestProfile: request }
  }
})

import { $canonicalRoomLive, canonicalRoomLiveKey } from './canonical-group-details'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { prepareCanonicalGroupSend, readCanonicalGroupSend } from './canonical-group-send'
import { CanonicalGroupWorkspace } from './canonical-group-workspace'

const originalDesktop = window.hermesDesktop
beforeEach(() => {
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, writable: true, value: undefined })
})
afterEach(() => {
  cleanup()
  request.mockReset()
  localStorage.clear()
  window.hermesDesktop = originalDesktop
  vi.unstubAllGlobals()
})

const members = [{ member_id: 'ops', profile: 'ops', handle: 'ops', display_name: 'Ops' }]

it('starts a new thread after a refused reply is cancelled', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'cancel-refused-reply' }

  const events = [
    {
      seq: 1,
      room_id: binding.roomId,
      event_id: 'm1',
      kind: 'message.member',
      actor: { kind: 'member', id: 'ops' },
      payload: { text: 'Deploy is green.', thread_id: 'thread-ops' }
    }
  ]

  let refuse = true
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {
      return { room: { name: 'Room', members }, driver_status: {} }
    }

    if (method === 'groups.log') {
      return { events }
    }

    if (method === 'groups.send') {
      if (refuse) {
        throw Object.assign(new Error('not admitted'), { data: { admitted: false } })
      }

      return { accepted: true, client_event_id: params.event_id }
    }

    return {}
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.click(await screen.findByRole('button', { name: 'Reply to Ops' }))
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '@ops ship it' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(CANONICAL_GROUP_LOCALES.en.sendRefused)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: 'Cancel reply' }))
  expect(screen.queryByText('Replying to Ops')).toBeNull()
  refuse = false
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(''))
  const sends = request.mock.calls.filter(call => call[1] === 'groups.send')
  expect(sends).toHaveLength(2)
  expect(sends[0][2].payload.thread_id).toBe('thread-ops')
  expect(sends[1][2].payload).toMatchObject({ text: '@ops ship it', thread_id: expect.any(String) })
  expect(sends[1][2].payload.thread_id).not.toBe('thread-ops')
  expect(sends[1][2].event_id).not.toBe(sends[0][2].event_id)
})

it('puts the caret in the composer, after the seeded mention, when Reply is pressed', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'reply-focus' }

  const events = [
    {
      seq: 1,
      room_id: binding.roomId,
      event_id: 'm1',
      kind: 'message.member',
      actor: { kind: 'member', id: 'ops' },
      payload: { text: 'Deploy is green.', thread_id: 'thread-ops' }
    }
  ]

  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room', members }, driver_status: {} }
      : method === 'groups.log'
        ? { events }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  const input = screen.getByRole('textbox') as HTMLTextAreaElement
  await waitFor(() => expect(input.disabled).toBe(false))
  fireEvent.click(await screen.findByRole('button', { name: 'Reply to Ops' }))
  await waitFor(() => expect(window.document.activeElement).toBe(input))
  expect(input.value).toBe('@ops ')
  expect(input.selectionStart).toBe(input.value.length)
})

it('drops a closed room’s Bots-list state, which nothing polls any more', async () => {
  const binding = { connectionId: 'local', profile: 'team', roomId: 'closing' }
  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? {
          room: { name: 'Room', members },
          driver_status: { working: true, member_activity: [{ member_id: 'ops', status: 'running' }] }
        }
      : method === 'groups.log'
        ? { events: [] }
        : {}
  )
  const view = render(<CanonicalGroupWorkspace binding={binding} />)
  const key = canonicalRoomLiveKey(binding, binding.roomId)
  await waitFor(() => expect($canonicalRoomLive.get()[key]?.working).toBe(true))

  view.unmount()
  expect($canonicalRoomLive.get()[key]).toBeUndefined()
})

it('keeps the thread this window was replying in while it sends another window’s pending intent', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'held-reply' }

  const events = [
    {
      seq: 1,
      room_id: 'held-reply',
      event_id: 'm1',
      kind: 'message.member',
      actor: { kind: 'member', id: 'ops' },
      payload: { text: 'Deploy is green.', thread_id: 'thread-ops' }
    }
  ]

  request.mockImplementation(async (_route, method, params) =>
    method === 'groups.state'
      ? { room: { name: 'Room', members }, driver_status: {} }
      : method === 'groups.log'
        ? { events }
        : method === 'groups.send'
          ? { accepted: true, client_event_id: params.event_id }
          : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.click(await screen.findByRole('button', { name: 'Reply to Ops' }))
  const other = await prepareCanonicalGroupSend(binding, { text: 'from the other window', attachments: [] })
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '@ops ship it' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('@ops ship it'))
  expect(screen.getByText('Replying to Ops')).toBeTruthy()
  expect(request.mock.calls.filter(call => call[1] === 'groups.send').map(call => call[2])).toEqual([
    { ...other.params, profile: 'team' }
  ])

  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.filter(c => c[1] === 'groups.send')).toHaveLength(2))
  expect(request.mock.calls.filter(c => c[1] === 'groups.send')[1][2].payload).toMatchObject({
    text: '@ops ship it',
    thread_id: 'thread-ops'
  })
})

it('keeps a refused other-window reply in its thread, then hands this window its reply back', async () => {
  const binding = { connectionId: 'remote', profile: 'team', roomId: 'refused-reply' }

  const events = [
    {
      seq: 1,
      room_id: binding.roomId,
      event_id: 'm1',
      kind: 'message.member',
      actor: { kind: 'member', id: 'ops' },
      payload: { text: 'Deploy is green.', thread_id: 'thread-ops' }
    }
  ]

  let refuse = true
  request.mockImplementation(async (_route, method, params) => {
    if (method === 'groups.state') {
      return { room: { name: 'Room', members }, driver_status: {} }
    }

    if (method === 'groups.log') {
      return { events }
    }

    if (method === 'groups.send') {
      if (refuse) {
        throw Object.assign(new Error('not admitted'), { data: { admitted: false } })
      }

      return { accepted: true, client_event_id: params.event_id }
    }

    return {}
  })
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.click(await screen.findByRole('button', { name: 'Reply to Ops' }))
  await prepareCanonicalGroupSend(binding, { text: 'other reply', attachments: [], thread_id: 'other-thread' })
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '@ops ship it' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText(CANONICAL_GROUP_LOCALES.en.sendRefused)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  expect(screen.queryByText('Replying to Ops')).toBeNull()
  refuse = false
  fireEvent.change(screen.getByRole('textbox'), { target: { value: 'edited other reply' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await screen.findByText('Replying to Ops')
  expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('@ops ship it')
  const sends = request.mock.calls.filter(call => call[1] === 'groups.send')
  expect(sends[1][2].payload).toMatchObject({ text: 'edited other reply', thread_id: 'other-thread' })
  expect(sends[1][2].event_id).not.toBe(sends[0][2].event_id)
})
