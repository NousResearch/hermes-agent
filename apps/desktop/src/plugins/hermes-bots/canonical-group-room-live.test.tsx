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

  request.mockImplementation(async (_route, method) =>
    method === 'groups.state'
      ? { room: { name: 'Room', members }, driver_status: {} }
      : method === 'groups.log'
        ? { events }
        : {}
  )
  render(<CanonicalGroupWorkspace binding={binding} />)
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).disabled).toBe(false))
  fireEvent.click(await screen.findByRole('button', { name: 'Reply to Ops' }))
  await prepareCanonicalGroupSend(binding, { text: 'from the other window', attachments: [] })
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '@ops ship it' } })
  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(async () => expect(await readCanonicalGroupSend(binding)).toBeUndefined())
  await waitFor(() => expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('@ops ship it'))
  expect(screen.getByText('Replying to Ops')).toBeTruthy()

  fireEvent.click(screen.getByRole('button', { name: 'Send' }))
  await waitFor(() => expect(request.mock.calls.filter(c => c[1] === 'groups.send')).toHaveLength(2))
  expect(request.mock.calls.filter(c => c[1] === 'groups.send')[1][2].payload).toMatchObject({
    text: '@ops ship it',
    thread_id: 'thread-ops'
  })
})
