/**
 * A retry after a lost `groups.create` reply is the same request: display data that changed in
 * between (a Bot's title, a mirrored room taking the generated name) never mints a second room.
 */
import { host } from '@hermes/plugin-sdk'
import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import {
  answerGroupRequests,
  PEER_HOME as HOME,
  PEER_MINI as MINI,
  PEER_REGISTERED,
  peerGroupCapabilities,
  peerInvitation,
  peerRosterRow,
  strictGroupCreation
} from './canonical-group-contract-test-utils'
import { CANONICAL_GROUP_LOCALES } from './canonical-group-locales'
import { $canonicalGroupBindings } from './canonical-group-registry'
import { CreateGroupChatDialog } from './create-dialog'
import { $botMeta } from './data'
import { $groupChats, updateGroupChat } from './group-chat'
import { translateBots } from './i18n-test-helper'
import { setPluginCtx } from './shared'
import type { RosterRow } from './types'

const { request, notify } = vi.hoisted(() => ({ request: vi.fn(), notify: vi.fn() }))
vi.mock('@hermes/plugin-sdk', async importOriginal =>
  (await import('./canonical-group-contract-test-utils')).localGatewaySdkMock(
    await importOriginal<typeof HermesSdk>(),
    request,
    () => translateBots,
    { notify }
  )
)

const words = CANONICAL_GROUP_LOCALES.en
const here: RosterRow = { name: 'default', connectionId: 'local' }
const beta: RosterRow = { name: 'beta', connectionId: 'local' }
const mini = peerRosterRow('mini', 'Mac Mini')

beforeEach(() => {
  ;(host.state.gateway as ReturnType<typeof HermesSdk.atom<string>>).set('open')
  $canonicalGroupBindings.set({})
  $botMeta.set({})
  $groupChats.set({})
  request.mockReset()
  notify.mockReset()
  Element.prototype.scrollIntoView = vi.fn()
})
afterEach(() => {
  cleanup()
  setPluginCtx(null)
  localStorage.clear()
})

/** A home whose first `groups.create` reply is lost after the room was made. */
function loseFirstReply() {
  const { rooms, create } = strictGroupCreation()
  let lost = true
  answerGroupRequests(request, {
    local: (method, params) =>
      method === 'groups.create'
        ? (() => {
            const answer = create(params)

            if (lost) {
              lost = false
              throw new Error('reply lost')
            }

            return answer
          })()
        : ({ 'groups.capabilities': HOME, 'groups.peer.register': PEER_REGISTERED } as Record<string, unknown>)[method],
    mini: method =>
      (({ 'groups.capabilities': MINI, 'groups.peer.invite': peerInvitation() }) as Record<string, unknown>)[method]
  })

  return rooms
}

const box = (face: Element | null | undefined) =>
  face?.closest('label')?.querySelector('button[role="checkbox"]') as HTMLButtonElement
const submit = () =>
  act(async () => {
    fireEvent.click(screen.getByRole('button', { name: /^Create group/ }))
  })
const creates = () => request.mock.calls.filter(call => call[1] === 'groups.create').map(call => call[2])

async function draft(roster: RosterRow[], faces: (base: HTMLElement) => Element[], name = 'Team') {
  const onCreated = vi.fn()
  const view = render(<CreateGroupChatDialog onClose={vi.fn()} onCreated={onCreated} open roster={roster} />)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.capabilities')).toBe(true))
  await act(async () => {})
  faces(view.baseElement).forEach(face => fireEvent.click(box(face)))

  if (name) {
    fireEvent.change(screen.getByRole('textbox', { name: 'Group name' }), { target: { value: name } })
  }
  await waitFor(() =>
    expect((screen.getByRole('button', { name: /^Create group/ }) as HTMLButtonElement).disabled).toBe(false)
  )
  await submit()
  expect(screen.getByText(words.createUncertain)).toBeTruthy()

  return { ...view, onCreated }
}

it("a retry after a lost reply reuses the room when a peer Bot's title changed in between", async () => {
  const rooms = loseFirstReply()
  const roster = [here, mini]
  const { onCreated, rerender } = await draft(roster, base => [...base.querySelectorAll('[data-bot-face="default"]')])
  // The roster refreshes: the Mac Mini's Bot has a title now, which changes its room label.
  rerender(
    <CreateGroupChatDialog
      onClose={vi.fn()}
      onCreated={onCreated}
      open
      roster={[here, { ...mini, title: 'Reviewer' }]}
    />
  )
  await act(async () => {})
  await submit()
  await waitFor(() => expect(onCreated).toHaveBeenCalledOnce())
  expect(creates()[1]).toEqual(creates()[0])
  expect(rooms.size).toBe(1)
})

// A blank name is filled from the Bots' names, which the roster refresh changes: the retry still
// resends the first request's name and room.
it.each([
  ['a peer Bot on another computer', [here, mini], { ...mini, title: 'Reviewer' }],
  ['a Bot on this computer', [here, beta], { ...beta, title: 'Reviewer' }]
] as const)(
  'a retry after a lost reply reuses the room of an unnamed draft when %s is renamed',
  async (_, roster, renamed) => {
    const rooms = loseFirstReply()

    const { onCreated, rerender } = await draft(
      [...roster],
      base =>
        [...base.querySelectorAll('[data-bot-face]')].filter(face =>
          ['default', roster[1].name].includes(face.getAttribute('data-bot-face')!)
        ),
      ''
    )

    rerender(<CreateGroupChatDialog onClose={vi.fn()} onCreated={onCreated} open roster={[here, renamed]} />)
    await act(async () => {})
    await submit()
    await waitFor(() => expect(onCreated).toHaveBeenCalledOnce())
    expect(creates()[1]).toEqual(creates()[0])
    expect(creates()[0].name).not.toContain('Reviewer')
    expect(rooms.size).toBe(1)
  }
)

it('a retry after a lost reply reuses the room when a mirrored room took its generated name', async () => {
  const rooms = loseFirstReply()
  const { onCreated } = await draft([here, beta], base => [
    base.querySelector('[data-bot-face="default"]')!,
    base.querySelector('[data-bot-face="beta"]')!
  ])
  // Another Desktop's room called "Team" arrives through sync: a fresh name would now be "Team 2".
  updateGroupChat('Team', room => ({ ...room, members: [here, beta] }), { sync: false })
  await submit()
  await waitFor(() => expect(onCreated).toHaveBeenCalledOnce())
  expect(creates()[1]).toEqual(creates()[0])
  expect(creates()[0].name).toBe('Team')
  expect(rooms.size).toBe(1)
})

it('invites every Bot of another computer at once, so a slow computer holds up no other', async () => {
  const STUDIO = peerGroupCapabilities('install:b27c0000000000000000000000000000')
  let release!: () => void
  const slow = new Promise<void>(resolve => {
    release = resolve
  })
  const { create } = strictGroupCreation()
  answerGroupRequests(request, {
    local: (method, params) =>
      method === 'groups.create'
        ? create(params)
        : (
            {
              'groups.capabilities': HOME,
              'groups.peer.register': { ...PEER_REGISTERED, target_install_id: params.catalog?.installation_id }
            } as Record<string, unknown>
          )[method],
    mini: method =>
      method === 'groups.peer.invite'
        ? slow.then(() => peerInvitation())
        : method === 'groups.capabilities'
          ? MINI
          : undefined,
    studio: method =>
      method === 'groups.peer.invite'
        ? { ...peerInvitation(), catalog: STUDIO.room_link.catalog, endpoint: STUDIO.room_link.endpoint }
        : method === 'groups.capabilities'
          ? STUDIO
          : undefined
  })
  const onCreated = vi.fn()
  const view = render(
    <CreateGroupChatDialog
      onClose={vi.fn()}
      onCreated={onCreated}
      open
      roster={[here, mini, peerRosterRow('studio', 'Studio')]}
    />
  )
  await waitFor(() => expect(request.mock.calls.filter(call => call[1] === 'groups.capabilities')).toHaveLength(3))
  await act(async () => {})
  view.baseElement.querySelectorAll('[data-bot-face="default"]').forEach(face => fireEvent.click(box(face)))
  fireEvent.change(screen.getByRole('textbox', { name: 'Group name' }), { target: { value: 'Team' } })
  await waitFor(() =>
    expect((screen.getByRole('button', { name: 'Create group (3)' }) as HTMLButtonElement).disabled).toBe(false)
  )
  await submit()
  const registered = () =>
    request.mock.calls.filter(call => call[1] === 'groups.peer.register').map(call => call[2].member_id)
  // The Mac Mini has not answered its invitation yet; the Studio's Bot joins anyway.
  await waitFor(() => expect(registered()).toEqual(['hermes-studio']))
  await act(async () => {
    release()
  })
  await waitFor(() => expect(onCreated).toHaveBeenCalledOnce())
  expect(registered()).toEqual(['hermes-studio', 'hermes-mac-mini'])
  expect(notify).not.toHaveBeenCalled()
})

it('offers no group picture in New group chat: a gateway group chat has none to save', async () => {
  loseFirstReply()
  render(<CreateGroupChatDialog onClose={vi.fn()} open roster={[here, beta]} />)
  await waitFor(() => expect(request.mock.calls.some(call => call[1] === 'groups.capabilities')).toBe(true))
  await act(async () => {})
  expect(screen.getByRole('textbox', { name: 'Group name' })).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'Upload' })).toBeNull()
})
