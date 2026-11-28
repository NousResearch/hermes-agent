/**
 * A room member's face must paint from the SAME meta record the Bots list uses.
 *
 * Bot meta is persisted under the route-qualified key (`connectionId::profile`,
 * `botMetaKey`) once the v2 migration ran, so the room's bare `meta[name]` read
 * missed every migrated install: the face fell back to the name-derived
 * shape/colour while the Bots list showed the bot's own image (#148923), and an
 * Edit Profile change never reached the group. The room knows only a profile
 * name, so the member resolves through its roster row + `botRosterMeta` — with
 * the room's connection the row is exact, and without it a bare name counts only
 * while exactly one roster row carries it (one connection's bot must never
 * borrow another's face). A bare-name key survives as the pre-migration rung,
 * single-source rosters only.
 */

import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')

  return pluginSdkMock({ state: { connectionId: { get: () => 'home' }, profile: { get: () => 'default' } } })
})

vi.mock('./shared', () => ({ getPluginCtx: () => ({ storage: undefined }), ID: 'hermes-bots' }))
vi.mock('./avatar-image', () => ({ isBackfilledFacePng: () => false }))

import { $botMeta, $lastRoster } from './data'
import { RoomBotAvatar } from './group-room-avatar'
import { indexAliasRoutes } from './routing'
import type { RosterRow } from './types'

const PHOTO = 'data:image/png;base64,AAAA'

const rosterRow = (over: Partial<RosterRow>): RosterRow =>
  ({ connectionId: 'home', connectionKind: 'remote', name: 'ada', sourceScoped: true, ...over }) as RosterRow

const photoOf = (container: HTMLElement) => container.querySelector('img')?.getAttribute('src') ?? null

beforeEach(() => {
  $botMeta.set({})
  $lastRoster.set([])
})

afterEach(cleanup)

describe('a canonical group member face', () => {
  it('comes from the scoped meta record the Bots list paints from', () => {
    $lastRoster.set([rosterRow({})])
    $botMeta.set({ 'home::ada': { image: PHOTO } })

    const { container } = render(<RoomBotAvatar connectionId="home" name="ada" />)

    expect(photoOf(container)).toBe(PHOTO)
  })

  it('never borrows a same-named bot from another connection', () => {
    $lastRoster.set([rosterRow({}), rosterRow({ connectionId: 'spare' })])
    $botMeta.set({ 'home::ada': { image: PHOTO } })

    const { container } = render(<RoomBotAvatar connectionId="spare" name="ada" />)

    expect(photoOf(container)).toBeNull()
  })

  it('stays faceless when the room owner is absent from the roster', () => {
    // The roster knows only another connection's same-named bot: a unique-name
    // fallback would paint that gateway's photo into this room.
    $lastRoster.set([rosterRow({ connectionId: 'spare' })])
    $botMeta.set({ 'spare::ada': { image: PHOTO }, ada: { image: PHOTO } })

    const { container } = render(<RoomBotAvatar connectionId="home" name="ada" />)

    expect(photoOf(container)).toBeNull()
  })

  it('paints its cached scoped record before the roster loads', () => {
    $botMeta.set({ 'home::ada': { image: PHOTO } })

    const { container } = render(<RoomBotAvatar connectionId="home" name="ada" />)

    expect(photoOf(container)).toBe(PHOTO)
  })

  it('reuses the alias mapping before the roster loads', () => {
    indexAliasRoutes([{ connectionId: 'other', mode: 'remote', profile: 'helper', targetProfile: 'default' }])
    $botMeta.set({ 'other::helper': { image: PHOTO } })

    const { container } = render(<RoomBotAvatar connectionId="other" name="default" />)

    expect(photoOf(container)).toBe(PHOTO)
  })

  it('resolves a unique name without the room connection (chips that only carry a profile)', () => {
    $lastRoster.set([rosterRow({})])
    $botMeta.set({ 'home::ada': { image: PHOTO } })

    const { container } = render(<RoomBotAvatar name="ada" />)

    expect(photoOf(container)).toBe(PHOTO)
  })

  it('does not guess between two connections carrying the same profile name', () => {
    $lastRoster.set([rosterRow({}), rosterRow({ connectionId: 'spare' })])
    $botMeta.set({ 'home::ada': { image: PHOTO } })

    const { container } = render(<RoomBotAvatar name="ada" />)

    expect(photoOf(container)).toBeNull()
  })

  it('still reads a pre-migration bare-name record on a single-source roster', () => {
    $lastRoster.set([{ name: 'ada' } as RosterRow])
    $botMeta.set({ ada: { image: PHOTO } })

    const { container } = render(<RoomBotAvatar name="ada" />)

    expect(photoOf(container)).toBe(PHOTO)
  })
})
