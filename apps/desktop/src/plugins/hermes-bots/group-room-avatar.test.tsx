/**
 * Room member faces resolve through the member's OWNING connection, not the
 * bare name. The v2 meta store keys faces by `connectionId::profile`, and
 * pre-v2 bare-name entries ride along inside migrated snapshots as v1
 * rollback residue — so a room avatar that looks the bare name up directly
 * paints the residue (on this class of install: a 160x160 roster backfill,
 * which the photo filter then blanks into the deterministic glyph) over the
 * connection's real face.
 *
 * The ladder RoomBotAvatar must follow:
 *   1. a source-scoped roster row for the member → that connection's scoped
 *      meta key (plus alias routes), exactly like the roster itself resolves
 *      (botRosterMeta) — the bare name is never consulted;
 *   2. no scoped row claims the member (legacy single-source install, or a
 *      roster snapshot that has not painted yet) → the bare name, the
 *      documented v1 contract in data.ts.
 */

import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@hermes/plugin-sdk', async () => ({ ...await (await import('./group-test-utils')).pluginSdkMock({}) }))

import { $botMeta, $lastRoster } from './data'
import { RoomBotAvatar } from './group-room-avatar'
import type { RosterRow } from './types'

/** A PNG-shaped data URL carrying the dimensions isBackfilledFacePng reads
 *  out of the IHDR (bytes 16-19 width, 20-23 height). */
const pngDataUrl = (width: number, height: number) => {
  const bytes = new Uint8Array(24)

  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0)
  bytes.set([0x49, 0x48, 0x44, 0x52], 12) // 'IHDR'

  const view = new DataView(bytes.buffer)
  view.setUint32(8, 13)
  view.setUint32(16, width)
  view.setUint32(20, height)

  return `data:image/png;base64,${btoa(String.fromCharCode(...bytes))}`
}

// The bug report's exact store: the bare `venus` entry is the stale 160x160
// roster backfill; the connection's scoped entry carries the real 256 face.
const STALE_BACKFILL = pngDataUrl(160, 160)
const REAL_PHOTO = pngDataUrl(256, 256)

const faceSrc = (name: string) => {
  const { container } = render(<RoomBotAvatar name={name} />)

  return container.querySelector('img')?.getAttribute('src') ?? null
}

const scopedRow = (name: string, connectionId: string): RosterRow => ({ connectionId, name, remoteSource: true, sourceScoped: true })

afterEach(() => {
  cleanup()
  $botMeta.set({})
  $lastRoster.set([])
})

it('resolves a member face through its owning connection before any bare-name entry', () => {
  $lastRoster.set([scopedRow('venus', 'remote-1')])
  $botMeta.set({ 'remote-1::venus': { image: REAL_PHOTO }, venus: { image: STALE_BACKFILL } })

  expect(faceSrc('venus')).toBe(REAL_PHOTO)
})

it('the filter judges the resolved image: a scoped 160x160 backfill degrades to the glyph instead of detouring to the bare photo', () => {
  $lastRoster.set([scopedRow('venus', 'remote-1')])
  // Inverse of the fixture above: the SCOPED entry is the 160x160 residue
  // and the bare entry is the tempting real photo. Resolution still ends on
  // the scoped key and the photo filter blanks it there — the filter only
  // judges the image resolution ENDED on, it must never run mid-resolution
  // to skip the scoped entry and borrow the bare photo.
  $botMeta.set({ 'remote-1::venus': { image: STALE_BACKFILL }, venus: { image: REAL_PHOTO } })

  expect(faceSrc('venus')).toBeNull()
})

it('keeps the bare-name fallback for legacy single-source installs', () => {
  // Legacy row shape: no source qualification, so no scoped key exists to
  // claim the member and the bare name remains its meta key (data.ts v1
  // contract — this fixture is the only guard for that contract on modern
  // code).
  $lastRoster.set([{ name: 'venus' }])
  $botMeta.set({ venus: { image: REAL_PHOTO } })

  expect(faceSrc('venus')).toBe(REAL_PHOTO)
})

it('renders a scoped-only snapshot', () => {
  $lastRoster.set([scopedRow('venus', 'remote-1')])
  $botMeta.set({ 'remote-1::venus': { image: REAL_PHOTO } })

  expect(faceSrc('venus')).toBe(REAL_PHOTO)
})

it('a scoped row without scoped meta degrades to glyph even when the bare entry is a real photo', () => {
  $lastRoster.set([scopedRow('venus', 'remote-1')])
  // The bare entry is a REAL photo on purpose, not the 160x160 backfill: a
  // borrowed backfill is blanked by the photo filter anyway, so only a real
  // image discriminates no-borrow from borrow-then-render. Same doctrine as
  // the roster — two sources must not wear each other's face; a scoped
  // member whose meta is missing degrades to the glyph.
  $botMeta.set({ venus: { image: REAL_PHOTO } })

  expect(faceSrc('venus')).toBeNull()
})

it('the filter still judges the resolved image: a bare backfill alone renders the glyph', () => {
  $lastRoster.set([{ name: 'venus' }])
  $botMeta.set({ venus: { image: STALE_BACKFILL } })

  expect(faceSrc('venus')).toBeNull()
})
