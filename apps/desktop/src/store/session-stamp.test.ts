import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

const patch = vi.fn<(id: string, stamps: string[], profile?: null | string) => Promise<{ ok: boolean }>>(() =>
  Promise.resolve({ ok: true })
)

vi.mock('@/hermes', () => ({
  // The session store reaches the profile store, which sets the request profile
  // at import time; this suite only cares about the stamp call.
  setApiRequestProfile: () => {},
  setSessionStampsRemote: (id: string, stamps: string[], profile?: null | string) => patch(id, stamps, profile)
}))

import { PROFILE_SWATCHES } from '@/lib/profile-color'
import { $cronSessions, $messagingSessions, $sessions } from '@/store/session'
import { $archivedSessions } from '@/store/sidebar-archive'

import {
  $deletedStampPresets,
  $sessionStamps,
  $stampColorOverrides,
  $stampPresets,
  $stampTitlePrefs,
  addStampTitle,
  applySessionStamp,
  applySessionStamps,
  deleteStampPreset,
  hasStampLabel,
  isEmojiStamp,
  normalizeSessionStamp,
  normalizeSessionStamps,
  restoreStampPresets,
  SESSION_STAMP_EMOJI,
  SESSION_STAMP_LIMIT,
  SESSION_STAMP_MAX_LENGTH,
  SESSION_STAMP_PRESETS,
  setStampColor,
  STAMP_SWATCHES,
  stampColorFor,
  stampLabels,
  toggleSessionStamp
} from './session-stamp'

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

beforeEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $archivedSessions.set([])
  $stampTitlePrefs.set({ added: [], deleted: [] })
  $stampColorOverrides.set({})
  window.localStorage.clear()
  patch.mockClear()
})

afterEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $archivedSessions.set([])
})

describe('normalizeSessionStamp', () => {
  it('trims, collapses interior whitespace and caps the length', () => {
    expect(normalizeSessionStamp('  Merged  ')).toBe('Merged')
    expect(normalizeSessionStamp('Waiting   on\n CI')).toBe('Waiting on CI')
    expect(normalizeSessionStamp('x'.repeat(40))?.length).toBe(24)
  })

  it('treats empty, whitespace-only and null as no stamp', () => {
    expect(normalizeSessionStamp('')).toBeNull()
    expect(normalizeSessionStamp('   ')).toBeNull()
    expect(normalizeSessionStamp(null)).toBeNull()
    expect(normalizeSessionStamp(undefined)).toBeNull()
  })

  it('caps on code points, so an emoji at the edge is never cut in half', () => {
    // Twenty-five fire glyphs are 50 UTF-16 units but 25 characters. A unit-based
    // slice keeps 24 units — twelve glyphs, and the boundary lands INSIDE a pair,
    // storing a lone surrogate that the backend keeps and the chip paints as a
    // replacement glyph.
    const capped = normalizeSessionStamp('🔥'.repeat(25)) as string

    expect(capped).toBe('🔥'.repeat(24))
    // `u` is load-bearing: without it the surrogate range matches the halves of
    // every astral character, so a correct string reads as broken.
    expect(/[\uD800-\uDFFF]/u.test(capped)).toBe(false)
    // And a stamp that is over the cap in UNITS while under it in characters is
    // not truncated at all (thirteen glyphs are 26 units, 13 characters).
    expect(normalizeSessionStamp('🔥'.repeat(13))).toBe('🔥'.repeat(13))
  })

  it('leaves an emoji stamp under the cap exactly as it was written', () => {
    // A ZWJ family is five code points and eight units: neither the count nor the
    // characters may be touched.
    expect(normalizeSessionStamp(' 👨‍👩‍👧 ')).toBe('👨‍👩‍👧')
    expect(normalizeSessionStamp('🔥')).toBe('🔥')
  })
})

describe('isEmojiStamp', () => {
  it('reads an emoji-only label as an emoji stamp, however the glyph is assembled', () => {
    expect(isEmojiStamp('🔥')).toBe(true)
    expect(isEmojiStamp('👍🏽')).toBe(true) // skin tone
    expect(isEmojiStamp('👨‍👩‍👧')).toBe(true) // ZWJ sequence
    expect(isEmojiStamp('🇺🇸')).toBe(true) // regional-indicator flag
    expect(isEmojiStamp('🔥✅')).toBe(true) // a row of them
  })

  it('leaves text, digits and mixed labels alone', () => {
    // A digit is not an emoji even though Unicode counts it as a component: a
    // numeric label must keep painting in the text chip.
    expect(isEmojiStamp('WIP')).toBe(false)
    expect(isEmojiStamp('123')).toBe(false)
    expect(isEmojiStamp('1🔥')).toBe(false)
    expect(isEmojiStamp('🔥 WIP')).toBe(false)
    expect(isEmojiStamp('')).toBe(false)
    expect(isEmojiStamp(null)).toBe(false)
  })
})

describe('normalizeSessionStamps', () => {
  it('keeps the order it is given, drops blanks and dedupes case-insensitively', () => {
    // The first spelling of a repeated label wins, so the chip is the one the
    // user typed first rather than whichever came last in the array.
    expect(normalizeSessionStamps(['WIP', ' Review ', 'wip', '', null, 'Hold'])).toEqual([
      'WIP',
      'Review',
      'Hold'
    ])
    expect(normalizeSessionStamps([])).toEqual([])
    expect(normalizeSessionStamps([null, undefined, '  '])).toEqual([])
  })
})

describe('stampLabels', () => {
  it('reads the list, and a row that only carries the older single label as a list of one', () => {
    expect(stampLabels(row('a', { stamps: ['WIP', 'Hold'] }))).toEqual(['WIP', 'Hold'])
    // Read-compat: an older backend (or an optimistic write) answers with `stamp`
    // alone, and that is still a stamped session.
    expect(stampLabels(row('a', { stamp: 'Merged' }))).toEqual(['Merged'])
    expect(stampLabels(row('a'))).toEqual([])
    expect(stampLabels(null)).toEqual([])
    // The list wins when both are present, so a stale singular mirror cannot double it.
    expect(stampLabels(row('a', { stamp: 'Merged', stamps: ['WIP'] }))).toEqual(['WIP'])
  })
})

describe('hasStampLabel', () => {
  it('matches a label case-insensitively, the way the menu marks a row', () => {
    expect(hasStampLabel(['WIP', 'Hold'], 'wip')).toBe(true)
    expect(hasStampLabel(['WIP'], 'Review')).toBe(false)
    expect(hasStampLabel(['WIP'], '   ')).toBe(false)
  })
})

describe('applySessionStamps', () => {
  it('paints the labels before the backend answers, and persists the normalized list', async () => {
    $sessions.set([row('a', { profile: 'work' })])

    const pending = applySessionStamps('a', 'work', ['  wip  ', 'Hold', 'wip'])

    // Optimistic: the row already shows them, so a slow round trip never reads as
    // "the click did nothing".
    expect($sessions.get()[0].stamps).toEqual(['wip', 'Hold'])

    await pending

    expect(patch).toHaveBeenCalledWith('a', ['wip', 'Hold'], 'work')
    expect($sessions.get()[0].stamps).toEqual(['wip', 'Hold'])
    // The singular mirror follows the list, so the two never disagree on the row.
    expect($sessions.get()[0].stamp).toBe('wip')
  })

  it('clears every label when handed an empty list', async () => {
    $sessions.set([row('a', { stamps: ['Merged', 'Hold'], stamp: 'Merged' })])

    await applySessionStamps('a', undefined, [])

    expect(patch).toHaveBeenCalledWith('a', [], undefined)
    expect($sessions.get()[0].stamps).toEqual([])
    expect($sessions.get()[0].stamp).toBeNull()
  })

  it('puts the row back when the backend refuses, rather than leaving stamps that are not there', async () => {
    patch.mockRejectedValueOnce(new Error('offline'))
    $sessions.set([row('a', { stamps: ['Hold'], stamp: 'Hold' })])

    const ok = await applySessionStamps('a', undefined, ['Merged'])

    expect(ok).toBe(false)
    expect($sessions.get()[0].stamps).toEqual(['Hold'])
  })

  it('patches every list that can hold the row, and leaves other rows untouched', async () => {
    $archivedSessions.set([row('a', { profile: 'default' })])
    $sessions.set([row('b')])

    await applySessionStamps('a', 'default', ['Review'])

    expect($archivedSessions.get()[0].stamps).toEqual(['Review'])
    expect($sessions.get()[0].stamps).toBeUndefined()
  })

  it('lands on a row addressed by an older lineage id, not only the live tip', async () => {
    // A tile's tab keeps the id it was opened with; after an auto-compression
    // that id is a lineage segment while the row's own id is the tip. Reading
    // the stamps through the tab must not wait for the next list poll.
    $sessions.set([row('tip', { _lineage_ids: ['root'], _lineage_root_id: 'root' })])

    await applySessionStamps('root', undefined, ['Handoff'])

    expect($sessions.get()[0].stamps).toEqual(['Handoff'])
    expect($sessionStamps.get().get('tip')).toEqual(['Handoff'])
    expect($sessionStamps.get().get('root')).toEqual(['Handoff'])
  })

  it('rolls a lineage-addressed row back to what it carried when the write fails', async () => {
    $sessions.set([row('tip', { _lineage_ids: ['root'], _lineage_root_id: 'root', stamps: ['Review'] })])
    patch.mockRejectedValueOnce(new Error('backend refused'))

    const ok = await applySessionStamps('root', undefined, ['Hold'])

    expect(ok).toBe(false)
    expect($sessions.get()[0].stamps).toEqual(['Review'])
  })

  it('leaves a row that already shows the labels alone, reference and all', async () => {
    const rows = [row('a', { stamps: ['WIP'] })]
    $sessions.set(rows)

    await applySessionStamps('a', undefined, ['WIP'])

    // No write happened (the page is unchanged), and the list keeps its identity
    // so React is not handed a fresh array of identical rows.
    expect(patch).toHaveBeenCalledWith('a', ['WIP'], undefined)
    expect($sessions.get()).toBe(rows)
  })
})

describe('applySessionStamp', () => {
  it('is the one-label door: it REPLACES the list', async () => {
    $sessions.set([row('a', { stamps: ['WIP', 'Hold'], stamp: 'WIP' })])

    await applySessionStamp('a', 'work', 'Merged')

    expect(patch).toHaveBeenCalledWith('a', ['Merged'], 'work')
    expect($sessions.get()[0].stamps).toEqual(['Merged'])
  })

  it('clears the list when handed an empty label', async () => {
    $sessions.set([row('a', { stamps: ['Merged'], stamp: 'Merged' })])

    await applySessionStamp('a', undefined, '')

    expect(patch).toHaveBeenCalledWith('a', [], undefined)
    expect($sessions.get()[0].stamps).toEqual([])
  })
})

describe('toggleSessionStamp', () => {
  it('adds a label the session does not carry, at the END of the list', async () => {
    $sessions.set([row('a', { stamps: ['WIP'] })])

    await toggleSessionStamp('a', undefined, '  Hold ')

    expect(patch).toHaveBeenCalledWith('a', ['WIP', 'Hold'], undefined)
    expect($sessions.get()[0].stamps).toEqual(['WIP', 'Hold'])
  })

  it('takes a label off when the session already carries it, case-insensitively', async () => {
    $sessions.set([row('a', { stamps: ['WIP', 'Hold', 'Review'] })])

    await toggleSessionStamp('a', undefined, 'hold')

    expect(patch).toHaveBeenCalledWith('a', ['WIP', 'Review'], undefined)
    expect($sessions.get()[0].stamps).toEqual(['WIP', 'Review'])
  })

  it('refuses a fourth label without writing, so the UI never sends what the API would refuse', async () => {
    $sessions.set([row('a', { stamps: ['WIP', 'Hold', 'Review'] })])

    const ok = await toggleSessionStamp('a', undefined, 'Merged')

    expect(ok).toBe(false)
    expect(patch).not.toHaveBeenCalled()
    expect($sessions.get()[0].stamps).toHaveLength(SESSION_STAMP_LIMIT)
  })

  it('still takes one off at the cap — a full session is not a locked one', async () => {
    $sessions.set([row('a', { stamps: ['WIP', 'Hold', 'Review'] })])

    expect(await toggleSessionStamp('a', undefined, 'Hold')).toBe(true)
    expect($sessions.get()[0].stamps).toEqual(['WIP', 'Review'])
  })
})

describe('$sessionStamps', () => {
  it('maps live and lineage ids to the label LIST, and skips unstamped rows', () => {
    $sessions.set([
      row('tip', { _lineage_ids: ['mid', 'root'], _lineage_root_id: 'root', stamps: ['WIP', 'Hold'] }),
      row('plain')
    ])

    expect($sessionStamps.get().get('tip')).toEqual(['WIP', 'Hold'])
    expect($sessionStamps.get().get('root')).toEqual(['WIP', 'Hold'])
    expect($sessionStamps.get().get('mid')).toEqual(['WIP', 'Hold'])
    expect($sessionStamps.get().has('plain')).toBe(false)
  })

  it('still resolves a row that only carries the older single label', () => {
    $sessions.set([row('tip', { stamp: 'Handoff' })])

    expect($sessionStamps.get().get('tip')).toEqual(['Handoff'])
  })
})

describe('the titles the Stamp submenu offers', () => {
  it('drops a deleted title, and puts it back without touching a stamp already written', () => {
    expect($stampPresets.get()).toEqual([...SESSION_STAMP_PRESETS])

    deleteStampPreset('Hold')

    expect($stampPresets.get()).toEqual(['Merged', 'WIP', 'Review', 'Handoff'])
    // Deleting the TITLE is not a data change: "Hold" stays a valid label (a
    // session already carrying it keeps it), it is simply no longer offered.
    expect(normalizeSessionStamp('Hold')).toBe('Hold')

    restoreStampPresets()

    expect($stampPresets.get()).toEqual([...SESSION_STAMP_PRESETS])
  })

  it('keeps a title the user adds, after the stock ones, and stores it', () => {
    addStampTitle('Blocked')

    expect($stampPresets.get()).toEqual([...SESSION_STAMP_PRESETS, 'Blocked'])
    // Stays for the next run — and for the next SESSION, which is the point.
    expect(window.localStorage.getItem('hermes.desktop.sessionStampTitles.v1')).toBe(
      '{"added":["Blocked"],"deleted":[]}'
    )
  })

  it('is idempotent and case-insensitive, and a title taken off stays off until it is typed again', () => {
    deleteStampPreset('wip')
    deleteStampPreset('WIP')
    addStampTitle('   ')

    expect($deletedStampPresets.get()).toEqual(['wip'])
    expect($stampPresets.get()).not.toContain('WIP')
    // None of those three calls added a title of their own.
    expect($stampPresets.get()).toEqual(['Merged', 'Review', 'Handoff', 'Hold'])

    // Typing it again is what brings a row back — as the user's own title, in
    // the casing they typed, and only once.
    addStampTitle('wip')

    expect($deletedStampPresets.get()).toEqual([])
    expect($stampPresets.get().filter(title => title.toLowerCase() === 'wip')).toEqual(['wip'])
  })

  it('re-creates a removed title with the spelling the user typed, never the old row', () => {
    // Reported: taking "Hold" off the menu and adding "hold" used to un-delete
    // the stock title, so the menu showed "Hold" again. Deleted means gone, and
    // the title that comes back is the one the user just wrote.
    deleteStampPreset('Hold')
    expect($stampPresets.get()).not.toContain('Hold')

    addStampTitle('hold')

    expect($stampPresets.get().filter(title => title.toLowerCase() === 'hold')).toEqual(['hold'])
    expect($stampPresets.get()).not.toContain('Hold')
  })

  it('restores a deleted title the user added as well as a stock one', () => {
    addStampTitle('Blocked')
    deleteStampPreset('Blocked')

    expect($stampPresets.get()).not.toContain('Blocked')

    restoreStampPresets()

    expect($stampPresets.get()).toEqual([...SESSION_STAMP_PRESETS, 'Blocked'])
  })
})

describe('the emoji the Emoji panel offers', () => {
  it('offers emoji only, one glyph each, with no repeats', () => {
    expect(SESSION_STAMP_EMOJI.length).toBeGreaterThan(0)
    // The wire contract: every entry has to read as an emoji stamp, or the grid
    // would paint a button whose label renders in the text chip.
    expect(SESSION_STAMP_EMOJI.every(emoji => isEmojiStamp(emoji))).toBe(true)
    // A stamp is a mark beside a title, not a phrase.
    expect(SESSION_STAMP_EMOJI.every(emoji => Array.from(emoji).length <= SESSION_STAMP_MAX_LENGTH)).toBe(true)
    expect(new Set(SESSION_STAMP_EMOJI).size).toBe(SESSION_STAMP_EMOJI.length)
  })

  it('lets a picked emoji become a menu title of its own, and be taken off again', () => {
    // This is what tapping in the panel does: the emoji joins the user's own
    // titles, so the ones actually in use are one tap next time.
    addStampTitle('🦄')

    expect($stampPresets.get()).toEqual([...SESSION_STAMP_PRESETS, '🦄'])

    deleteStampPreset('🦄')

    expect($stampPresets.get()).not.toContain('🦄')
    // Off the MENU, still a valid label: a session already stamped 🦄 keeps it.
    expect(normalizeSessionStamp('🦄')).toBe('🦄')
  })
})

describe('stamp colours', () => {
  it('sets, resolves and clears one title’s colour', () => {
    setStampColor('Merged', '#ff0000')

    expect(stampColorFor('Merged', $stampColorOverrides.get())).toBe('#ff0000')
    // Case-insensitive on read, so the chip and the menu always agree.
    expect(stampColorFor('merged', $stampColorOverrides.get())).toBe('#ff0000')
    expect(stampColorFor('WIP', $stampColorOverrides.get())).toBeNull()
    expect(window.localStorage.getItem('hermes.desktop.sessionStampColors')).toContain('#ff0000')

    setStampColor('Merged', null)

    expect(stampColorFor('Merged', $stampColorOverrides.get())).toBeNull()
  })

  it('offers a finer wheel than the profile rail, on the profile palette’s own saturation', () => {
    expect(STAMP_SWATCHES.length).toBeGreaterThan(PROFILE_SWATCHES.length)
    // Every hue at the profile palette's saturation/lightness, so a color picked
    // before the panel widened still rings its own swatch as the current one.
    expect(STAMP_SWATCHES.every(swatch => /^hsl\(\d+ 68% 58%\)$/.test(swatch))).toBe(true)
    expect(PROFILE_SWATCHES.every(swatch => STAMP_SWATCHES.includes(swatch))).toBe(true)
  })
})
