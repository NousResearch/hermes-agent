import { describe, expect, it } from 'vitest'

import {
  activeTimelineIndex,
  deriveTimelineEntries,
  dockGrowth,
  dockRows,
  dockScale,
  sameTimelineEntries,
  timelineBarWidth,
  timelineDockLabel,
  timelineIndexAt,
  timelinePreview
} from './timeline-data'

describe('timelinePreview', () => {
  it('collapses whitespace to a single line', () => {
    expect(timelinePreview('hello\n\n  world\tagain')).toBe('hello world again')
  })

  it('truncates with an ellipsis past the limit', () => {
    const out = timelinePreview('abcdefghij', 5)
    expect(out).toBe('abcd…')
    expect(out.length).toBe(5)
  })
})

describe('timelineDockLabel', () => {
  it('keeps the first thirty characters and ellipsizes the rest', () => {
    expect(timelineDockLabel('abcdefghij'.repeat(4))).toBe(`${'abcdefghij'.repeat(3)}…`)
  })

  it('leaves a fitting prompt alone and collapses whitespace', () => {
    expect(timelineDockLabel('fix  the\nrail')).toBe('fix the rail')
    expect(timelineDockLabel('hello world again')).toBe('hello world again')
  })

  it('counts code points so CJK and emoji stay whole', () => {
    expect(timelineDockLabel('あ'.repeat(40))).toBe(`${'あ'.repeat(30)}…`)
    expect(timelineDockLabel('🧪'.repeat(40))).toBe(`${'🧪'.repeat(30)}…`)
  })
})

describe('timelineBarWidth', () => {
  it('keeps active maximal and uses the same curve for hover', () => {
    expect(timelineBarWidth(10, 10, 50)).toBe(1)
    expect(timelineBarWidth(50, 10, 50)).toBe(1)
    expect(timelineBarWidth(11, 10, null)).toBe(timelineBarWidth(51, 10, 50))
  })

  it('does not change bars outside the local focus neighborhood', () => {
    expect(timelineBarWidth(30, 10, 50)).toBe(0.5)
    expect(timelineBarWidth(11, 10, null)).toBeGreaterThan(timelineBarWidth(12, 10, null))
    expect(timelineBarWidth(11, 0, 10.5)).toBe(timelineBarWidth(10, 0, 10.5))
  })
})

describe('dockScale', () => {
  it('peaks at the pointer and decays to nothing at the radius', () => {
    expect(dockScale(0)).toBe(1)
    expect(dockScale(1)).toBeGreaterThan(dockScale(2))
    expect(dockScale(-2.6)).toBe(0)
    expect(dockScale(Number.NaN)).toBe(0)
  })

  it('is continuous between ticks, so the ramp follows the cursor', () => {
    expect(Math.abs(dockScale(0.4) - dockScale(0.5))).toBeLessThan(0.12)
  })
})

describe('dockRows', () => {
  const starts = [0, 7, 14, 21, 28]

  it('leaves the rail alone without a pointer', () => {
    expect(dockRows(starts, 7, null, 6)).toEqual([
      { height: 7, scale: 0, top: 0 },
      { height: 7, scale: 0, top: 7 },
      { height: 7, scale: 0, top: 14 },
      { height: 7, scale: 0, top: 21 },
      { height: 7, scale: 0, top: 28 }
    ])
  })

  it('magnifies the row under the pointer and pushes the ones below down', () => {
    const rows = dockRows(starts, 7, 17.5, 6)

    expect(rows[2].scale).toBe(1)
    expect(rows[2].height).toBe(13)
    expect(rows[3].top).toBeGreaterThan(21)
    expect(rows[4].top).toBeGreaterThan(rows[3].top + 7)
  })

  it('reports the extra height the rail has to be given', () => {
    expect(dockGrowth(dockRows(starts, 7, null, 6), 7)).toBe(0)

    const rows = dockRows(starts, 7, 17.5, 6)

    expect(dockGrowth(rows, 7)).toBeCloseTo(6 * rows.reduce((sum, row) => sum + row.scale, 0))
    // Enough for the tallest row plus the ramp, and nothing near the pitch count.
    expect(dockGrowth(rows, 7)).toBeGreaterThan(6)
    expect(dockGrowth(rows, 7)).toBeLessThan(6 * starts.length)
  })

  it('follows the cursor between ticks instead of snapping', () => {
    const above = dockRows(starts, 7, 17.3, 6)
    const below = dockRows(starts, 7, 17.7, 6)

    expect(above[1].scale).toBeGreaterThan(below[1].scale)
    expect(below[3].scale).toBeGreaterThan(above[3].scale)
  })
})

describe('timelineIndexAt', () => {
  // Row 2 is magnified to the label box's height; the resting pitch is 7.
  const bands = [
    { height: 7, index: 0, top: 0 },
    { height: 7, index: 1, top: 7 },
    { height: 24, index: 2, top: 14 },
    { height: 7, index: 3, top: 38 }
  ]

  it('lets a magnified row own its whole painted band', () => {
    // y=34 is far below row 2's resting cell but still inside its band.
    expect(timelineIndexAt(34, bands, 7, 10)).toBe(2)
  })

  it('keeps the current row while the pointer is inside its band', () => {
    expect(timelineIndexAt(15, bands, 7, 10, 2)).toBe(2)
    expect(timelineIndexAt(37, bands, 7, 10, 2)).toBe(2)
  })

  it('falls back to the resting grid outside every band, clamped to the ends', () => {
    expect(timelineIndexAt(50, bands, 7, 10)).toBe(7)
    expect(timelineIndexAt(-30, bands, 7, 10, 0)).toBe(0)
    expect(timelineIndexAt(50, bands, 7, 3, 99)).toBe(2)
  })
})

describe('deriveTimelineEntries', () => {
  it('keeps non-empty user prompts in order', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: 'first' },
        { id: 'a1', role: 'assistant', text: 'answer' },
        { id: 'u2', role: 'user', text: '  second  ' }
      ])
    ).toEqual([
      { id: 'u1', label: 'first', preview: 'first' },
      { id: 'u2', label: 'second', preview: 'second' }
    ])
  })

  it('drops blanks and background-process notifications', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: '   ' },
        { id: 'u2', role: 'user', text: '[IMPORTANT: Background process 123 finished]' },
        { id: 'u3', role: 'user', text: 'real prompt' }
      ]).map(e => e.id)
    ).toEqual(['u3'])
  })

  it('carries a short label beside the long preview', () => {
    const [entry] = deriveTimelineEntries([{ id: 'u1', role: 'user', text: 'あ'.repeat(40) }])

    expect(entry.label).toBe(`${'あ'.repeat(30)}…`)
    expect(entry.preview).toBe('あ'.repeat(40))
  })
})

describe('sameTimelineEntries', () => {
  const rail = [
    { id: 'u1', label: 'first', preview: 'first' },
    { id: 'u2', label: 'second', preview: 'second' }
  ]

  it('treats an identical derivation as unchanged, so the memo can reuse it', () => {
    expect(sameTimelineEntries(rail, [...rail.map(e => ({ ...e }))])).toBe(true)
  })

  it('detects a changed preview, id, or length', () => {
    expect(sameTimelineEntries(rail, [rail[0], { id: 'u2', label: 'edited', preview: 'edited' }])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0], { id: 'u9', label: 'second', preview: 'second' }])).toBe(false)
    expect(sameTimelineEntries(rail, [rail[0]])).toBe(false)
  })

  it('is stable when a filtered-out prompt joins the transcript', () => {
    const withNoise = deriveTimelineEntries([
      { id: 'u1', role: 'user', text: 'first' },
      { id: 'u2', role: 'user', text: 'second' },
      { id: 'u3', role: 'user', text: '[IMPORTANT: Background process 7 finished]' }
    ])

    expect(sameTimelineEntries(rail, withNoise)).toBe(true)
  })
})

describe('activeTimelineIndex', () => {
  it('returns the last prompt scrolled to or above the top edge', () => {
    expect(activeTimelineIndex([-400, -10, 320])).toBe(1)
  })

  it('falls back to the first rendered entry', () => {
    expect(activeTimelineIndex([null, 120, 480])).toBe(1)
    expect(activeTimelineIndex([null, null])).toBe(0)
  })
})
