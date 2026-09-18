import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

vi.mock('@/hermes', () => ({
  setApiRequestProfile: () => {},
  setSessionStampsRemote: vi.fn()
}))

import { $cronSessions, $selectedStoredSessionId, $sessions } from '@/store/session'
import { setStampColor } from '@/store/session-stamp'

import { SessionStamp, SessionStamps, SessionTabStamp } from './session-stamp'

afterEach(cleanup)

beforeEach(() => {
  $sessions.set([])
  $cronSessions.set([])
  $selectedStoredSessionId.set(null)
})

const chip = (container: HTMLElement, label: string) => container.querySelector<HTMLElement>(`[data-session-stamp="${label}"]`)

// jsdom's selector engine cannot match an ATTRIBUTE VALUE holding an astral
// character (`[data-session-stamp="🔥"]` returns null over correct DOM), so an
// emoji chip is found by its text. The bubble the app itself paints is unaffected.
const emojiChip = (container: HTMLElement, label: string) =>
  [...container.querySelectorAll<HTMLElement>('[data-session-stamp]')].find(node => node.textContent === label)

const row = (id: string, extra: Partial<SessionInfo> = {}): SessionInfo =>
  ({ id, message_count: 1, source: 'cli', started_at: 0, title: id, ...extra }) as SessionInfo

describe('SessionStamp', () => {
  it('renders nothing at all when the session carries no stamp', () => {
    expect(render(<SessionStamp stamp={null} />).container.firstChild).toBeNull()
    expect(render(<SessionStamp stamp="" />).container.firstChild).toBeNull()
    expect(render(<SessionStamp stamp="   " />).container.firstChild).toBeNull()
  })

  it('tints a preset by its meaning and falls back to the accent for anything else', () => {
    expect(chip(render(<SessionStamp stamp="Merged" />).container, 'Merged')?.className).toContain('--ui-green')
    expect(chip(render(<SessionStamp stamp="Hold" />).container, 'Hold')?.className).toContain('--ui-red')
    // A label the app has never heard of still renders — as the user's own.
    expect(chip(render(<SessionStamp stamp="Waiting on CI" />).container, 'Waiting on CI')?.className).toContain(
      '--ui-accent'
    )
  })

  it('keeps the label’s own case inside a tab label, which uppercases its text', () => {
    // PaneTabLabel wraps every tab label in a span carrying `uppercase`, and text
    // transforms inherit: without `normal-case` on the chip itself, "mrg" reads
    // as "MRG" in the strip while the session list shows "mrg".
    expect(chip(render(<SessionStamp stamp="mrg" />).container, 'mrg')?.className).toContain('normal-case')
  })

  it('paints an emoji stamp as the GLYPH — no capsule, no hue', () => {
    // A colour emoji is drawn by the platform's own emoji font (Apple Color Emoji
    // on macOS, Segoe UI Emoji on Windows), so a tinted capsule would claim a
    // colour choice nobody can see and 9px text sizing would shrink the mark. The
    // chip is the emoji at its own size; the text chip keeps both.
    const emoji = emojiChip(render(<SessionStamp stamp="🔥" />).container, '🔥') as HTMLElement
    const text = chip(render(<SessionStamp stamp="Merged" />).container, 'Merged') as HTMLElement

    expect(emoji.className).toContain('text-[0.8125rem]')
    expect(emoji.className).not.toContain('color-mix')
    expect(emoji.className).not.toContain('--ui-accent')
    expect(text.className).toContain('color-mix')
    expect(text.className).toContain('--ui-green')
  })

  it('never tints an emoji, even with a colour stored for that label', () => {
    // The menu hides the colour door for an emoji title for the same reason; a
    // stale override (written before the emoji was a stamp) must not resurrect it.
    setStampColor('🔥', 'hsl(0 68% 58%)')

    const emoji = emojiChip(render(<SessionStamp stamp="🔥" />).container, '🔥') as HTMLElement

    expect(emoji.style.color).toBe('')
    expect(emoji.className).not.toContain('color-mix')

    setStampColor('🔥', null)
  })
})

describe('SessionStamps', () => {
  it('paints every label the session carries, in order', () => {
    const { container } = render(<SessionStamps stamps={['WIP', 'Handoff']} />)
    const labels = [...container.querySelectorAll('[data-session-stamp]')].map(node => node.textContent)

    expect(labels).toEqual(['WIP', 'Handoff'])
  })

  it('renders nothing at all for a session with no stamps', () => {
    expect(render(<SessionStamps stamps={[]} />).container.firstChild).toBeNull()
  })

  it('never paints more than the cap, whatever it is handed', () => {
    // The write path refuses a fourth label; this is the render-side belt to that
    // brace, so a hand-built list cannot paint a row nobody can read.
    const { container } = render(<SessionStamps stamps={['WIP', 'Hold', 'Review', 'Merged']} />)

    expect(container.querySelectorAll('[data-session-stamp]')).toHaveLength(3)
  })
})

describe('SessionTabStamp', () => {
  it('resolves the session behind a tile pane, through the lineage ids too', () => {
    $sessions.set([row('tip', { _lineage_root_id: 'root', stamps: ['Review', 'Hold'] })])

    const one = render(<SessionTabStamp paneId="session-tile:tip" />).container
    expect(chip(one, 'Review')).toBeTruthy()
    expect(chip(one, 'Hold')).toBeTruthy()
    // A pane holding an older segment of the compression chain still resolves.
    expect(chip(render(<SessionTabStamp paneId="session-tile:root" />).container, 'Hold')).toBeTruthy()
  })

  it('renders nothing for a non-session pane or an unstamped session', () => {
    $sessions.set([row('tip')])

    expect(render(<SessionTabStamp paneId="workspace" />).container.firstChild).toBeNull()
    expect(render(<SessionTabStamp paneId="session-tile:tip" />).container.firstChild).toBeNull()
    expect(render(<SessionTabStamp paneId="terminal" />).container.firstChild).toBeNull()
  })

  it('resolves the MAIN tab to the window’s primary session', () => {
    // The main tab's pane id carries no session of its own: it IS whichever
    // session the window is showing, which is why it was the one tab whose
    // stamps never painted.
    $sessions.set([row('primary', { stamps: ['WIP', 'Hold'] })])
    $selectedStoredSessionId.set('primary')

    const main = render(<SessionTabStamp paneId="workspace" />).container

    expect(chip(main, 'WIP')).toBeTruthy()
    expect(chip(main, 'Hold')).toBeTruthy()
    // A fresh draft is a main tab with no session behind it — nothing to wear.
    $selectedStoredSessionId.set(null)
    expect(render(<SessionTabStamp paneId="workspace" />).container.firstChild).toBeNull()
  })

  it('still shows a row that carries only the older single label', () => {
    $sessions.set([row('tip', { stamp: 'Handoff' })])

    expect(chip(render(<SessionTabStamp paneId="session-tile:tip" />).container, 'Handoff')).toBeTruthy()
  })

  it('paints the LIST row\u2019s own label, never a re-cased lineage mate', () => {
    // One conversation can be claimed by two rows (a recents row on the live tip
    // and, say, a cron row resolving through the same lineage root). The tab has
    // to show the string the session LIST shows \u2014 same case, same label \u2014 rather
    // than whichever row happened to be mapped last.
    $sessions.set([row('tip', { _lineage_root_id: 'root', stamps: ['mrg'] })])
    $cronSessions.set([row('other', { _lineage_root_id: 'root', stamps: ['Merged'] })])

    expect(chip(render(<SessionTabStamp paneId="session-tile:root" />).container, 'mrg')).toBeTruthy()
  })
})
