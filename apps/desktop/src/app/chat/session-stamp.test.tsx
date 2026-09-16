import { cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

vi.mock('@/hermes', () => ({
  setApiRequestProfile: () => {},
  setSessionStampRemote: vi.fn()
}))

import { $sessions } from '@/store/session'

import { SessionStamp, SessionTabStamp } from './session-stamp'

afterEach(cleanup)

beforeEach(() => {
  $sessions.set([])
})

const chip = (container: HTMLElement, label: string) => container.querySelector<HTMLElement>(`[data-session-stamp="${label}"]`)

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
})

describe('SessionTabStamp', () => {
  it('resolves the session behind a tile pane, through the lineage ids too', () => {
    $sessions.set([row('tip', { _lineage_root_id: 'root', stamp: 'Review' })])

    expect(chip(render(<SessionTabStamp paneId="session-tile:tip" />).container, 'Review')).toBeTruthy()
    // A pane holding an older segment of the compression chain still resolves.
    expect(chip(render(<SessionTabStamp paneId="session-tile:root" />).container, 'Review')).toBeTruthy()
  })

  it('renders nothing for a non-session pane or an unstamped session', () => {
    $sessions.set([row('tip')])

    expect(render(<SessionTabStamp paneId="workspace" />).container.firstChild).toBeNull()
    expect(render(<SessionTabStamp paneId="session-tile:tip" />).container.firstChild).toBeNull()
  })
})
