import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { promptSuggestions } from './prompt-suggestions'

const NOW = Date.UTC(2026, 10, 5, 0, 0, 0)
const DAY = 24 * 60 * 60 * 1000

function session(overrides: Partial<SessionInfo> & { id: string; title: string }): SessionInfo {
  return {
    ended_at: null,
    input_tokens: 0,
    is_active: false,
    last_active: NOW - DAY,
    message_count: 4,
    model: null,
    output_tokens: 0,
    preview: null,
    source: 'desktop',
    started_at: NOW - DAY,
    tool_call_count: 0,
    ...overrides
  } as SessionInfo
}

const texts = (sessions: SessionInfo[], options?: Parameters<typeof promptSuggestions>[1]) =>
  promptSuggestions(sessions, { now: NOW, ...options }).map(item => item.text)

describe('promptSuggestions', () => {
  it('returns recent human asks newest first', () => {
    const rows = [
      session({ id: 'a', last_active: NOW - 3 * DAY, title: 'Audit budget after payday' }),
      session({ id: 'b', last_active: NOW - DAY, title: 'Track Becky messages and update NAHC repo' }),
      session({ id: 'c', last_active: NOW - 2 * DAY, title: 'Script to download Cole saved Instagram media' })
    ]

    expect(texts(rows)).toEqual([
      'Track Becky messages and update NAHC repo',
      'Script to download Cole saved Instagram media',
      'Audit budget after payday'
    ])
  })

  it('never suggests the conversation on screen', () => {
    const rows = [
      session({ id: 'open', title: 'Fix the composer placeholder' }),
      session({ id: 'other', title: 'Set up NAHC Squarespace redesign repo' })
    ]

    expect(texts(rows, { excludeSessionId: 'open' })).toEqual(['Set up NAHC Squarespace redesign repo'])
  })

  it('drops the agent own work: cron, one-shots, delegate children', () => {
    const rows = [
      session({ id: 'cron', source: 'cron', title: 'nahc-becky-watch · Nov 05 19:00' }),
      session({ id: 'oneshot', source: 'oneshot', title: 'What HTTP port did the agent board use' }),
      session({ id: 'child', source: 'subagent', title: 'Review the uncommitted diff' }),
      session({ id: 'titled', source: 'desktop', title: 'Subagent: Fix ONLY these review findings' }),
      session({ id: 'real', title: 'Update the astro-blog work entry' })
    ]

    expect(texts(rows)).toEqual(['Update the astro-blog work entry'])
  })

  it('drops drafts, hidden rows, stale rows and slash commands', () => {
    const rows = [
      session({ id: 'draft', message_count: 1, title: 'Half typed idea about wei' }),
      session({ hidden: true, id: 'hidden', title: 'Hidden bot chat about the board' }),
      session({ id: 'stale', last_active: NOW - 90 * DAY, title: 'Old thing from last quarter' }),
      session({ id: 'slash', title: '/compact now please' }),
      session({ id: 'real', title: 'Pick up the Beth Hurley revisions' })
    ]

    expect(texts(rows)).toEqual(['Pick up the Beth Hurley revisions'])
  })

  it('collapses a repeated title into one chip', () => {
    const rows = [
      session({ id: 'a', last_active: NOW - DAY, title: 'Audit AWS stacks and shut down what is idle #2' }),
      session({ id: 'b', last_active: NOW - 2 * DAY, title: 'Audit AWS stacks and shut down what is idle' }),
      session({ id: 'c', last_active: NOW - 3 * DAY, title: 'Set up the spine feeds' })
    ]

    expect(texts(rows)).toEqual(['Audit AWS stacks and shut down what is idle', 'Set up the spine feeds'])
  })

  it('clips a long title on a word boundary rather than mid-word', () => {
    const title = 'Review the Natural Animal Health Clinic website as Marcus, the persona in personas/02.md'
    const [chip] = texts([session({ id: 'a', title })])

    expect(chip!.length).toBeLessThanOrEqual(72)
    expect(chip!.endsWith('…')).toBe(true)
    expect(chip!.slice(0, -1).endsWith(' ')).toBe(false)
    expect(title.startsWith(chip!.slice(0, -1))).toBe(true)
  })

  it('honours the limit so one pool can feed chips and the placeholder', () => {
    const rows = Array.from({ length: 9 }, (_, index) =>
      session({ id: `s${index}`, last_active: NOW - index * DAY, title: `Distinct task number ${index} for the board` })
    )

    expect(promptSuggestions(rows, { limit: 3, now: NOW })).toHaveLength(3)
  })
})
