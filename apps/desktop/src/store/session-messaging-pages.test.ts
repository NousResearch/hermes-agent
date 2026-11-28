import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { reconcileMessagingRefresh } from './session-messaging-pages'

const row = (id: string, source: string, over: Partial<SessionInfo> = {}): SessionInfo =>
  ({
    archived: false,
    cwd: null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: false,
    last_active: 0,
    message_count: 2,
    model: null,
    output_tokens: 0,
    preview: null,
    profile: 'default',
    source,
    started_at: 0,
    title: id,
    tool_call_count: 0,
    ...over
  }) as SessionInfo

describe('reconcileMessagingRefresh', () => {
  it('a capped seed keeps rows paged in past it, and drops only rows it proves are gone', () => {
    // 13 Feishu rows loaded through "Load more" (f-0 newest); the next seed
    // is capped by Telegram traffic and carries only the 4 newest Feishu rows.
    const paged = Array.from({ length: 13 }, (_, i) => row(`f-${i}`, 'feishu', { last_active: 100 - i }))
    const gone = row('f-gone', 'feishu', { last_active: 99 }) // newer than the seed's oldest, yet missing
    const otherProfile = row('w-1', 'feishu', { last_active: 1, profile: 'work' })

    const seed = [
      { ...paged[0], title: 'updated' },
      ...paged.slice(1, 4),
      ...Array.from({ length: 96 }, (_, i) => row(`t-${i}`, 'telegram', { last_active: 97 }))
    ]

    const result = reconcileMessagingRefresh([gone, ...paged, otherProfile], seed, true, 'default')
    const feishu = result.filter(s => s.source === 'feishu').map(s => s.id)

    expect(feishu).toEqual(paged.map(s => s.id))
    expect(result.find(s => s.id === 'f-0')?.title).toBe('updated')
    expect(result.filter(s => s.source === 'telegram')).toHaveLength(96)

    // A seed under the cap is the whole list.
    expect(reconcileMessagingRefresh(paged, seed.slice(0, 2), false, 'default').map(s => s.id)).toEqual(['f-0', 'f-1'])
  })
})
