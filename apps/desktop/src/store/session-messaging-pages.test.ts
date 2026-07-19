import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { reconcileMessagingRefresh } from './session-messaging-pages'

describe('reconcileMessagingRefresh', () => {
  const row = (id: string, source: string, over: Partial<SessionInfo> = {}): SessionInfo =>
    ({ archived: false, cwd: null, ended_at: null, id, input_tokens: 0, is_active: false, last_active: 0, message_count: 2, model: null, output_tokens: 0, preview: null, source, started_at: 0, title: id, tool_call_count: 0, ...over }) as SessionInfo

  it('keeps paged-over rows when the cap was hit', () => {
    const paged = Array.from({ length: 13 }, (_, i) => row(`f-${i}`, 'feishu'))
    const seed = [
      ...paged.slice(0, 4).map(r => ({ ...r, last_active: r.last_active + 1 })),
      ...Array.from({ length: 96 }, (_, i) => row(`t-${i}`, 'telegram'))
    ]

    const result = reconcileMessagingRefresh(paged, seed, true)

    expect(result.filter(s => s.source === 'feishu')).toHaveLength(13)
    expect(result.filter(s => s.source === 'telegram')).toHaveLength(96)
  })

  it('truncates to incoming when not truncated', () => {
    const paged = Array.from({ length: 5 }, (_, i) => row(`f-${i}`, 'feishu'))
    const seed = [row('f-0', 'feishu', { last_active: 2 })]

    const result = reconcileMessagingRefresh(paged, seed, false)

    expect(result).toHaveLength(1)
    expect(result[0].id).toBe('f-0')
  })

  it('deduplicates paged rows that appear in the incoming seed', () => {
    const paged = [row('f-0', 'feishu'), row('f-1', 'feishu')]
    const seed = [row('f-0', 'feishu', { last_active: 10, title: 'updated' })]

    const result = reconcileMessagingRefresh(paged, seed, true)

    expect(result).toHaveLength(2)
    expect(result[0].title).toBe('updated')
    expect(result[1].id).toBe('f-1')
  })
})
