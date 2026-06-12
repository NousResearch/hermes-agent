import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { isMessagingSource, logicalSessionSource } from './session-source'

const session = (
  over: Partial<Pick<SessionInfo, 'handoff_platform' | 'handoff_state' | 'source'>> = {}
): Pick<SessionInfo, 'handoff_platform' | 'handoff_state' | 'source'> => ({
  handoff_platform: null,
  handoff_state: null,
  source: null,
  ...over
})

describe('logicalSessionSource', () => {
  it('uses the completed handoff platform for restored messaging sessions', () => {
    expect(
      logicalSessionSource(
        session({
          handoff_platform: 'weixin',
          handoff_state: 'completed',
          source: 'desktop'
        })
      )
    ).toBe('weixin')
  })

  it('ignores local handoff origins and falls back to the live source', () => {
    expect(
      logicalSessionSource(
        session({
          handoff_platform: 'desktop',
          handoff_state: 'completed',
          source: 'desktop'
        })
      )
    ).toBe('desktop')
  })

  it('treats completed handoff origins as messaging sources', () => {
    expect(
      isMessagingSource(
        logicalSessionSource(
          session({
            handoff_platform: 'weixin',
            handoff_state: 'completed',
            source: 'desktop'
          })
        )
      )
    ).toBe(true)
  })
})
