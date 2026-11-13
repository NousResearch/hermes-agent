import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import type { SessionInfo } from '@/hermes'

import { SessionOriginBadge } from './session-origin-badge'

function makeSession(overrides: Partial<SessionInfo> & { title: string }): SessionInfo {
  return {
    id: 's1',
    last_active: 0,
    profile: 'default',
    started_at: 0,
    ...overrides
  } as unknown as SessionInfo
}

// A messaging row's title is the conversation's topic, never WHO is on the
// other end — several people's DMs read identically without the sender's name.
describe('SessionOriginBadge', () => {
  const badge = (container: HTMLElement) => container.querySelector('[data-origin-name]')

  it('paints the sender name of a messaging session', () => {
    const origin = JSON.stringify({ platform: 'dingtalk', chat_type: 'dm', user_name: '郑裕川' })
    const { container } = render(<SessionOriginBadge session={makeSession({ origin_json: origin, source: 'dingtalk', title: '权限修改记录' })} />)

    expect(badge(container)?.textContent).toBe('郑裕川')
  })

  it('paints nothing for a local session and for a messaging row without a name', () => {
    const local = render(<SessionOriginBadge session={makeSession({ title: 'Local chat' })} />)

    expect(badge(local.container)).toBeNull()

    const unnamed = render(<SessionOriginBadge session={makeSession({ source: 'dingtalk', title: 'No name' })} />)

    expect(badge(unnamed.container)).toBeNull()
  })
})
