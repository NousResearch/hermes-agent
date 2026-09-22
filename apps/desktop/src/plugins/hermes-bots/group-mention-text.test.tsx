/**
 * #91359 — recognized group @mentions render as semantic inline references;
 * everything else stays prose. The classifier is the round loop's own parser,
 * so a token is styled exactly when routing would honour it.
 */

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { classifyGroupMention, groupMentionText, renderGroupMentionText } from './group-mention-text'
import type { GroupMember } from './types'

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    host: {
      request: vi.fn(),
      state: { connectionId: { get: () => 'local' }, profile: { get: () => 'default' } }
    },
    queryClient: { invalidateQueries: vi.fn() },
    useQuery: vi.fn(),
    useValue: vi.fn()
  }
})

vi.mock('./shared', () => ({ getPluginCtx: () => null, ID: 'hermes-bots' }))

const members = [
  { name: 'planner', handle: 'planner' },
  { name: 'reviewer-mac-mini', handle: 'reviewer-mac-mini' }
] as GroupMember[]

describe('group mention rendering', () => {
  it('classifies seated bots, the human and broadcasts, and leaves unknown tokens plain', () => {
    expect(classifyGroupMention('planner', members)).toBe('agent')
    expect(classifyGroupMention('reviewer-mac-mini', members)).toBe('agent')
    expect(classifyGroupMention('user', members)).toBe('human')
    expect(classifyGroupMention('everyone', members)).toBe('broadcast')
    expect(classifyGroupMention('all', members)).toBe('broadcast')
    expect(classifyGroupMention('stranger', members)).toBeNull()
  })

  it('wraps only recognized mentions and keeps the visible text byte-identical', () => {
    const text = '@planner compare the options, then @user picks; mail ops@example.com — @everyone review'
    const html = renderToStaticMarkup(<>{renderGroupMentionText(text, members)}</>)

    expect(html).toContain('<span class="ref font-medium" data-ref="agent"')
    expect(html).toContain('data-ref="human"')
    expect(html).toContain('data-ref="broadcast"')
    // `@example.com` is not a room identity — no span around it.
    expect(html).not.toContain('data-ref="agent" title="Bot in this room">@example')
    expect(html.replace(/<[^>]+>/g, '')).toBe(text)
  })

  it('colours a mention nested in emphasis (bot markdown **@name**)', () => {
    const html = renderToStaticMarkup(
      <p>{groupMentionText(members)(['Please ', <strong key="s">@planner</strong>, ' take this'])}</p>
    )

    expect(html).toContain('<span class="ref font-medium" data-ref="agent"')
    expect(html).toContain('>@planner</span>')
    expect(html.replace(/<[^>]+>/g, '')).toBe('Please @planner take this')
  })

  it('leaves a mention inside inline code as prose', () => {
    const html = renderToStaticMarkup(<p>{groupMentionText(members)(['run ', <code key="c">@planner</code>])}</p>)

    expect(html).not.toContain('data-ref')
    expect(html).toContain('@planner')
  })
})
