/**
 * #91359 — recognized group @mentions render as semantic inline references;
 * everything else stays prose. The classifier is the round loop's own parser,
 * so a token is styled exactly when routing would honour it.
 */

import { render, screen } from '@testing-library/react'
import { useMemo } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'

import { GROUP_CHAT_LOCALES } from './group-chat-locales'
import { classifyGroupMention, groupMentionComponents, groupMentionText, renderGroupMentionText } from './group-mention-text'
import { useBots } from './i18n'
import { translateBotsIn } from './i18n-test-helper'
import type { GroupMember } from './types'

const selectedLocale = vi.hoisted(() => ({ value: 'en' }))

vi.mock('@hermes/plugin-sdk', async () => {
  const { atom } = await import('nanostores')

  return {
    atom,
    usePluginI18n: () => translateBotsIn(selectedLocale.value),
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
    const html = renderToStaticMarkup(<>{renderGroupMentionText(text, members, GROUP_CHAT_LOCALES.en)}</>)

    expect(html).toContain('data-ref="agent"')
    expect(html).toContain('data-ref="human"')
    expect(html).toContain('data-ref="broadcast"')
    // `@example.com` is not a room identity — no span around it.
    expect(html).not.toMatch(/data-ref="[^"]*"[^>]*>@example/)
    expect(html.replace(/<[^>]+>/g, '')).toBe(text)
  })
})

function TranslatedMentions() {
  const b = useBots()
  const decorate = useMemo(() => groupMentionText(members, b.group), [b.group])
  const { p: Paragraph } = useMemo(() => groupMentionComponents(members, b.group), [b.group])

  return <>
    <div>{decorate('@planner @everyone @user')}</div>
    <Paragraph>@planner @everyone @user</Paragraph>
  </>
}

it('updates mention tooltips in both transcript renderers when the locale changes', () => {
  const view = render(<TranslatedMentions />)

  expect(screen.getAllByTitle(GROUP_CHAT_LOCALES.en.mentionBot)).toHaveLength(2)
  selectedLocale.value = 'ja'
  view.rerender(<TranslatedMentions />)

  for (const title of [GROUP_CHAT_LOCALES.ja.mentionBot, GROUP_CHAT_LOCALES.ja.mentionEveryone, GROUP_CHAT_LOCALES.ja.mentionUser]) {
    expect(screen.getAllByTitle(title)).toHaveLength(2)
  }

  expect(screen.queryByTitle(GROUP_CHAT_LOCALES.en.mentionBot)).toBeNull()
})
