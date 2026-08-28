import { describe, expect, it } from 'vitest'

import { CANONICAL_BOT_CHAT_TITLE, canonicalSessionTabCaption, isCanonicalBotChatTitle } from './canonical-tab-title'

describe('isCanonicalBotChatTitle', () => {
  it('treats root_title Bot Chat as canonical even when the listing title drifted', () => {
    expect(isCanonicalBotChatTitle('hey there', 'Bot Chat')).toBe(true)
  })

  it('treats stored title Bot Chat as canonical when root_title is absent', () => {
    expect(isCanonicalBotChatTitle('Bot Chat', undefined)).toBe(true)
    expect(isCanonicalBotChatTitle('Bot Chat', '')).toBe(true)
  })

  it('does not treat a preview-shaped title as canonical', () => {
    expect(isCanonicalBotChatTitle('hey there', undefined)).toBe(false)
    expect(isCanonicalBotChatTitle('', 'Daily notes')).toBe(false)
  })
})

describe('canonicalSessionTabCaption', () => {
  it('keeps Bot Chat on canonical re-bind instead of a preview-derived caption', () => {
    expect(
      canonicalSessionTabCaption({
        preview: 'what is the weather in oslo',
        rootTitle: 'Bot Chat',
        title: '',
        untitledFallback: 'New session'
      })
    ).toBe(CANONICAL_BOT_CHAT_TITLE)

    expect(
      canonicalSessionTabCaption({
        preview: 'what is the weather in oslo',
        title: 'Bot Chat'
      })
    ).toBe(CANONICAL_BOT_CHAT_TITLE)

    expect(
      canonicalSessionTabCaption({
        preview: 'what is the weather in oslo',
        title: '',
        workspaceTabTitle: 'Bot Chat'
      })
    ).toBe(CANONICAL_BOT_CHAT_TITLE)
  })

  it('keeps preview captions for non-canonical sessions', () => {
    expect(
      canonicalSessionTabCaption({
        preview: 'what is the weather in oslo',
        title: '',
        untitledFallback: 'New session'
      })
    ).toBe('what is the weather in oslo')

    expect(
      canonicalSessionTabCaption({
        preview: 'ignored because a real title exists',
        title: 'Daily notes'
      })
    ).toBe('Daily notes')
  })
})
