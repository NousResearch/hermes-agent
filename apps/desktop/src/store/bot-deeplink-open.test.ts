import { beforeEach, describe, expect, it } from 'vitest'

import { $pendingDeepLinkBot, requestBotChatFromDeepLink } from './bot-deeplink-open'

describe('requestBotChatFromDeepLink', () => {
  beforeEach(() => {
    $pendingDeepLinkBot.set(null)
  })
  it('parks a trimmed bot profile name for the consumer', () => {
    expect(requestBotChatFromDeepLink(' ops ')).toBe(true)
    expect($pendingDeepLinkBot.get()).toBe('ops')
  })

  it('rejects an empty or whitespace-only profile', () => {
    expect(requestBotChatFromDeepLink('')).toBe(false)
    expect(requestBotChatFromDeepLink('   ')).toBe(false)
    expect(requestBotChatFromDeepLink(undefined)).toBe(false)
    expect($pendingDeepLinkBot.get()).toBe(null)
  })
})
