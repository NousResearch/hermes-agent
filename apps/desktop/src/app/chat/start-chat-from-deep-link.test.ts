import { beforeEach, describe, expect, it } from 'vitest'

import { $freshDraftKey, rotateFreshDraftKey, takeSessionDraft } from '@/store/composer'
import { $freshSessionRequest, $newChatProfile } from '@/store/profile'

import { onComposerInsertRequest } from './composer/focus'
import { startChatFromDeepLink } from './start-chat-from-deep-link'

/**
 * `hermes://chat/new` lands here. The contract worth pinning is the one a web
 * page could otherwise abuse: it opens a chat and SEEDS the composer, and it
 * never submits.
 */
describe('startChatFromDeepLink', () => {
  beforeEach(() => {
    $newChatProfile.set('stale-profile')
  })

  it('requests a fresh chat on the live profile', () => {
    const before = $freshSessionRequest.get()

    startChatFromDeepLink('')

    expect($freshSessionRequest.get()).toBe(before + 1)
    // A link must not inherit a stale per-profile quick-create selection.
    expect($newChatProfile.get()).toBeNull()
    rotateFreshDraftKey()
  })

  it("seeds the prompt into the new chat's own draft, not the chat on screen", () => {
    const outgoing = $freshDraftKey.get()

    startChatFromDeepLink('hermes migrate grok-bot')
    // Stands in for the fresh-session effect, which rotates the key.
    const fresh = rotateFreshDraftKey()

    expect(takeSessionDraft(fresh).text).toBe('hermes migrate grok-bot')
    expect(takeSessionDraft(outgoing).text).toBe('')
    // Consumed once: a later unrelated new chat starts empty.
    expect(takeSessionDraft(rotateFreshDraftKey()).text).toBe('')
  })

  it('never submits and never paints through the insert bus', async () => {
    const inserts: unknown[] = []
    const submits: unknown[] = []
    const offInsert = onComposerInsertRequest(detail => inserts.push(detail))
    // A submit on this bus would start a turn the user never asked for.
    const onSubmit = (event: Event) => submits.push(event)
    window.addEventListener('hermes:composer-submit', onSubmit)

    startChatFromDeepLink('hermes migrate grok-bot')
    rotateFreshDraftKey()
    // Let any deferred bus dispatch land before asserting nothing did.
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(submits).toEqual([])
    expect(inserts).toEqual([])

    offInsert()
    window.removeEventListener('hermes:composer-submit', onSubmit)
  })

  it('leaves the new chat empty when the link has no prompt', () => {
    startChatFromDeepLink('')

    expect(takeSessionDraft(rotateFreshDraftKey()).text).toBe('')
  })
})
