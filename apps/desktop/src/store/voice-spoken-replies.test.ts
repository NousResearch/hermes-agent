import { afterEach, expect, it } from 'vitest'

import { $gptLiveSpokenReplies, applyGptLiveSpokenRepliesFromConfig } from './voice-prefs'

afterEach(() => $gptLiveSpokenReplies.set('collapse'))

it('reads voice.gpt_live.spoken_replies, collapsing by default', () => {
  applyGptLiveSpokenRepliesFromConfig({ voice: { gpt_live: { spoken_replies: 'hide' } } })
  expect($gptLiveSpokenReplies.get()).toBe('hide')
  applyGptLiveSpokenRepliesFromConfig({ voice: { gpt_live: { spoken_replies: 'something-else' } } })
  expect($gptLiveSpokenReplies.get()).toBe('collapse')
  applyGptLiveSpokenRepliesFromConfig({})
  expect($gptLiveSpokenReplies.get()).toBe('collapse')
})
