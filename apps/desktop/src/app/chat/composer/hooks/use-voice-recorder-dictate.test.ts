import { describe, expect, it } from 'vitest'

import { dictationPressIgnored } from './use-voice-recorder'

/**
 * The reported defect: dictate → speak → press to stop (transcription starts) → press again → the words
 * never appear. The second press re-entered `stop()` for the same dictation, and the two invocations
 * raced through `live()`: the later one nulls `dictationRef`, the earlier one then drops its finished
 * transcript before `onTranscript`.
 */
describe('dictationPressIgnored', () => {
  it('ignores a press while the previous clip is still transcribing', () => {
    expect(dictationPressIgnored('transcribing')).toBe(true)
  })

  it('lets a press start a recording from idle', () => {
    expect(dictationPressIgnored('idle')).toBe(false)
  })

  it('lets a press stop a running recording', () => {
    expect(dictationPressIgnored('recording')).toBe(false)
  })
})
