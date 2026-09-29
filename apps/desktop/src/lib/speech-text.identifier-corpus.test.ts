import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

import { sanitizeTextForSpeech } from './speech-text'

// The shared corpus both speech normalizers must agree on (#119207): the
// identical file is loaded by tests/tools/test_tts_identifier_speech.py, so
// the desktop (speech-text.ts) and gateway/CLI (tts_text_normalize.py)
// halves stay in lockstep. Policy: identifier-dense tokens are silence,
// never hardcoded English placeholder words (#86602); ordinary prose,
// emails, dates and ratios pass verbatim.
const corpusPath = resolve(__dirname, '../../../../tests/fixtures/identifier_speech_corpus.json')
const corpus = JSON.parse(readFileSync(corpusPath, 'utf8')) as {
  identifier_tokens: [string, string[], string[]][]
  pass_through_tokens: [string, string[], string[]][]
}

describe('sanitizeTextForSpeech identifier-dense tokens (#119207)', () => {
  it('silences identifier tokens while the prose around them survives', () => {
    for (const [text, mustNotContain, mustContain] of corpus.identifier_tokens) {
      const spoken = sanitizeTextForSpeech(text)
      for (const needle of mustNotContain) {
        expect(spoken, `${needle} leaked from ${text}`).not.toContain(needle)
      }
      for (const needle of mustContain) {
        expect(spoken, `${needle} lost from ${text}`).toContain(needle)
      }
    }
  })

  it('passes ordinary speech through untouched', () => {
    for (const [text, , mustContain] of corpus.pass_through_tokens) {
      const spoken = sanitizeTextForSpeech(text)
      for (const needle of mustContain) {
        expect(spoken, `${needle} lost from ${text}`).toContain(needle)
      }
    }
  })

  it('reads the issue repro: filenames never spell out character by character', () => {
    const spoken = sanitizeTextForSpeech(
      'Saved peyton-sample-20260922.wav and peyton-sample-20260922.ogg.'
    )
    expect(spoken).not.toContain('.wav')
    expect(spoken).not.toContain('.ogg')
    expect(spoken).not.toContain('peyton')
    expect(spoken).toContain('Saved')
  })

  it('keeps code fences, links and MEDIA: tokens working alongside the new pass', () => {
    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe(
      'Here is code. Done.'
    )
    expect(sanitizeTextForSpeech('Use `git status` after the change.')).toBe(
      'Use git status after the change.'
    )
  })
})
