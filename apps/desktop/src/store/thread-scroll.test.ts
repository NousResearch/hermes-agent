import { describe, expect, it } from 'vitest'

import { DEFAULT_THREAD_JUMP_STATE, latestAnswerJumpState } from './thread-scroll'

describe('latestAnswerJumpState', () => {
  it('hides the jump pill for short answers', () => {
    expect(
      latestAnswerJumpState({
        answerHeight: 300,
        answerTop: 100,
        clientHeight: 700,
        scrollTop: 700
      })
    ).toEqual(DEFAULT_THREAD_JUMP_STATE)
  })

  it('jumps up to the answer start when parked near the bottom of a long answer', () => {
    expect(
      latestAnswerJumpState({
        answerHeight: 1800,
        answerTop: 240,
        clientHeight: 700,
        scrollTop: 1340
      })
    ).toEqual({ target: 'answer-start', visible: true })
  })

  it('jumps down to the answer end when already near the answer start', () => {
    expect(
      latestAnswerJumpState({
        answerHeight: 1800,
        answerTop: 240,
        clientHeight: 700,
        scrollTop: 300
      })
    ).toEqual({ target: 'answer-end', visible: true })
  })

  it('treats invalid geometry as hidden instead of producing a broken target', () => {
    expect(
      latestAnswerJumpState({
        answerHeight: Number.NaN,
        answerTop: 240,
        clientHeight: 700,
        scrollTop: 300
      })
    ).toEqual(DEFAULT_THREAD_JUMP_STATE)
  })
})
