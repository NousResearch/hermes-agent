import { describe, expect, it } from 'vitest'

import { isTtsEcho, sequenceMatcherRatio } from './voice-tts-echo'

// Mirrors tests/tools/test_voice_tts_echo_guard.py::TestIsTtsEcho so the
// desktop guard and the CLI guard agree on what counts as echo.
describe('isTtsEcho', () => {
  it('flags a near-verbatim repeat', () => {
    const spoken =
      '맞아요. 사용자가 마이크를 끄는 게 아니라 앱이 제 음성은 에코 제거로 걸러내고 실제 사용자 음성만 끼어들기로 받아야 해요.'

    expect(isTtsEcho(spoken, spoken)).toBe(true)
  })

  it('flags a repeat with a leading stutter', () => {
    expect(
      isTtsEcho('네 방금 네 방금도 제 답변이 그대로 다시 입력됐어요.', '네, 방금도 제 답변이 그대로 다시 입력됐어요.')
    ).toBe(true)
  })

  it('does not flag an unrelated interjection', () => {
    expect(
      isTtsEcho(
        'actually can you also check my calendar for tomorrow',
        'The weather today is sunny with a light breeze from the west.'
      )
    ).toBe(false)
  })

  it('does not flag a short unrelated reply', () => {
    expect(isTtsEcho('stop', "I've finished summarizing the document you shared earlier.")).toBe(false)
  })

  it('flags a short fragment of a longer multi-sentence reply', () => {
    const spoken =
      "Sure, here's a summary of what we found. The build failed because of a missing dependency in the " +
      "lockfile. I've already gone ahead and regenerated it, and the tests are passing again locally. Let " +
      "me know if you'd like me to open a PR for this or if you want to review the diff first before I do " +
      'anything else.'

    expect(isTtsEcho("Sure, here's a summary of what we found.", spoken)).toBe(true)
  })

  it('flags a fragment from the middle of a reply', () => {
    const spoken =
      'The deployment finished successfully. All three services came up healthy, and the smoke tests passed without any errors.'

    expect(isTtsEcho('the smoke tests passed without any errors', spoken)).toBe(true)
  })

  it('does not flag a short genuine acknowledgement that appears inside the reply', () => {
    expect(isTtsEcho('yes', 'Yes, I can help with that -- let me pull up the details for you.')).toBe(false)
  })

  it('flags a fragment in a no-whitespace language', () => {
    const spoken = '部署已经成功完成。所有三个服务都正常运行状态良好,冒烟测试也全部通过,没有发现任何错误。'

    expect(isTtsEcho('所有三个服务都正常运行状态良好', spoken)).toBe(true)
  })

  it('treats empty inputs as not echo', () => {
    expect(isTtsEcho('', 'hello')).toBe(false)
    expect(isTtsEcho('hello', '')).toBe(false)
    expect(isTtsEcho('', '')).toBe(false)
  })

  it('is case and whitespace insensitive', () => {
    expect(isTtsEcho('  SURE,   I can help with that   right away.  ', 'Sure, I can help with that right away.')).toBe(
      true
    )
  })

  it('honors a custom threshold', () => {
    const spoken = 'This is a moderately similar sentence about testing.'
    const transcript = 'This is a rather different sentence about coding.'

    expect(isTtsEcho(transcript, spoken, 0.5)).toBe(true)
    expect(isTtsEcho(transcript, spoken, 0.95)).toBe(false)
  })
})

describe('sequenceMatcherRatio', () => {
  // Reference values from CPython difflib.SequenceMatcher(None, a, b).ratio().
  it.each([
    ['abcd', 'bcde', 0.75],
    [
      'This is a rather different sentence about coding.',
      'This is a moderately similar sentence about testing.',
      0.693069306930693
    ],
    ['', '', 1]
  ])('%j vs %j', (a, b, expected) => {
    expect(sequenceMatcherRatio(Array.from(a), Array.from(b))).toBeCloseTo(expected, 12)
  })

  it('applies autojunk to a 200+ element b like CPython', () => {
    const a = Array.from('the smoke tests passed without any errors')
    const b = Array.from('The deployment finished successfully. '.repeat(6).toLowerCase())

    expect(sequenceMatcherRatio(a, b)).toBeCloseTo(0.02973977695167286, 12)
  })
})
