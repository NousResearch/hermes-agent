import { describe, expect, it } from 'vitest'

import { sourceFences } from './transcript-code-blocks'

describe('literal transcript fences', () => {
  it('preserves fence identity and distinguishes literal completion from Markdown repair', () => {
    expect(sourceFences('Before\r\n```my-form\r\nfirst\r\n```\r\n~~~other\r\nsecond')).toMatchObject([
      { language: 'my-form', code: 'first', closed: true, index: 0, startLine: 2 },
      { language: 'other', code: 'second', closed: false, index: 1, startLine: 5 }
    ])
  })

  it('does not treat a nested example or wrong/short closer as a complete program', () => {
    expect(sourceFences('````text\n```my-form\nexample\n```\n````')).toMatchObject([
      { language: 'text', code: '```my-form\nexample\n```', closed: true }
    ])
    expect(sourceFences('~~~~my-form\nexample\n```\n~~~')).toMatchObject([{ language: 'my-form', closed: false }])
    expect(sourceFences('```my`form\nexample')).toEqual([])
  })
})
