import { expect, it } from 'vitest'

import { looksBinary } from './file-preview-content'

it('keeps UTF-8 Korean and normal whitespace available as text previews', () => {
  expect(looksBinary(Buffer.from('한글 메모\n둘째 줄\t끝\r\n', 'utf8'))).toBe(false)
  expect(looksBinary(new Uint8Array())).toBe(false)
})

it('blocks NUL-containing and control-heavy data from text preview', () => {
  expect(looksBinary(Buffer.from('text\0hidden'))).toBe(true)
  expect(looksBinary(new Uint8Array([1, 2, 3, 65, 66]))).toBe(true)
})
