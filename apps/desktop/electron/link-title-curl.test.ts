import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { parseCurlTitleResponse } from './link-title-curl'

const BIG5_TITLE = Buffer.from([
  ...Buffer.from('<title>'),
  0xb4,
  0xa3,
  0xa5,
  0xdc,
  0xab,
  0x48,
  0xae,
  0xa7,
  ...Buffer.from('</title>')
])

const TRAILER = Buffer.from(
  '\nhermes-content-type:text/html; charset=big5\nhermes-url-effective:https://example.test/final'
)

describe('parseCurlTitleResponse', () => {
  test('decodes a legacy page from curl content-type metadata', () => {
    assert.deepEqual(parseCurlTitleResponse(Buffer.concat([BIG5_TITLE, TRAILER]), Buffer.alloc(0)), {
      effectiveUrl: 'https://example.test/final',
      html: '<title>提示信息</title>'
    })
  })

  test('reads the trailer from the retained tail after the body budget is exhausted', () => {
    assert.deepEqual(parseCurlTitleResponse(BIG5_TITLE, TRAILER), {
      effectiveUrl: 'https://example.test/final',
      html: '<title>提示信息</title>'
    })
  })
})
