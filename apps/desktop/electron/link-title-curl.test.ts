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
      html: '<title>提示信息</title>',
      redirectUrl: '',
      httpCode: 0
    })
  })

  test('reads the trailer from the retained tail after the body budget is exhausted', () => {
    assert.deepEqual(parseCurlTitleResponse(BIG5_TITLE, TRAILER), {
      effectiveUrl: 'https://example.test/final',
      html: '<title>提示信息</title>',
      redirectUrl: '',
      httpCode: 0
    })
  })

  test('reads the redirect target and HTTP status of a 3xx hop', () => {
    const body = Buffer.concat([
      Buffer.from('<title>Moved</title>'),
      Buffer.from(
        '\nhermes-content-type:text/html\nhermes-url-effective:http://origin.test/a\n' +
          'hermes-url-redirect:https://origin.test/b\nhermes-http-code:302'
      )
    ])

    assert.deepEqual(parseCurlTitleResponse(body, Buffer.alloc(0)), {
      effectiveUrl: 'http://origin.test/a',
      html: '<title>Moved</title>',
      redirectUrl: 'https://origin.test/b',
      httpCode: 302
    })
  })

  test('keeps an unparseable HTTP status at 0 without failing the parse', () => {
    const body = Buffer.concat([
      Buffer.from('<title>ok</title>'),
      Buffer.from(
        '\nhermes-content-type:text/html\nhermes-url-effective:https://a.test/\nhermes-http-code:not-a-number'
      )
    ])

    assert.equal(parseCurlTitleResponse(body, Buffer.alloc(0)).httpCode, 0)
  })
})
