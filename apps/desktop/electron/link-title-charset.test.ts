import assert from 'node:assert/strict'

import { test } from 'vitest'

import { decodeHttpBody } from './link-title-charset'

// Big5 bytes for "中文" (0xA4 0xA4 = 中, 0xA4 0xE5 = 文). Decoded as UTF-8 these
// four bytes are not valid continuation sequences and collapse entirely to
// U+FFFD — the exact mojibake shape reported for non-UTF-8 link-card titles.
const BIG5_ZHONGWEN = Buffer.from([0xa4, 0xa4, 0xa4, 0xe5])

test('a charset declared in Content-Type decodes a non-UTF-8 body correctly', () => {
  const html = Buffer.concat([Buffer.from('<title>', 'ascii'), BIG5_ZHONGWEN, Buffer.from('</title>', 'ascii')])

  assert.equal(decodeHttpBody(html, 'text/html; charset=big5'), '<title>中文</title>')
})

test('with no declared charset, a <meta charset> in the body is sniffed', () => {
  const html = Buffer.concat([
    Buffer.from('<meta charset="big5"><title>', 'ascii'),
    BIG5_ZHONGWEN,
    Buffer.from('</title>', 'ascii')
  ])

  assert.equal(decodeHttpBody(html, ''), '<meta charset="big5"><title>中文</title>')
})

test('an unrecognised charset label falls back to the UTF-8 decode instead of throwing', () => {
  const html = Buffer.from('<title>hello</title>', 'ascii')

  assert.equal(decodeHttpBody(html, 'text/html; charset=not-a-real-charset'), '<title>hello</title>')
})

test('a plain UTF-8 body with no declared charset is unaffected', () => {
  const html = Buffer.from('<title>Straße — café</title>', 'utf8')

  assert.equal(decodeHttpBody(html, 'text/html'), '<title>Straße — café</title>')
})
