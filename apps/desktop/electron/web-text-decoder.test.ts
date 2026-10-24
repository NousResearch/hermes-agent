import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { decodeWebText } from './web-text-decoder'

describe('decodeWebText', () => {
  test('honours a quoted Big5 charset in the response header', () => {
    const bytes = new Uint8Array([
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

    assert.equal(decodeWebText(bytes, 'text/html; charset="big5"'), '<title>提示信息</title>')
  })

  test('sniffs a Shift-JIS meta charset before decoding the document', () => {
    const bytes = new Uint8Array([
      ...Buffer.from('<meta charset=shift_jis><title>'),
      0x93,
      0xfa,
      0x96,
      0x7b,
      0x8c,
      0xea,
      ...Buffer.from('</title>')
    ])

    assert.equal(decodeWebText(bytes), '<meta charset=shift_jis><title>日本語</title>')
  })

  test('sniffs charset from an http-equiv content attribute', () => {
    const bytes = new Uint8Array([
      ...Buffer.from('<meta http-equiv="Content-Type" content="text/html; charset=gbk"><title>'),
      0xd6,
      0xd0,
      0xce,
      0xc4,
      ...Buffer.from('</title>')
    ])

    assert.ok(decodeWebText(bytes).includes('<title>中文</title>'))
  })

  test('falls back to UTF-8 when a server declares an unknown charset', () => {
    const bytes = new TextEncoder().encode('<title>Résumé</title>')

    assert.equal(decodeWebText(bytes, 'text/html; charset=not-a-real-encoding'), '<title>Résumé</title>')
  })

  test('the HTTP charset takes precedence over a conflicting meta declaration', () => {
    const bytes = new Uint8Array([...Buffer.from('<meta charset=utf-8><title>'), 0xd6, 0xd0, 0xce, 0xc4])

    assert.equal(decodeWebText(bytes, 'text/html; charset=gbk'), '<meta charset=utf-8><title>中文')
  })
})
