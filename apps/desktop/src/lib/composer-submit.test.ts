import { describe, expect, it } from 'vitest'

import {
  bubbleAttachmentRefsForRow,
  collectInlineOsImageAttachments,
  normalizeInlineRefWireForm,
  stripInlineImageRefs
} from './composer-submit'

describe('normalizeInlineRefWireForm', () => {
  it('quotes unquoted paths after optional whitespace after the colon', () => {
    expect(normalizeInlineRefWireForm('@file: Desktop/sage/xhs_covers 封面图')).toBe(
      '@file:`Desktop/sage/xhs_covers` 封面图'
    )
  })
})

describe('collectInlineOsImageAttachments', () => {
  it('collects absolute OS image paths from inline refs', () => {
    const path = '/home/user/tmp/wechat_sim/photo.jpg'
    const text = `use this @image:` + path
    const extra = collectInlineOsImageAttachments(text, [])

    expect(extra).toHaveLength(1)
    expect(extra[0].kind).toBe('image')
    expect(extra[0].path).toBe(path)
  })

  it('skips paths already represented as attachment pills', () => {
    const path = '/home/user/photo.jpg'
    const existing = [{ id: 'image:x', kind: 'image' as const, label: 'photo.jpg', path }]

    expect(collectInlineOsImageAttachments(`see @image:${path}`, existing)).toEqual([])
  })
})

describe('stripInlineImageRefs', () => {
  it('removes inline refs for attached image paths', () => {
    const path = '/home/user/photo.jpg'
    const stripped = stripInlineImageRefs(`look @image:` + path, new Set([path]))

    expect(stripped).toBe('look')
  })
})

describe('bubbleAttachmentRefsForRow', () => {
  it('drops refs already present in bubble body text', () => {
    const refs = ['@folder:/home/user/xhs_covers', 'data:image/jpeg;base64,abc']
    const body = '@folder:/home/user/xhs_covers\n\nhello'

    expect(bubbleAttachmentRefsForRow(refs, body)).toEqual(['data:image/jpeg;base64,abc'])
  })
})
