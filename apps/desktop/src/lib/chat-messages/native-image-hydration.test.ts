import { describe, expect, it } from 'vitest'

import { chatMessageText, toChatMessages } from '@/lib/chat-messages'

const image = { type: 'image_url', image_url: { url: `data:image/png;base64,${'A'.repeat(64)}` } }

describe('structured native user images', () => {
  it.each([false, true])('normalizes native image content (placeholders: %s)', placeholders => {
    const caption = 'Explain [Image attached at: /literal.png] in this sentence.'
    const hints = `${caption}\n\n[Image attached at: /images/a [1].png]\n[Image attached: /images/b.png]`

    const content = [
      { type: 'text', text: placeholders ? `${hints}\n[screenshot]\n[screenshot]` : hints },
      image,
      image
    ]

    const original = JSON.stringify(content)
    const [structured] = toChatMessages([{ role: 'user', row_id: 153, content, timestamp: 103 }])

    const [flattened] = toChatMessages([
      { role: 'user', row_id: 153, content: `${hints}\n[screenshot]\n[screenshot]`, timestamp: 103 }
    ])

    expect(chatMessageText(structured)).toBe(caption)
    expect(structured.attachmentRefs).toEqual(flattened.attachmentRefs)
    expect(structured.attachmentRefs).toEqual(['@image:`/images/a [1].png`', '@image:/images/b.png'])
    expect(structured.rowId).toBe(153)
    expect(JSON.stringify(content)).toBe(original)
  })

  it('retains an image-only native user turn', () => {
    const [message] = toChatMessages([
      {
        role: 'user',
        content: [{ type: 'text', text: '\n\n[Image attached at: /images/only.png]' }, image],
        timestamp: 1
      }
    ])

    expect(chatMessageText(message)).toBe('')
    expect(message.attachmentRefs).toEqual(['@image:/images/only.png'])
  })

  it('does not treat literal or mismatched path hints as a native image suffix', () => {
    const literal = 'Quoted text\n\n[Image attached at: /literal.png]'

    const [prose, mismatch] = toChatMessages([
      { role: 'user', content: [{ type: 'text', text: literal }], timestamp: 1 },
      { role: 'user', content: [{ type: 'text', text: literal }, image, image], timestamp: 2 }
    ])

    expect(chatMessageText(prose)).toBe(literal)
    expect(prose.attachmentRefs).toBeUndefined()
    expect(chatMessageText(mismatch)).toContain(literal)
    expect(mismatch.attachmentRefs).toBeUndefined()
  })
})
