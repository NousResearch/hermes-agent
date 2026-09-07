import { describe, expect, it } from 'vitest'

import { assistantTextPart, renderMediaTags } from './chat-messages/parts'
import { extractEmbeddedImages } from './embedded-images'
import { stripGeneratedImageEchoes } from './generated-images'
import { preprocessMarkdown } from './markdown-preprocess'
import { stripPreviewTargets } from './preview-targets'

describe('Markdown whitespace semantics', () => {
  it('preserves hard and soft breaks and fenced whitespace through display preprocessing', () => {
    for (const input of ['First line  \nSecond line', 'Soft first\nSoft second', '```python\nvalue = 1  \nvalue = 2 \n```']) {
      expect(preprocessMarkdown(input)).toBe(input)
    }
  })

  it('preserves prose hard breaks when media and preview extraction runs', () => {
    const prose = 'First line  \nSecond line\n\n```python\nvalue = 1  \nvalue = 2 \n\n\nvalue = 3\n```'
    const image = 'data:image/png;base64,' + 'A'.repeat(64)
    expect(stripPreviewTargets(prose + '\n\n[Preview: x](#preview:test)')).toContain(prose)
    expect(renderMediaTags(prose + '\n\nMEDIA: /tmp/image.png')).toContain(prose)
    expect(assistantTextPart(prose)).toMatchObject({ type: 'text', text: prose })
    expect(extractEmbeddedImages(prose + '\n\n' + image).cleanedText).toContain(prose)
    expect(stripGeneratedImageEchoes(prose + '\n\n![result](/tmp/image.png)', ['/tmp/image.png'])).toContain(prose)
  })
})
