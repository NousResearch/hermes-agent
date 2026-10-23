import { describe, expect, it } from 'vitest'

import { assistantTextPart, renderMediaTags } from './chat-messages/parts'
import { extractEmbeddedImages, extractImageRefs } from './embedded-images'
import { stripGeneratedImageEchoes } from './generated-images'
import { preprocessMarkdown } from './markdown-preprocess'
import { stripPreviewTargets } from './preview-targets'

const samples = [
  'First line  \nSecond line',
  'Soft first\nSoft second',
  '    value = 1\n\n',
  '```python\nvalue = 1  ',
  '```python\n\nvalue = 1  \nvalue = 2 \n\n\n```'
]

describe('Markdown whitespace semantics', () => {
  it('preserves significant whitespace through display preprocessing', () => {
    for (const input of samples) {
      expect(preprocessMarkdown(input)).toBe(input)
      expect(preprocessMarkdown('\n\n' + input)).toBe('\n\n' + input)
    }
  })

  it('preserves text outside removed media and preview spans', () => {
    const image = 'data:image/png;base64,' + 'A'.repeat(64)

    for (const input of samples) {
      expect(assistantTextPart(input)).toMatchObject({ type: 'text', text: input })
      expect(renderMediaTags(input)).toBe(input)
      // Prefix attachments so the unfinished fence remains the document end.
      expect(stripPreviewTargets('[Preview: x](#preview:test)' + input)).toBe(input)
      expect(extractEmbeddedImages(image + '\n' + input).cleanedText).toBe('\n' + input)
      expect(stripGeneratedImageEchoes('![result](/tmp/image.png)\n' + input, ['/tmp/image.png'])).toBe('\n' + input)
      expect(extractImageRefs('@image:/tmp/image.png\n' + input).cleanedText).toBe(input)
      expect(extractEmbeddedImages(input + '\n\n' + image).cleanedText).toBe(input + '\n\n')
      expect(stripGeneratedImageEchoes(input + '\n\n![result](/tmp/image.png)', ['/tmp/image.png'])).toBe(
        input + '\n\n'
      )
    }
  })
})
