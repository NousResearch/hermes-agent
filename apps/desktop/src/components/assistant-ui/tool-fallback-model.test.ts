import { describe, expect, it } from 'vitest'

import { buildToolView, type ToolPart } from './tool-fallback-model'

const part = (overrides: Partial<ToolPart>): ToolPart => ({
  args: {},
  isError: false,
  result: {},
  toolCallId: 'call_1',
  toolName: 'vision_analyze',
  type: 'tool-call',
  ...overrides
})

describe('buildToolView image handling', () => {
  // vision_analyze reports the input image as a local path; an <img> pointed at
  // a bare path resolves against the renderer origin and 404s, so we render the
  // tool codicon instead of a broken image.
  it('drops bare filesystem paths', () => {
    expect(buildToolView(part({ args: { path: '/Users/me/shot.png' } }), '').imageUrl).toBe('')
    expect(buildToolView(part({ result: { image_path: '/tmp/out.jpg' } }), '').imageUrl).toBe('')
  })

  it('keeps fetchable data URLs', () => {
    const dataUrl = 'data:image/png;base64,AAAA'

    expect(buildToolView(part({ result: { image_url: dataUrl } }), '').imageUrl).toBe(dataUrl)
  })

  it('keeps remote http(s) image URLs', () => {
    const url = 'https://example.com/pic.webp'

    expect(buildToolView(part({ result: { url } }), '').imageUrl).toBe(url)
  })
})
