import { describe, expect, it } from 'vitest'

import { previewKindFor } from './preview-kind'

describe('previewKindFor', () => {
  it.each(['.3mf', '.glb', '.gltf', '.obj', '.ply', '.stl'])('treats %s as a 3D model even though it sniffs as binary', ext => {
    expect(previewKindFor(ext, 'application/octet-stream', true)).toBe('model3d')
  })

  it('keeps existing kinds ahead of the binary/text fallback', () => {
    expect(previewKindFor('.html', 'text/html', false)).toBe('html')
    expect(previewKindFor('.png', 'image/png', true)).toBe('image')
    expect(previewKindFor('.pdf', 'application/pdf', true)).toBe('pdf')
    expect(previewKindFor('.bin', 'application/octet-stream', true)).toBe('binary')
    expect(previewKindFor('.ts', 'text/plain', false)).toBe('text')
  })
})
