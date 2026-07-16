import { fireEvent, render } from '@testing-library/react'
import { createElement } from 'react'
import { describe, expect, it } from 'vitest'

import { createReviewContext } from '@/store/annotations'

import {
  buildSvgSandboxDocument,
  clampVisualZoom,
  svgHasExternalResources,
  VisualAnnotationPreview
} from './visual-preview'

describe('visual SVG preview', () => {
  it('preserves SVG structure inside a scriptless, network-blocked sandbox', () => {
    const source = `
      <svg xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="paint"><stop offset="1" /></linearGradient></defs>
        <script>alert(1)</script>
        <text onclick="alert(2)">Selectable text</text>
        <image href="https://example.com/tracker.png" />
        <use href="#safe-shape" />
      </svg>
    `

    const result = buildSvgSandboxDocument(source)

    expect(result).toContain('Selectable text')
    expect(result).toContain('<use href="#safe-shape"')
    expect(result).toContain('<linearGradient')
    expect(result).toContain("default-src 'none'")
    expect(result).toContain('img-src data: blob:')
    expect(svgHasExternalResources(source)).toBe(true)
    expect(svgHasExternalResources('<svg><use href="#local" /></svg>')).toBe(false)
  })

  it('bounds image zoom without preventing useful close-up inspection', () => {
    expect(clampVisualZoom(0)).toBe(25)
    expect(clampVisualZoom(175)).toBe(175)
    expect(clampVisualZoom(800)).toBe(400)
  })

  it('keeps image zoom and pan controls at the start of the visual toolbar', () => {
    const rendered = render(
      createElement(VisualAnnotationPreview, {
        dataUrl: 'data:image/png;base64,',
        filePath: '/work/image.png',
        label: 'image.png',
        mediaKind: 'png',
        reviewContext: createReviewContext({ artifactPath: '/work/image.png', kind: 'document' })
      })
    )

    expect(rendered.getByRole('group', { name: 'Image preview zoom' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'Pan image' })).toBeTruthy()
    fireEvent.click(rendered.getByRole('button', { name: 'Increase image preview size' }))
    expect(rendered.getByRole('button', { name: 'Reset image preview size from 125%' })).toBeTruthy()
  })
})
