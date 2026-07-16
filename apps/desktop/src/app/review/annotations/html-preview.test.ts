import { fireEvent, render } from '@testing-library/react'
import { createElement } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { clampHtmlZoom, HtmlZoomControls, sanitizeHtmlAnnotationDocument } from './html-preview'

describe('HTML annotation document sanitizer', () => {
  it('removes active content and event handlers while injecting the bridge', () => {
    const result = sanitizeHtmlAnnotationDocument(`
      <html><head><script>window.evil = true</script></head>
      <head><meta http-equiv="refresh" content="0; url=https://example.com"></head>
      <body onload="evil()">
        <a href="https://example.com">External</a>
        <svg><a xlink:href="javascript:evil()"><text>SVG link</text></a></svg>
        <p>Hello</p>
      </body></html>
    `)

    expect(result).not.toContain('window.evil')
    expect(result).not.toContain('onload=')
    expect(result).not.toContain('javascript:evil')
    expect(result).not.toContain('http-equiv="refresh"')
    expect(result).not.toContain('href="https://example.com"')
    expect(result).not.toContain('xlink:href')
    expect(result).toContain('hermes:annotation:selection')
    expect(result).toContain('Content-Security-Policy')
    expect(result).toContain('hermes:annotation:zoom')
    expect(result).toContain('hermes:annotation:pan')
    expect(result).toContain('startNodeOffset')
    expect(result).toContain('startPath')
    expect(result).toContain('name="viewport"')
  })

  it('offers compact bounded zoom controls with a one-click reset', () => {
    const onPanChange = vi.fn()
    const onZoomChange = vi.fn()
    const rendered = render(createElement(HtmlZoomControls, { onPanChange, onZoomChange, zoom: 100 }))

    fireEvent.click(rendered.getByRole('button', { name: 'Decrease HTML preview size' }))
    fireEvent.click(rendered.getByRole('button', { name: 'Increase HTML preview size' }))
    fireEvent.click(rendered.getByRole('button', { name: 'Reset HTML preview size from 100%' }))
    fireEvent.click(rendered.getByRole('button', { name: 'Drag HTML preview to pan' }))

    expect(onZoomChange).toHaveBeenNthCalledWith(1, 90)
    expect(onZoomChange).toHaveBeenNthCalledWith(2, 110)
    expect(onZoomChange).toHaveBeenNthCalledWith(3, 100)
    expect(onPanChange).toHaveBeenCalledWith(true)
    expect(clampHtmlZoom(10)).toBe(50)
    expect(clampHtmlZoom(250)).toBe(200)
  })
})
