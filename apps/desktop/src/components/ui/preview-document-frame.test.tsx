import { cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { PreviewDocumentFrame } from './preview-document-frame'

afterEach(() => {
  cleanup()
  document.documentElement.removeAttribute('data-hermes-desktop-host')
  vi.unstubAllGlobals()
})

it.each(['allow-scripts', ''])('sends browser HTML only to its own isolated frame (%s)', sandbox => {
  document.documentElement.dataset.hermesDesktopHost = 'browser'
  vi.stubGlobal('__HERMES_BASE_PATH__', '/hermes')
  vi.stubGlobal('__HERMES_UI_SURFACE__', 'webapp')
  const { container, rerender } = render(<PreviewDocumentFrame sandbox={sandbox} srcDoc="<h1>private</h1>" />)
  const frame = container.querySelector('iframe')!
  expect(frame.getAttribute('srcdoc')).toBeNull()
  expect(frame.src).toBe(`${window.location.origin}/hermes/assets/webapp-preview.html`)
  expect(frame.getAttribute('sandbox')).toBe('allow-scripts')
  const send = vi.spyOn(frame.contentWindow!, 'postMessage')
  fireEvent.load(frame)
  expect(send).toHaveBeenLastCalledWith({
    type: 'hermes-preview-document', html: '<h1>private</h1>', scripts: sandbox === 'allow-scripts'
  }, '*')
  rerender(<PreviewDocumentFrame sandbox={sandbox} srcDoc="<h1>updated</h1>" />)
  expect(send).toHaveBeenLastCalledWith(expect.objectContaining({ html: '<h1>updated</h1>' }), '*')
})

it('keeps native Desktop srcdoc and its original script permission', () => {
  const { container } = render(<PreviewDocumentFrame sandbox="" srcDoc="<h1>native</h1>" />)
  const frame = container.querySelector('iframe')!
  expect(frame.getAttribute('srcdoc')).toBe('<h1>native</h1>')
  expect(frame.getAttribute('sandbox')).toBe('')
  expect(frame.getAttribute('src')).toBeNull()
})
