import { afterAll, afterEach, expect, it } from 'vitest'

import { $layoutScale } from '@/lib/layout-scale'
import { $shellChoice } from '@/store/shell-variant'

import { installLayoutScale } from './layout-scale'

const root = window.document.documentElement
let factor = 1
let changed: () => void = () => undefined

// The installer reads the host and the zoom bridge once, at install: both stand for this file.
const bridge = window.hermesDesktop

root.dataset.hermesDesktopHost = 'browser'
window.hermesDesktop = {
  zoom: { factor: () => factor, onChanged: (listener: () => void) => void (changed = listener) }
} as never
installLayoutScale()

afterAll(() => {
  window.hermesDesktop = bridge
  delete root.dataset.hermesDesktopHost
})

function setScale(next: number) {
  factor = next
  changed()
}

afterEach(() => {
  setScale(1)
  $shellChoice.set(null)
})

// The rules that take a length back to the window under the UI scale (a dialog's height cap,
// portal-floating.css) are keyed on this mark, so at 100% they are not in the cascade at all and a
// dialog's own cap is the one that applies.
it('marks the document only while the redesign lays out for a UI scale', () => {
  $shellChoice.set(true)
  expect(root.hasAttribute('data-layout-scaled')).toBe(false)

  setScale(2)
  expect($layoutScale.get()).toBe(2)
  expect(root.hasAttribute('data-layout-scaled')).toBe(true)

  setScale(1)
  expect(root.hasAttribute('data-layout-scaled')).toBe(false)
})

it('leaves classic unmarked and unscaled at any UI scale', () => {
  $shellChoice.set(false)
  setScale(2)

  expect($layoutScale.get()).toBe(1)
  expect(root.hasAttribute('data-layout-scaled')).toBe(false)
})

it('rewrites a stylesheet size query to the scale and restores it as written at 100%', () => {
  $shellChoice.set(true)
  const style = window.document.createElement('style')
  style.textContent = '@media (min-width: 1280px) { body { color: red } }'
  window.document.head.append(style)
  const media = () => (style.sheet!.cssRules[0] as CSSMediaRule).media.mediaText

  setScale(2)
  expect(media()).toBe('(min-width: 2560px)')

  setScale(1)
  expect(media()).toBe('(min-width: 1280px)')
  style.remove()
})

it('rescales for a stylesheet that finishes loading, not for an image that does', async () => {
  $shellChoice.set(true)
  setScale(2)
  const style = window.document.createElement('style')
  style.textContent = '@media (min-width: 640px) { body { color: red } }'
  // Outside <head>, so only a load can bring the rescale.
  window.document.body.append(style)
  const media = () => (style.sheet!.cssRules[0] as CSSMediaRule).media.mediaText
  const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve))

  const image = window.document.body.appendChild(window.document.createElement('img'))
  image.dispatchEvent(new Event('load'))
  await nextFrame()
  expect(media()).toBe('(min-width: 640px)')

  const link = window.document.body.appendChild(window.document.createElement('link'))
  link.dispatchEvent(new Event('load'))
  await nextFrame()
  expect(media()).toBe('(min-width: 1280px)')

  style.remove()
  image.remove()
  link.remove()
})
