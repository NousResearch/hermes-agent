import { describe, expect, it } from 'vitest'

import { captureTextPosition, contentFingerprint, restoreTextPosition } from './anchors'

describe('annotation anchors', () => {
  it('captures and restores a selection spanning text nodes', () => {
    const root = document.createElement('div')
    root.innerHTML = '<p>Hello <strong>cross node</strong> selection</p>'
    const start = root.querySelector('p')!.firstChild!
    const end = root.querySelector('strong')!.firstChild!
    const range = document.createRange()
    range.setStart(start, 3)
    range.setEnd(end, 5)

    const position = captureTextPosition(root, range)!
    expect(position.quote).toBe('lo cross')
    expect(restoreTextPosition(root, position)?.toString()).toBe('lo cross')
  })

  it('uses prefix and suffix to disambiguate repeated text', () => {
    const root = document.createElement('div')
    root.textContent = 'first target middle second target end'
    const text = root.firstChild!
    const start = root.textContent.indexOf('target', 10)
    const range = document.createRange()
    range.setStart(text, start)
    range.setEnd(text, start + 6)
    const position = captureTextPosition(root, range)!

    root.innerHTML = '<span>first target middle second </span><strong>target</strong><span> end</span>'
    expect(restoreTextPosition(root, position)?.startContainer.parentElement?.tagName).toBe('STRONG')
  })

  it('fingerprints both content and length', () => {
    expect(contentFingerprint('alpha')).toMatch(/^fnv1a-[0-9a-f]{8}-5$/)
    expect(contentFingerprint('alpha')).not.toBe(contentFingerprint('beta'))
  })
})
