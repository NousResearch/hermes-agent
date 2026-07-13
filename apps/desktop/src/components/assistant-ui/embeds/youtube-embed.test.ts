import { describe, expect, it } from 'vitest'

import { youtubeSrc } from './youtube-embed'

describe('youtubeSrc', () => {
  it('enables autoplay after the user clicks the privacy facade', () => {
    const src = youtubeSrc('https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?rel=0', true)

    expect(src).toContain('autoplay=1')
  })

  it('does not autoplay embeds loaded by standing consent', () => {
    const src = youtubeSrc('https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ?rel=0')

    expect(src).not.toContain('autoplay=1')
  })
})
