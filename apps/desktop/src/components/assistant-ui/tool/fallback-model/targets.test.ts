import { describe, expect, it } from 'vitest'

import { isPreviewableTarget } from './targets'

describe('isPreviewableTarget', () => {
  it('rejects virtual app.asar files without hiding real unpacked files', () => {
    expect(
      isPreviewableTarget(
        'file:///C:/Hermes/resources/app.asar/dist/index.html#/sessions/123'
      )
    ).toBe(false)
    expect(isPreviewableTarget('file:///C:/Hermes/resources/app.asar.unpacked/dist/index.html')).toBe(true)
  })
})
