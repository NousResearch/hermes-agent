import { expect, it } from 'vitest'

import { readLensGuest } from './capture'

it('rejects a read that completes after its guest navigates elsewhere', async () => {
  const source = {
    url: 'https://example.com/item',
    title: 'Source',
    text: 'Evidence',
    selector: '#item',
    tag: 'P',
    truncated: false
  }

  let url = source.url

  const guest = {
    getURL: () => url,
    executeJavaScript: async () => {
      url = 'https://example.com/next'

      return source
    },
    addEventListener() {},
    removeEventListener() {}
  }

  await expect(readLensGuest(guest, 'page')).rejects.toThrow('sourceChanged')
})
