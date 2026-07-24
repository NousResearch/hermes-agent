import { describe, expect, it, vi } from 'vitest'

import { discoverBundledPlugins } from './plugins'

describe('Desktop plugin code boundary', () => {
  it('discovers the bundled Kanban plugin without creating an asset script', () => {
    const createElement = vi.spyOn(document, 'createElement')
    const appendChild = vi.spyOn(document.body, 'appendChild')

    discoverBundledPlugins()

    expect(createElement).not.toHaveBeenCalledWith('script')
    expect(
      appendChild.mock.calls.some(([node]) => node instanceof HTMLScriptElement)
    ).toBe(false)
  })
})
