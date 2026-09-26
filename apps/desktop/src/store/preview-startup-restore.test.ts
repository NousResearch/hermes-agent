/** Regression coverage for #119895: persisted Preview tabs must survive module startup. */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const TABS_KEY = 'hermes.desktop.previewTabs.v2'

const persistedTab = {
  id: 'url:browser-restored',
  target: {
    kind: 'url',
    label: 'Restored page',
    source: 'https://example.com/restored',
    url: 'https://example.com/restored'
  }
}

describe('Preview tab startup restore', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  afterEach(() => {
    window.localStorage.clear()
    vi.resetModules()
  })

  it('hydrates the default profile before persistence can overwrite it', async () => {
    const persisted = { default: [persistedTab] }
    window.localStorage.setItem(TABS_KEY, JSON.stringify(persisted))

    const { $previewTabs, setPreviewScope } = await import('./preview')

    expect($previewTabs.get()).toEqual([persistedTab])

    setPreviewScope('default')

    expect($previewTabs.get()).toEqual([persistedTab])
    expect(JSON.parse(window.localStorage.getItem(TABS_KEY) ?? 'null')).toEqual(persisted)
  })

  it.each(['default', 'tess'])('adopts legacy tabs into the first %s scope', async scope => {
    const legacy = [persistedTab]
    window.localStorage.setItem(TABS_KEY, JSON.stringify(legacy))

    const { $previewTabs, setPreviewScope } = await import('./preview')
    setPreviewScope(scope)

    expect($previewTabs.get()).toEqual(legacy)
    expect(JSON.parse(window.localStorage.getItem(TABS_KEY) ?? 'null')).toEqual({ [scope]: legacy })
  })
})
