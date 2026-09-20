// The in-app Browser's tab store is window-scoped: the IDE (and the pop-out it
// spawns) read one key, every other window reads the global one. A pop-out that
// misses the scope finds no tab and renders blank — the regression this pins.
import { describe, expect, it } from 'vitest'

import { PREVIEW_TABS_GLOBAL_KEY, PREVIEW_TABS_IDE_KEY, previewTabsStorageKey } from './windows'

describe('previewTabsStorageKey', () => {
  it('scopes the IDE window and its pop-outs to the ide key', () => {
    expect(previewTabsStorageKey('?win=ide&profile=default&connectionId=')).toBe(PREVIEW_TABS_IDE_KEY)
    expect(previewTabsStorageKey('?win=browser&scope=ide&tab=url%3A1')).toBe(PREVIEW_TABS_IDE_KEY)
  })

  it('keeps every other window on the global key', () => {
    expect(previewTabsStorageKey('')).toBe(PREVIEW_TABS_GLOBAL_KEY)
    expect(previewTabsStorageKey('?win=browser&tab=url%3A1')).toBe(PREVIEW_TABS_GLOBAL_KEY)
    expect(previewTabsStorageKey('?win=hud&profile=default')).toBe(PREVIEW_TABS_GLOBAL_KEY)
    expect(previewTabsStorageKey('?win=secondary&sessionId=s1')).toBe(PREVIEW_TABS_GLOBAL_KEY)
  })
})
