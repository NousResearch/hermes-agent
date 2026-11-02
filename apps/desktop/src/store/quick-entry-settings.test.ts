// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $quickEntry, loadQuickEntrySettings, type QuickEntryStatus, saveQuickEntrySettings } from './quick-entry'

const shipped: QuickEntryStatus = {
  enabled: true,
  error: null,
  registered: true,
  shortcut: 'CommandOrControl+Shift+Space',
  target: 'current'
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

// Main is authoritative for this preference (it owns the persisted file), so the
// bridge only has to prove the renderer ASKS for the target and adopts whatever
// the main process reports back.
function bridge(status: QuickEntryStatus = shipped) {
  const getSettings = vi.fn(async () => status)

  const setSettings = vi.fn(async (patch: { enabled?: boolean; shortcut?: string; target?: string }) => ({
    ...status,
    ...patch
  }))

  vi.stubGlobal('hermesDesktop', { quickEntry: { getSettings, setSettings } })

  return { getSettings, setSettings }
}

describe('quick entry target preference', () => {
  it('reads the persisted target into the settings store', async () => {
    bridge({ ...shipped, target: 'new' })

    await loadQuickEntrySettings()

    expect($quickEntry.get().target).toBe('new')
  })

  it('writes the target and adopts the authoritative reply', async () => {
    const { setSettings } = bridge()

    await saveQuickEntrySettings({ target: 'new' })

    expect(setSettings).toHaveBeenLastCalledWith({ target: 'new' })
    expect($quickEntry.get().target).toBe('new')
  })

  it('falls back to the current chat when main reports an unusable target', async () => {
    bridge({ ...shipped, target: 'session-42' as never })

    await loadQuickEntrySettings()

    // Never leave the picker on a value the select cannot render, and never let a
    // stale preference point a prompt at a session that may not exist.
    expect($quickEntry.get().target).toBe('current')
  })
})
