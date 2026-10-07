import { beforeEach, expect, it, vi } from 'vitest'

import { LARGE_PASTE_ATTACHMENT_THRESHOLD, shouldConvertPasteToAttachment } from '@/app/chat/composer/large-paste'

const KEY = 'hermes.desktop.large-paste-attachment-threshold.v1'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
})

it('uses the existing default when no preference has been saved', async () => {
  const { $largePasteAttachmentThreshold } = await import('./large-paste-threshold')
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
})

it.each([0, 50_000, 100_000])('persists and restores %s across a renderer restart', async value => {
  const { setLargePasteAttachmentThreshold } = await import('./large-paste-threshold')
  setLargePasteAttachmentThreshold(value)
  expect(localStorage.getItem(KEY)).toBe(String(value))

  vi.resetModules()
  const { $largePasteAttachmentThreshold } = await import('./large-paste-threshold')
  expect($largePasteAttachmentThreshold.get()).toBe(value)
})

it('removes the saved override when restoring the default', async () => {
  const { setLargePasteAttachmentThreshold } = await import('./large-paste-threshold')
  setLargePasteAttachmentThreshold(0)
  setLargePasteAttachmentThreshold(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  expect(localStorage.getItem(KEY)).toBeNull()

  vi.resetModules()
  const { $largePasteAttachmentThreshold } = await import('./large-paste-threshold')
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
})

it.each(['', ' ', 'invalid', '-1', '1.5', 'NaN', 'Infinity', '100001'])('ignores invalid storage %s', async value => {
  localStorage.setItem(KEY, value)
  const { $largePasteAttachmentThreshold } = await import('./large-paste-threshold')
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
})

it.each([-1, 1.5, NaN, Infinity, 100_001])('sanitizes an invalid setter value %s', async value => {
  const { $largePasteAttachmentThreshold, setLargePasteAttachmentThreshold } = await import('./large-paste-threshold')
  setLargePasteAttachmentThreshold(value)
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  expect(localStorage.getItem(KEY)).toBeNull()
})

// Separate module instances and event targets model two already-open renderers.
// Storage is shared, but only the receiving window gets the browser's event.
it('synchronizes open renderer stores without persisting received storage events', async () => {
  const storage = localStorage
  const firstWindow = Object.assign(new EventTarget(), { localStorage: storage })
  const secondWindow = Object.assign(new EventTarget(), { localStorage: storage })

  try {
    vi.stubGlobal('window', firstWindow)
    const first = await import('./large-paste-threshold')
    vi.resetModules()
    vi.stubGlobal('window', secondWindow)
    const second = await import('./large-paste-threshold')
    const writes = vi.spyOn(storage, 'setItem')
    const removals = vi.spyOn(storage, 'removeItem')
    const goal = '/goal ' + 'a'.repeat(11_694)

    const receive = (key: string | null = KEY) => {
      vi.stubGlobal('window', secondWindow)
      secondWindow.dispatchEvent(new StorageEvent('storage', { key }))
    }

    const change = (value: number) => {
      vi.stubGlobal('window', firstWindow)
      first.setLargePasteAttachmentThreshold(value)
      receive()
      expect(second.$largePasteAttachmentThreshold.get()).toBe(value)
    }

    change(50_000)
    expect(shouldConvertPasteToAttachment(goal, second.$largePasteAttachmentThreshold.get())).toBe(false)
    change(0)
    expect(storage.getItem(KEY)).toBe('0')
    expect(shouldConvertPasteToAttachment('a'.repeat(100_001), second.$largePasteAttachmentThreshold.get())).toBe(false)
    change(10_000)
    expect(shouldConvertPasteToAttachment(goal, second.$largePasteAttachmentThreshold.get())).toBe(true)
    change(LARGE_PASTE_ATTACHMENT_THRESHOLD)
    expect(storage.getItem(KEY)).toBeNull()
    expect(writes).toHaveBeenCalledTimes(3)
    expect(removals).toHaveBeenCalledTimes(1)

    for (const value of ['invalid', '-1', '1.5', 'Infinity', '100001']) {
      change(0)
      storage.setItem(KEY, value)
      receive()
      expect(second.$largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
    }

    change(50_000)
    storage.setItem(KEY, '0')
    receive('unrelated-preference')
    expect(second.$largePasteAttachmentThreshold.get()).toBe(50_000)
    storage.clear()
    receive(null)
    expect(second.$largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  } finally {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  }
})
