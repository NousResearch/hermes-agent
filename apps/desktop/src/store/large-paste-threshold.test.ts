import { beforeEach, expect, it, vi } from 'vitest'

import { LARGE_PASTE_ATTACHMENT_THRESHOLD } from '@/app/chat/composer/large-paste'

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
