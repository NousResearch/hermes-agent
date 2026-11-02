// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, expect, it, vi } from 'vitest'

import { en } from '@/i18n/en'

import { QuickEntrySettings } from './quick-entry-settings'

vi.mock('@/i18n', () => ({ useI18n: () => ({ t: en }) }))

const copy = en.settings.quickEntry

// Radix Select needs these in jsdom (pointer capture + scrollIntoView are not
// implemented), otherwise the trigger never opens and no option can be picked.
beforeAll(() => {
  window.HTMLElement.prototype.hasPointerCapture = vi.fn(() => false)
  window.HTMLElement.prototype.releasePointerCapture = vi.fn()
  window.HTMLElement.prototype.scrollIntoView = vi.fn()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const shipped = {
  enabled: true,
  error: null,
  registered: true,
  shortcut: 'CommandOrControl+Shift+Space',
  target: 'current' as const
}

it('offers the default send target next to the shortcut, disabled with the feature', async () => {
  const getSettings = vi.fn().mockResolvedValue({ ...shipped, enabled: false, registered: false })

  vi.stubGlobal('hermesDesktop', { quickEntry: { getSettings, setSettings: vi.fn() } })

  render(<QuickEntrySettings />)

  expect(screen.getByText(copy.targetTitle)).toBeTruthy()
  expect(screen.getByText(copy.targetDesc)).toBeTruthy()

  // The preference is only meaningful while the feature can be summoned, so the
  // picker follows the same disabled state as the shortcut field.
  const picker = await screen.findByRole('combobox', { name: copy.targetTitle })
  await waitFor(() => expect(picker).toHaveProperty('disabled', true))
  expect(screen.getByRole('switch', { name: copy.enabledTitle })).toHaveProperty('ariaChecked', 'false')
})

it('writes the picked target through the bridge', async () => {
  const setSettings = vi.fn(async (patch: Record<string, unknown>) => ({ ...shipped, ...patch }))

  vi.stubGlobal('hermesDesktop', { quickEntry: { getSettings: vi.fn().mockResolvedValue(shipped), setSettings } })

  render(<QuickEntrySettings />)

  const picker = await screen.findByRole('combobox', { name: copy.targetTitle })
  await waitFor(() => expect(picker).toHaveProperty('disabled', false))

  fireEvent.pointerDown(picker, { button: 0, ctrlKey: false, pointerType: 'mouse' })
  fireEvent.click(await screen.findByRole('option', { name: copy.targetNew }))

  await waitFor(() => expect(setSettings).toHaveBeenCalledWith({ target: 'new' }))
})
