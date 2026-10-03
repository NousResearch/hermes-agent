import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { LARGE_PASTE_ATTACHMENT_THRESHOLD } from '@/app/chat/composer/large-paste'
import { I18nProvider } from '@/i18n'
import { $largePasteAttachmentThreshold, setLargePasteAttachmentThreshold } from '@/store/large-paste-threshold'

import { LargePasteThresholdSetting } from './large-paste-threshold-setting'
import { SETTING_IDS, settingElementId } from './settings-manifest'

beforeEach(() => {
  setLargePasteAttachmentThreshold(LARGE_PASTE_ATTACHMENT_THRESHOLD)
})

afterEach(() => {
  cleanup()
  setLargePasteAttachmentThreshold(LARGE_PASTE_ATTACHMENT_THRESHOLD)
})

function renderSetting() {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <LargePasteThresholdSetting />
    </I18nProvider>
  )

  return screen.getByRole('spinbutton', { name: 'Large paste attachment threshold' }) as HTMLInputElement
}

it.each([0, 50_000, 100_000])('commits %s on blur and persists it', value => {
  const input = renderSetting()
  fireEvent.change(input, { target: { value: String(value) } })
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  fireEvent.blur(input)
  expect($largePasteAttachmentThreshold.get()).toBe(value)
  expect(localStorage.getItem('hermes.desktop.large-paste-attachment-threshold.v1')).toBe(String(value))
  expect(input.value).toBe(String(value))
})

it('commits on Enter, but not while composing with an IME', () => {
  const input = renderSetting()
  input.focus()
  fireEvent.change(input, { target: { value: '50000' } })
  fireEvent.keyDown(input, { key: 'Enter', isComposing: true })
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  fireEvent.keyDown(input, { key: 'Enter' })
  expect($largePasteAttachmentThreshold.get()).toBe(50_000)
})

it('resets blank input to the default', () => {
  setLargePasteAttachmentThreshold(0)
  const input = renderSetting()
  fireEvent.change(input, { target: { value: '' } })
  fireEvent.blur(input)
  expect($largePasteAttachmentThreshold.get()).toBe(LARGE_PASTE_ATTACHMENT_THRESHOLD)
  expect(input.value).toBe(String(LARGE_PASTE_ATTACHMENT_THRESHOLD))
  expect(localStorage.getItem('hermes.desktop.large-paste-attachment-threshold.v1')).toBeNull()
})

it('does not treat an unfinished numeric entry as a blank reset', () => {
  setLargePasteAttachmentThreshold(50_000)
  const input = renderSetting()
  fireEvent.change(input, { target: { value: '' } })
  // jsdom does not model Chromium's empty-value/badInput pair for "1e".
  Object.defineProperty(input, 'validity', { value: { badInput: true } })
  fireEvent.blur(input)
  expect($largePasteAttachmentThreshold.get()).toBe(50_000)
  expect(input.value).toBe('50000')
})

it.each(['-1', '1.5', '100001'])('restores the last committed value after invalid input %s', value => {
  setLargePasteAttachmentThreshold(50_000)
  const input = renderSetting()
  fireEvent.change(input, { target: { value } })
  fireEvent.blur(input)
  expect($largePasteAttachmentThreshold.get()).toBe(50_000)
  expect(input.value).toBe('50000')
})

it('reflects external changes and exposes the settings deep-link target', () => {
  const input = renderSetting()
  act(() => setLargePasteAttachmentThreshold(0))
  expect(input.value).toBe('0')
  expect(window.document.getElementById(settingElementId(SETTING_IDS.chat.largePasteThreshold))?.contains(input)).toBe(
    true
  )
})
