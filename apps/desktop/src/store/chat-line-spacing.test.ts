import { beforeEach, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.chat-line-spacing.v1'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  document.documentElement.style.removeProperty('--chat-line-spacing')
})

it('publishes the multiplier for the conversation without touching the other chat levers', async () => {
  const { $chatTextScale, setChatTextScale } = await import('./chat-text-scale')
  const textScale = $chatTextScale.get()
  const { $chatLineSpacing, setChatLineSpacing, CHAT_LINE_SPACING_PRESETS } = await import('./chat-line-spacing')
  const selected = CHAT_LINE_SPACING_PRESETS.find(value => value !== $chatLineSpacing.get())!
  const root = document.documentElement
  // A theme's own leading is the baseline the multiplier scales, so it must
  // survive a line-spacing change untouched.
  root.style.setProperty('--dt-line-height', '1.65')

  setChatLineSpacing(selected)

  expect(root.style.getPropertyValue('--chat-line-spacing')).toBe(String(selected / 100))
  expect(root.style.getPropertyValue('--dt-line-height')).toBe('1.65')
  expect($chatTextScale.get()).toBe(textScale)
  expect(localStorage.getItem(KEY)).toBe(String(selected))

  vi.resetModules()
  const restored = await import('./chat-line-spacing')
  expect(restored.$chatLineSpacing.get()).toBe(selected)

  restored.setChatLineSpacing($chatLineSpacing.get())
  root.style.removeProperty('--dt-line-height')
  setChatTextScale(textScale)
})

it('spans 75% to 175% in even 25% steps with the 100% no-op present', async () => {
  const { CHAT_LINE_SPACING_MAX, CHAT_LINE_SPACING_MIN, CHAT_LINE_SPACING_PRESETS } =
    await import('./chat-line-spacing')

  const presets = [...CHAT_LINE_SPACING_PRESETS]

  expect(CHAT_LINE_SPACING_MIN).toBe(75)
  expect(CHAT_LINE_SPACING_MAX).toBe(175)
  expect(presets).toContain(100)
  expect(presets).toEqual(presets.toSorted((a, b) => a - b))
  expect(new Set(presets).size).toBe(presets.length)
  expect(presets.every((value, index) => index === 0 || value - presets[index - 1] === 25)).toBe(true)
})

it('falls back to the pre-setting rendering for invalid storage and clears an explicit default', async () => {
  const initial = await import('./chat-line-spacing')
  const fallback = initial.$chatLineSpacing.get()
  expect(fallback).toBe(100)

  localStorage.setItem(KEY, 'not-a-preset')
  vi.resetModules()
  const { $chatLineSpacing, setChatLineSpacing } = await import('./chat-line-spacing')
  expect($chatLineSpacing.get()).toBe(fallback)
  expect(document.documentElement.style.getPropertyValue('--chat-line-spacing')).toBe('1')

  setChatLineSpacing(125)
  expect(localStorage.getItem(KEY)).toBe('125')
  setChatLineSpacing(fallback)
  expect(localStorage.getItem(KEY)).toBeNull()
})
