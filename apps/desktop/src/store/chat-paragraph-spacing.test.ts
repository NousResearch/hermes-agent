import { beforeEach, expect, it, vi } from 'vitest'

const KEY = 'hermes.desktop.chat-paragraph-spacing.v1'

beforeEach(() => {
  vi.resetModules()
  localStorage.clear()
  document.documentElement.style.removeProperty('--chat-paragraph-spacing')
})

it('publishes the multiplier for the transcript without touching the other chat levers', async () => {
  const { $chatLineSpacing, setChatLineSpacing } = await import('./chat-line-spacing')
  const lineSpacing = $chatLineSpacing.get()

  const { CHAT_PARAGRAPH_SPACING_PRESETS, $chatParagraphSpacing, setChatParagraphSpacing } =
    await import('./chat-paragraph-spacing')

  const selected = CHAT_PARAGRAPH_SPACING_PRESETS.find(value => value !== $chatParagraphSpacing.get())!
  const root = document.documentElement
  // The surface's own gap is the baseline the multiplier scales, so it must
  // survive a paragraph-spacing change untouched.
  root.style.setProperty('--paragraph-gap-base', '0.7rem')

  setChatParagraphSpacing(selected)

  expect(root.style.getPropertyValue('--chat-paragraph-spacing')).toBe(String(selected / 100))
  expect(root.style.getPropertyValue('--paragraph-gap-base')).toBe('0.7rem')
  expect($chatLineSpacing.get()).toBe(lineSpacing)
  expect(localStorage.getItem(KEY)).toBe(String(selected))

  vi.resetModules()
  const restored = await import('./chat-paragraph-spacing')
  expect(restored.$chatParagraphSpacing.get()).toBe(selected)

  restored.setChatParagraphSpacing($chatParagraphSpacing.get())
  root.style.removeProperty('--paragraph-gap-base')
  setChatLineSpacing(lineSpacing)
})

it('spans 75% to 250% in widening steps with the 100% no-op present', async () => {
  const { CHAT_PARAGRAPH_SPACING_MAX, CHAT_PARAGRAPH_SPACING_MIN, CHAT_PARAGRAPH_SPACING_PRESETS } =
    await import('./chat-paragraph-spacing')

  const presets = [...CHAT_PARAGRAPH_SPACING_PRESETS]

  // The shipped ladder is a UI contract: 100% has to be selectable (it is the
  // no-op) and the top stop has to clear the block gaps a reply already paints
  // (a frozen 17.875px list gap beat the 22.4px paragraph gap at 200%), so the
  // steps widen above the default.
  expect(presets).toEqual([75, 100, 150, 200, 250])
  expect(CHAT_PARAGRAPH_SPACING_MIN).toBe(75)
  expect(CHAT_PARAGRAPH_SPACING_MAX).toBe(250)
  expect(presets).toContain(100)
  expect(presets).toEqual(presets.toSorted((a, b) => a - b))
  expect(new Set(presets).size).toBe(presets.length)
  expect(presets.every((value, index) => index === 0 || value - presets[index - 1] <= 50)).toBe(true)
})

it('falls back to the pre-setting rendering for invalid storage and clears an explicit default', async () => {
  const initial = await import('./chat-paragraph-spacing')
  const fallback = initial.$chatParagraphSpacing.get()
  expect(fallback).toBe(100)

  localStorage.setItem(KEY, 'not-a-preset')
  vi.resetModules()
  const { $chatParagraphSpacing, setChatParagraphSpacing } = await import('./chat-paragraph-spacing')
  expect($chatParagraphSpacing.get()).toBe(fallback)
  expect(document.documentElement.style.getPropertyValue('--chat-paragraph-spacing')).toBe('1')

  setChatParagraphSpacing(250)
  expect(localStorage.getItem(KEY)).toBe('250')

  // 125% is no longer on the ladder (the steps widen above the default), so it
  // normalises to the 100% no-op and clears the stored override.
  setChatParagraphSpacing(125)
  expect(localStorage.getItem(KEY)).toBeNull()

  setChatParagraphSpacing(fallback)
  expect(localStorage.getItem(KEY)).toBeNull()
})
