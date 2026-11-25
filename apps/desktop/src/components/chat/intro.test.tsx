import { act, cleanup, render } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider, useI18n } from '@/i18n'
import type { I18nContextValue } from '@/i18n'

import { Intro } from './intro'
import stock from './intro-copy.jsonl?raw'

const CJK_LOCALES = new Set(['zh', 'zh-hant', 'ja'])

let i18n: I18nContextValue

function Controls() {
  i18n = useI18n()

  return null
}

function Fixture({ personality, seed = 0 }: { personality?: string; seed?: number }) {
  return (
    <I18nProvider configClient={null} initialLocale="en">
      <Controls />
      <Intro personality={personality} seed={seed} />
    </I18nProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})
it('translates every shipped stock body at the same personality and rotation position', async () => {
  vi.spyOn(Math, 'random').mockReturnValue(0)

  const entries = stock
    .trim()
    .split('\n')
    .map(line => JSON.parse(line) as { personality: string; body: string })

  const { container, rerender } = render(<Fixture />)

  for (const locale of ['zh', 'zh-hant', 'ja', 'fr', 'de', 'es'] as const) {
    await act(() => i18n.setLocale(locale))
    const indices = new Map<string, number>()

    for (const personality of new Set(entries.map(entry => entry.personality))) {
      expect(i18n.t.intro.stock[personality]).toHaveLength(
        entries.filter(entry => entry.personality === personality).length
      )
    }

    for (const entry of entries) {
      const seed = indices.get(entry.personality) ?? 0
      indices.set(entry.personality, seed + 1)
      rerender(<Fixture personality={entry.personality} seed={seed} />)
      const body = container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent
      expect(body).toBeTruthy()
      expect(body).not.toBe(entry.body)

      if (CJK_LOCALES.has(locale)) {
        expect(body).toMatch(/[\u3040-\u30ff\u3400-\u9fff]/)
      }
    }
  }

  await act(() => i18n.setLocale('en'))
  rerender(<Fixture personality={entries[0].personality} seed={0} />)
  expect(container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent).toBe(entries[0].body)
})
it('localizes the custom-personality fallback without translating its user-supplied name', async () => {
  vi.spyOn(Math, 'random').mockReturnValue(0)
  const { container } = render(<Fixture personality="My Custom Voice" seed={4} />)
  const english = container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent
  await act(() => i18n.setLocale('zh'))
  expect(container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent).not.toBe(english)
  expect(container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent).toContain(
    'My Custom Voice'
  )
  await act(() => i18n.setLocale('ja'))
  expect(container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent).toContain(
    'My Custom Voice'
  )

  for (const locale of ['fr', 'de', 'es'] as const) {
    await act(() => i18n.setLocale(locale))
    const body = container.querySelector('[data-slot="aui_intro"] > div > p:last-child')!.textContent
    expect(body).not.toBe(english)
    expect(body).toContain('My Custom Voice')
  }
})
