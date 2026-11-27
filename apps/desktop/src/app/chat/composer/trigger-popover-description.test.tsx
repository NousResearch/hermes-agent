import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

import { ComposerTriggerPopover } from './trigger-popover'

afterEach(cleanup)

it('reveals the complete slash description on hover without intercepting selection', async () => {
  const description = 'Complete command help '.repeat(20)
  const item = { id: '/proof', type: 'slash', label: 'proof', metadata: { display: '/proof', meta: description } }
  const onPick = vi.fn()
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerTriggerPopover
        activeIndex={0}
        items={[item]}
        kind="/"
        loading={false}
        onHover={vi.fn()}
        onPick={onPick}
      />
    </I18nProvider>
  )
  const row = screen.getByRole('button')
  fireEvent.pointerMove(row, { pointerType: 'mouse' })
  expect((await screen.findByRole('tooltip')).textContent).toBe(description)
  fireEvent.click(row)
  expect(onPick).toHaveBeenCalledWith(item)
})

it.each([
  ['file', 'src/app/chat/composer/components/main.tsx', 'main.tsx'],
  ['folder', 'src/app/chat/composer/components/assets/', 'assets/']
])('keeps the last segment of a %s suggestion distinct from its parent path', async (type, path, leaf) => {
  const item = { id: path, type, label: path, metadata: { display: path, rawText: `@${type}:${path}` } }
  const onPick = vi.fn()

  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ComposerTriggerPopover
        activeIndex={0}
        items={[item]}
        kind="@"
        loading={false}
        onHover={vi.fn()}
        onPick={onPick}
      />
    </I18nProvider>
  )

  const row = screen.getByRole('button', { name: path })
  expect(within(row).getByText(leaf).textContent).toBe(leaf)
  fireEvent.pointerMove(row, { pointerType: 'mouse' })
  expect((await screen.findByRole('tooltip')).textContent).toBe(path)
  fireEvent.click(row)
  expect(onPick).toHaveBeenCalledWith(item)
})
