import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { ComposerStatusItem } from '@/store/composer-status'

import { StatusItemRow } from './status-row'

const longTodoTitle =
  'Obtain independent Codex, Kimi, and xAI OAuth grants without sharing credentials across hosts'

function renderRow(item: ComposerStatusItem) {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <StatusItemRow item={item} />
    </I18nProvider>
  )
}

describe('StatusItemRow todo readability', () => {
  afterEach(() => {
    cleanup()
  })

  it('shows the full todo title without truncating classes', () => {
    renderRow({
      id: 'todo:1',
      state: 'running',
      title: longTodoTitle,
      todoStatus: 'in_progress',
      type: 'todo'
    })

    const title = screen.getByText(longTodoTitle)
    expect(title).toBeTruthy()
    expect(title.className).toMatch(/break-words/)
    expect(title.className).toMatch(/whitespace-normal/)
    expect(title.className).not.toMatch(/\btruncate\b/)
    expect(title.className).not.toMatch(/max-w-\[18rem\]/)
  })

  it('keeps non-todo rows on the compact single-line truncate chrome', () => {
    renderRow({
      id: 'bg:1',
      state: 'running',
      title: longTodoTitle,
      type: 'background'
    })

    const title = screen.getByText(longTodoTitle)
    expect(title.className).toMatch(/\btruncate\b/)
    expect(title.className).toMatch(/max-w-\[18rem\]/)
    expect(title.className).not.toMatch(/break-words/)
  })

  it('keeps completed todo titles readable (not heavily muted)', () => {
    renderRow({
      id: 'todo:done',
      state: 'done',
      title: longTodoTitle,
      todoStatus: 'completed',
      type: 'todo'
    })

    const title = screen.getByText(longTodoTitle)
    expect(title.className).toMatch(/text-foreground\/78/)
    expect(title.className).not.toMatch(/text-muted-foreground\/75/)
  })
})
