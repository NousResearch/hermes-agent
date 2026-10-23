import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

const reveal = vi.hoisted(() => vi.fn())
vi.mock('@hermes/plugin-sdk', async () => {
  const { MessageTextContent } = await import('@/components/assistant-ui/markdown-text')

  return { host: { revealFileInTree: reveal }, MessageTextContent }
})

import { TaskMarkdown } from './task-markdown'

afterEach(() => {
  cleanup()
  reveal.mockClear()
})

it('reveals prose paths without replacing the Markdown renderer', async () => {
  const { container } = render(
    <TaskMarkdown linkPaths text={'**Changed** /workspace/src/main.ts. Then /workspace/tests/main.test.ts.'} />
  )

  const first = await screen.findByRole('button', { name: 'Reveal /workspace/src/main.ts in file tree' })
  fireEvent.click(first)
  expect(reveal).toHaveBeenCalledWith('/workspace/src/main.ts')
  expect(container.querySelector('[data-streamdown="strong"]')?.textContent).toBe('Changed')
  expect(container.textContent).toContain('/workspace/src/main.ts.')
  expect(screen.getAllByRole('button', { name: /^Reveal / })).toHaveLength(2)
})

it('does not change paths on surfaces that did not opt in', () => {
  render(<TaskMarkdown text="See /workspace/src/main.ts" />)
  expect(screen.queryByRole('button', { name: /^Reveal / })).toBeNull()
})

it('preserves inline formatting, external links, and fenced-code boundaries', async () => {
  const text =
    '**/workspace/bold.ts** and `/workspace/inline.ts`. /help is a command.\n\n' +
    '[existing](https://example.com/a/b.txt)\n\n```text\n/workspace/fenced.ts\n```'

  const { container } = render(<TaskMarkdown linkPaths text={text} />)
  await screen.findByRole('button', { name: 'Reveal /workspace/inline.ts in file tree' })
  expect(screen.getAllByRole('button', { name: /^Reveal / })).toHaveLength(2)
  expect(screen.queryByRole('button', { name: /Reveal \/help / })).toBeNull()
  expect(screen.queryByRole('button', { name: /fenced/ })).toBeNull()
  expect(container.querySelector('a[href="https://example.com/a/b.txt"]')).not.toBeNull()
  expect(container.querySelector('code button')?.textContent).toBe('/workspace/inline.ts')
})

it('recognizes Windows drive and UNC paths without including punctuation', async () => {
  const text = 'See `' + String.raw`C:\workspace\src\main.ts` + '` and `' + String.raw`\\server\share\file.txt` + '`.'
  render(<TaskMarkdown linkPaths text={text} />)
  const drive = await screen.findByRole('button', { name: String.raw`Reveal C:\workspace\src\main.ts in file tree` })
  fireEvent.click(drive)
  expect(reveal).toHaveBeenCalledWith(String.raw`C:\workspace\src\main.ts`)
  expect(screen.getByRole('button', { name: String.raw`Reveal \\server\share\file.txt in file tree` })).toBeTruthy()
})
