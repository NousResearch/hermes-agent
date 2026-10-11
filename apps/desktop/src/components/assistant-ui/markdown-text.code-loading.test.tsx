import { cleanup, render, waitFor, within } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { MarkdownTextContent } from './markdown-text'

afterEach(cleanup)

it('renders prose and later fenced code with Hermes cards and copy controls', async () => {
  const { container, rerender } = render(<MarkdownTextContent isRunning text={'# Answer\n\nOrdinary **prose**.'} />)

  expect(container.querySelector('h1')?.textContent).toBe('Answer')
  expect(container.textContent).toContain('Ordinary prose.')
  expect(container.querySelector('[data-slot="code-card"]')).toBeNull()

  rerender(<MarkdownTextContent isRunning={false} text={'# Answer\n\n```python\nprint(42)\n```'} />)
  await waitFor(() => expect(container.querySelector('[data-slot="code-card"]')).not.toBeNull())
  const card = container.querySelector<HTMLElement>('[data-slot="code-card"]')!

  expect(card.querySelector('[data-slot="code-card-body"] pre')?.textContent).toContain('print(42)')
  expect(within(card).getByRole('button', { name: 'Copy code' })).toBeDefined()
  expect(container.textContent).not.toContain('```python')
})

it.each([
  '```python\nprint(42)',
  '~~~python\nprint(42)\n~~~',
  '\n    print(42)',
  '\n\tprint(42)',
  '>\n>     print(42)',
  '- Example:\n\n  ```python\n  print(42)'
])('keeps streaming code inside a Hermes card with copy controls: %s', async text => {
  const { container } = render(<MarkdownTextContent isRunning text={text} />)

  await waitFor(() => expect(container.querySelector('[data-slot="code-card"]')).not.toBeNull())
  const card = container.querySelector<HTMLElement>('[data-slot="code-card"]')!

  expect(card.dataset.streaming).toBe('true')
  expect(card.querySelector('[data-slot="code-card-body"] pre')?.textContent).toContain('print(42)')
  expect(within(card).getByRole('button', { name: 'Copy code' })).toBeDefined()
})
