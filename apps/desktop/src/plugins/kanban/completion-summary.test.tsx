import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { needsCompletionSummary, useCompletionSummary } from './completion-summary'
import { en, KANBAN_LOCALES } from './i18n'

describe('needsCompletionSummary', () => {
  it.each([
    ['todo', undefined, 'done', true],
    ['running', null, 'done', true],
    ['blocked', '   ', 'done', true],
    ['todo', 'shipped', 'done', false],
    ['review', undefined, 'done', false],
    ['todo', undefined, 'ready', false],
    ['todo', undefined, 'blocked', false]
  ])('%s card with result %j moved to %s: %s', (status, result, target, expected) => {
    expect(needsCompletionSummary({ result, status }, target)).toBe(expected)
  })
})

describe('useCompletionSummary', () => {
  let disposeLocales: () => void

  beforeEach(() => {
    disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  })

  afterEach(() => {
    cleanup()
    disposeLocales()
  })

  function Harness({ count }: { count?: number }) {
    const { ask, dialog } = useCompletionSummary()
    const [answer, setAnswer] = useState('unanswered')

    return (
      <>
        <button onClick={() => void ask(count).then(summary => setAnswer(String(summary)))} type="button">
          open
        </button>
        <output>{answer}</output>
        {dialog}
      </>
    )
  }

  const open = (count?: number) => {
    render(<Harness count={count} />)
    fireEvent.click(screen.getByRole('button', { name: 'open' }))
  }

  const field = () => screen.getByPlaceholderText(en.completePlaceholder)

  it('resolves with the trimmed summary the user confirms', async () => {
    open()
    fireEvent.change(field(), { target: { value: '  approved by hand \n' } })
    fireEvent.click(screen.getByRole('button', { name: en.complete }))

    expect((await screen.findByRole('status')).textContent).toBe('approved by hand')
  })

  it('refuses to confirm an empty summary', () => {
    open()
    const confirm = screen.getByRole('button', { name: en.complete }) as HTMLButtonElement

    expect(confirm.disabled).toBe(true)
    fireEvent.change(field(), { target: { value: '   ' } })
    expect(confirm.disabled).toBe(true)
    fireEvent.change(field(), { target: { value: 'done' } })
    expect(confirm.disabled).toBe(false)
  })

  it('resolves null when the user cancels, so the move is dropped', async () => {
    open()
    fireEvent.change(field(), { target: { value: 'half typed' } })
    fireEvent.click(screen.getByRole('button', { name: en.cancel }))

    expect((await screen.findByRole('status')).textContent).toBe('null')
  })

  it('submits on Enter and keeps Shift+Enter for a new line', async () => {
    open()
    fireEvent.change(field(), { target: { value: 'line one' } })
    fireEvent.keyDown(field(), { key: 'Enter', shiftKey: true })
    // The open modal hides the rest of the page from the accessibility tree.
    expect(screen.getByRole('status', { hidden: true }).textContent).toBe('unanswered')

    fireEvent.keyDown(field(), { key: 'Enter' })
    expect((await screen.findByRole('status')).textContent).toBe('line one')
  })

  it('titles the dialog for the number of cards it completes', () => {
    open(3)
    expect(screen.getByRole('heading', { name: en.completeTitle(3) })).toBeTruthy()
  })

  it('starts each ask with an empty field', async () => {
    open()
    fireEvent.change(field(), { target: { value: 'first' } })
    fireEvent.click(screen.getByRole('button', { name: en.complete }))
    await screen.findByText('first')

    fireEvent.click(screen.getByRole('button', { name: 'open' }))
    expect((field() as HTMLTextAreaElement).value).toBe('')
  })
})

describe('useCompletionSummary withSummary', () => {
  let disposeLocales: () => void

  beforeEach(() => {
    disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  })

  afterEach(() => {
    cleanup()
    disposeLocales()
  })

  function Harness({ needed }: { needed: boolean }) {
    const { dialog, withSummary } = useCompletionSummary()
    const [applied, setApplied] = useState('not applied')

    return (
      <>
        <button onClick={() => void withSummary(needed, summary => setApplied(String(summary)))} type="button">
          move
        </button>
        <output>{applied}</output>
        {dialog}
      </>
    )
  }

  const move = (needed: boolean) => {
    render(<Harness needed={needed} />)
    fireEvent.click(screen.getByRole('button', { name: 'move' }))
  }

  it('applies at once, without a dialog, when no summary is needed', async () => {
    move(false)

    expect((await screen.findByRole('status')).textContent).toBe('undefined')
    expect(screen.queryByPlaceholderText(en.completePlaceholder)).toBeNull()
  })

  it('applies with the confirmed summary', async () => {
    move(true)
    fireEvent.change(screen.getByPlaceholderText(en.completePlaceholder), { target: { value: 'shipped' } })
    fireEvent.click(screen.getByRole('button', { name: en.complete }))

    expect((await screen.findByRole('status')).textContent).toBe('shipped')
  })

  it('drops the move when the user cancels', async () => {
    move(true)
    fireEvent.click(screen.getByRole('button', { name: en.cancel }))

    expect((await screen.findByRole('status')).textContent).toBe('not applied')
  })
})
