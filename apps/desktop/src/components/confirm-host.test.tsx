import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $confirmRequest, confirm } from '@/store/confirm'

import { ConfirmHost } from './confirm-host'

afterEach(() => {
  cleanup()
  $confirmRequest.set(null)
})

/** Open a confirm and wait for the dialog, without awaiting the answer. */
async function ask(title = 'Delete it?') {
  let answer: boolean | undefined
  const pending = confirm({ title }).then(ok => (answer = ok))

  const dialog = await screen.findByRole('dialog')

  return { dialog, pending, read: () => answer }
}

describe('confirm()', () => {
  it('renders nothing until something asks', () => {
    render(<ConfirmHost />)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('resolves true when confirmed and false when cancelled', async () => {
    render(<ConfirmHost />)

    const yes = await ask()
    fireEvent.click(screen.getByRole('button', { name: /confirm/i }))
    await yes.pending
    expect(yes.read()).toBe(true)

    const no = await ask()
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
    await no.pending
    expect(no.read()).toBe(false)
  })

  it('confirms on Enter from wherever focus landed', async () => {
    render(<ConfirmHost />)

    const { dialog, pending, read } = await ask()

    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true))
    // eslint-disable-next-line no-restricted-globals -- asserting real focus requires the live document
    fireEvent.keyDown(document.activeElement!, { key: 'Enter' })

    await pending
    expect(read()).toBe(true)
  })

  it('answers no to Escape', async () => {
    render(<ConfirmHost />)

    const { pending, read } = await ask()
    fireEvent.keyDown(window.document, { key: 'Escape' })

    await pending
    expect(read()).toBe(false)
  })

  it('keeps async actions open through progress, inline failure, retry and completion', async () => {
    render(<ConfirmHost />)
    let fail!: (error: Error) => void
    let finish!: () => void
    let attempts = 0
    let answer: boolean | undefined

    const pending = confirm({
      title: 'Install “pdf”?',
      confirmLabel: 'Install',
      busyLabel: 'Installing…',
      doneLabel: 'Installed',
      details: [{ label: 'Source', value: 'official/productivity/pdf' }],
      onConfirm: () =>
        new Promise<void>((resolve, reject) => {
          attempts += 1
          finish = resolve
          fail = reject
        })
    }).then(value => {
      answer = value
    })

    expect(await screen.findByText('official/productivity/pdf')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    expect(await screen.findByRole('button', { name: 'Installing…' })).toBeTruthy()
    expect(answer).toBeUndefined()
    expect(await confirm({ title: 'Must not interrupt' })).toBe(false)
    await act(async () => fail(new Error('Network unavailable')))
    expect(await screen.findByText('Network unavailable')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Install' }))
    await act(async () => finish())
    expect(await screen.findByRole('button', { name: 'Installed' })).toBeTruthy()
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    await pending
    expect(answer).toBe(true)
    expect(attempts).toBe(2)
  })

  it('supersedes an open request, answering the one it replaces no', async () => {
    render(<ConfirmHost />)

    const first = await ask('First?')
    const second = await act(async () => ask('Second?'))

    await first.pending
    expect(first.read()).toBe(false)
    expect(screen.getByText('Second?')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /confirm/i }))
    await second.pending
    expect(second.read()).toBe(true)
  })
})
