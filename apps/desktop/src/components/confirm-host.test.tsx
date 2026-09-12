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

  it('resets typed intent when a new request replaces an open prompt', async () => {
    render(<ConfirmHost />)
    let first!: Promise<boolean>
    act(() => {
      first = confirm({ title: 'First restart', typedConfirmation: 'RESTART' })
    })
    fireEvent.change(await screen.findByRole('textbox'), { target: { value: 'RESTART' } })
    let second!: Promise<boolean>
    act(() => {
      second = confirm({ title: 'Second restart', typedConfirmation: 'RESTART' })
    })
    expect(await first).toBe(false)
    await screen.findByText('Second restart')
    expect((screen.getByRole('textbox') as HTMLInputElement).value).toBe('')
    fireEvent.click(screen.getByRole('button', { name: /confirm/i }))
    expect($confirmRequest.get()).not.toBeNull()
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'RESTART' } })
    fireEvent.click(screen.getByRole('button', { name: /confirm/i }))
    expect(await second).toBe(true)
  })
})
