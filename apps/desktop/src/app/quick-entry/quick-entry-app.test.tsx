import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { QuickEntryApp } from './quick-entry-app'

function setup() {
  let ownerChanged = (_owner?: { connectionId: string; profile: string }) => {}

  let shown = () => {}

  let statePush = (_state: unknown) => {}

  const state = {
    owner: { connectionId: 'local', profile: 'work' },
    token: 'owner-a',
    draft: { id: 'one', text: '' },
    thoughts: [] as { id: string; text: string; createdAt: string }[]
  }

  const api = {
    readThoughts: vi.fn(async () => structuredClone(state)),
    saveThoughtDraft: vi.fn(async ({ draft }) => {
      state.draft = draft
    }),
    saveThought: vi.fn(async ({ draft }) => ({
      ...state,
      draft: { id: 'two', text: '' },
      thoughts: [{ ...draft, createdAt: '2026-01-01' }]
    })),
    submit: vi.fn(async () => true),
    dismiss: vi.fn(),
    expandThoughts: vi.fn(),
    onShown: vi.fn(fn => {
      shown = fn

      return () => {}
    }),
    onState: vi.fn(fn => {
      statePush = fn

      return () => {}
    }),
    onThoughtOwnerChanged: vi.fn(fn => {
      ownerChanged = fn

      return () => {}
    })
  }

  vi.stubGlobal('hermesDesktop', { quickEntry: api })

  return {
    api,
    state,
    changeOwner: (owner?: { connectionId: string; profile: string }) => ownerChanged(owner),
    show: () => shown(),
    connect: () => statePush({ connected: true, sessions: [] })
  }
}

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('Quick Entry local thoughts', () => {
  it('saves offline, reopens without sending, and requires an explicit Enter to hand off to chat', async () => {
    const { api, connect, show } = setup()
    render(<QuickEntryApp />)
    const input = screen.getByRole<HTMLInputElement>('textbox')
    await waitFor(() => expect(input.disabled).toBe(false))
    fireEvent.change(input, { target: { value: '  Keep this thought  ' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save thought' }))
    await screen.findByText('Saved locally')
    expect(api.saveThought).toHaveBeenCalledWith({
      token: 'owner-a',
      draft: { id: 'one', text: '  Keep this thought  ' }
    })
    expect(api.submit).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Saved thoughts' }))
    fireEvent.click(screen.getByRole('button', { name: 'Use in input' }))
    expect(input.value).toBe('  Keep this thought  ')
    expect(api.submit).not.toHaveBeenCalled()
    act(connect)
    fireEvent.keyDown(input, { key: 'Enter' })
    await waitFor(() =>
      expect(api.submit).toHaveBeenCalledWith({
        target: 'current',
        text: 'Keep this thought',
        thoughtOwnerToken: 'owner-a'
      })
    )
    act(show)
    await waitFor(() => expect(input.value).toBe('  Keep this thought  '))
    await screen.findByText(/delivery is unconfirmed/)
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(api.submit).toHaveBeenCalledTimes(1)
    cleanup()
    render(<QuickEntryApp />)
    await screen.findByText(/delivery is unconfirmed/)
    act(connect)
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter' })
    expect(api.submit).toHaveBeenCalledTimes(1)
  })

  it('retains text after a failed save and rejects a stale save result after an owner switch', async () => {
    const { api, state, changeOwner } = setup()
    api.saveThought.mockRejectedValueOnce(new Error('disk full'))
    api.saveThoughtDraft.mockRejectedValue(new Error('disk full'))
    render(<QuickEntryApp />)
    const input = screen.getByRole<HTMLInputElement>('textbox')
    await waitFor(() => expect(input.disabled).toBe(false))
    fireEvent.change(input, { target: { value: 'Do not lose this' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save thought' }))
    await screen.findByText('Could not save. Your text is still here; retry saving.')
    expect(input.value).toBe('Do not lose this')
    state.owner.profile = 'other'
    state.token = 'owner-b'
    act(changeOwner)
    await waitFor(() => expect(input.value).toBe(''))
    state.owner.profile = 'work'
    state.token = 'owner-a'
    act(changeOwner)
    await waitFor(() => expect(input.value).toBe('Do not lose this'))
    let complete!: (value: any) => void
    api.saveThought.mockImplementationOnce(
      () =>
        new Promise(resolve => {
          complete = resolve
        })
    )
    fireEvent.click(screen.getByRole('button', { name: 'Save thought' }))
    state.owner.profile = 'other'
    state.token = 'owner-b'
    act(changeOwner)
    await waitFor(() => expect(input.value).toBe(''))
    await act(async () => complete({ ...state, token: 'owner-a', draft: { id: 'old', text: 'wrong profile' } }))
    expect(input.value).toBe('')
    expect(api.submit).not.toHaveBeenCalled()
  })

  it('keeps ordinary connected chat usable when local capture cannot load', async () => {
    const { api, connect } = setup()
    api.readThoughts.mockRejectedValue(new Error('capture unavailable'))
    render(<QuickEntryApp />)
    const input = screen.getByRole<HTMLInputElement>('textbox')
    await waitFor(() => expect(input.disabled).toBe(false))
    act(connect)
    fireEvent.change(input, { target: { value: 'ordinary chat' } })
    expect(input.disabled).toBe(false)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'Save thought' }).disabled).toBe(true)
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(api.submit).toHaveBeenCalledWith({ target: 'current', text: 'ordinary chat', thoughtOwnerToken: undefined })
  })
})

it('does not restore a failed draft into a recreated profile with the same name', async () => {
  const { api, state, changeOwner } = setup()
  api.saveThoughtDraft.mockRejectedValue(new Error('disk full'))
  render(<QuickEntryApp />)
  const input = screen.getByRole<HTMLInputElement>('textbox')
  await waitFor(() => expect(input.disabled).toBe(false))
  fireEvent.change(input, { target: { value: 'old profile fragment' } })
  await screen.findByText('Could not save. Your text is still here; retry saving.')
  state.token = 'new-incarnation'
  act(() => changeOwner(state.owner))
  await waitFor(() => expect(input.value).toBe(''))
})

it('saves with the keyboard, restores focus, and appends without replacing an existing draft', async () => {
  const { api } = setup()
  render(<QuickEntryApp />)
  const input = screen.getByRole<HTMLTextAreaElement>('textbox')
  await waitFor(() => expect(input.disabled).toBe(false))
  input.focus()
  fireEvent.change(input, { target: { value: 'saved fragment' } })
  fireEvent.keyDown(input, { key: 's', metaKey: true })
  await screen.findByText('Saved locally')
  expect(document.activeElement).toBe(input)
  fireEvent.change(input, { target: { value: 'unfinished draft' } })
  fireEvent.click(screen.getByRole('button', { name: 'Saved thoughts' }))
  fireEvent.click(screen.getByRole('button', { name: 'Append to draft' }))
  expect(input.value).toBe('unfinished draft\nsaved fragment')
  expect(document.activeElement).toBe(input)
  expect(api.submit).not.toHaveBeenCalled()
})

it('keeps rejected handoffs visible and does not send when their recovery write fails', async () => {
  const { api, connect } = setup()
  api.submit.mockResolvedValue(false)
  render(<QuickEntryApp />)
  const input = screen.getByRole<HTMLTextAreaElement>('textbox')
  await waitFor(() => expect(input.disabled).toBe(false))
  act(connect)
  fireEvent.change(input, { target: { value: 'retain this' } })
  fireEvent.keyDown(input, { key: 'Enter' })
  await screen.findByText('Chat handoff was rejected. Your text is still here.')
  expect(input.value).toBe('retain this')
  expect(api.submit).toHaveBeenCalledTimes(1)
  fireEvent.click(screen.getByRole('button', { name: 'Allow another send' }))
  api.saveThoughtDraft.mockRejectedValueOnce(new Error('disk full'))
  fireEvent.keyDown(input, { key: 'Enter' })
  await screen.findByText('Could not save. Your text is still here; retry saving.')
  expect(input.value).toBe('retain this')
  expect(api.submit).toHaveBeenCalledTimes(1)
})
