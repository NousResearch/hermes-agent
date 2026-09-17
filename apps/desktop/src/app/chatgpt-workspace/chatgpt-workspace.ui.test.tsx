import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { focus, insert, navigate, submit } = vi.hoisted(() => ({
  focus: vi.fn(),
  insert: vi.fn(),
  navigate: vi.fn(),
  submit: vi.fn()
}))

vi.mock('react-router', () => ({
  useNavigate: () => navigate
}))

vi.mock('@/app/chat/composer/focus', () => ({
  requestComposerFocus: focus,
  requestComposerInsert: insert,
  requestComposerSubmit: submit
}))

import { ChatGptWorkspace } from './chatgpt-workspace'

describe('ChatGptWorkspace delegation', () => {
  beforeEach(() => {
    navigate.mockReset()
    insert.mockReset()
    focus.mockReset()
    submit.mockReset()
  })

  it('requires explicit text and opens a visible Hermes draft without submitting it', () => {
    render(<ChatGptWorkspace />)

    const delegate = screen.getByRole('button', { name: 'Delegate to Hermes' })
    expect((delegate as HTMLButtonElement).disabled).toBe(true)

    fireEvent.change(screen.getByLabelText('Approved handoff to Hermes'), {
      target: { value: 'Implement the approved plan.' }
    })
    fireEvent.click(delegate)

    expect(navigate).toHaveBeenCalledWith('/')
    expect(insert).toHaveBeenCalledWith(expect.stringContaining('Implement the approved plan.'), { target: 'main' })
    expect(focus).toHaveBeenCalledWith('main')
    expect(submit).not.toHaveBeenCalled()
  })
})
