import { act, fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { PreviewTextCommentCard } from './preview-text-comment-card'

const recorder = vi.hoisted(() => ({
  dictate: vi.fn(),
  onTranscript: (_text: string) => {},
  voiceStatus: 'idle',
  voiceActivityState: { partial: '' }
}))

vi.mock('../composer/hooks/use-voice-recorder', () => ({
  useVoiceRecorder: (options: { onTranscript: (text: string) => void }) => {
    recorder.onTranscript = options.onTranscript

    return recorder
  }
}))

describe('PreviewTextCommentCard', () => {
  const props = {
    onCancel: vi.fn(),
    onSubmit: vi.fn(() => false),
    paneHeight: 320,
    paneWidth: 320,
    rect: { height: 20, width: 150, x: 150, y: 290 },
    selectedText: 'Selected preview text'
  }

  it('requires a comment and retains it if the destination cannot accept the edit', () => {
    render(<PreviewTextCommentCard {...props} />)
    const field = screen.getByRole('textbox', { name: 'Add comment…' })
    const send = screen.getByRole('button', { name: 'Send edit' })
    expect(send.hasAttribute('disabled')).toBe(true)
    expect(window.document.activeElement).toBe(field)
    fireEvent.change(field, { target: { value: '  Make this clearer  ' } })
    fireEvent.click(send)
    expect(props.onSubmit).toHaveBeenCalledWith('Make this clearer')
    expect((field as HTMLTextAreaElement).value).toBe('  Make this clearer  ')
    expect(screen.getByRole('alert').textContent).toContain('Your comment has not been sent')
    expect(props.onCancel).not.toHaveBeenCalled()
  })

  it('dismisses once on Escape without handing the gesture to the underlying pane', () => {
    const underlyingEscape = vi.fn()
    const onCancel = vi.fn()
    render(
      <div onKeyDown={underlyingEscape}>
        <PreviewTextCommentCard {...props} onCancel={onCancel} />
      </div>
    )
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Escape' })
    expect(onCancel).toHaveBeenCalledOnce()
    expect(underlyingEscape).not.toHaveBeenCalled()
  })

  it('dismisses on an outside pointer gesture without consuming the outside action', async () => {
    const onCancel = vi.fn()
    const outsideClick = vi.fn()
    render(
      <>
        <button onClick={outsideClick}>Outside</button>
        <PreviewTextCommentCard {...props} onCancel={onCancel} />
      </>
    )
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0))
    })
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Outside' }))
    fireEvent.pointerUp(screen.getByRole('button', { name: 'Outside' }))
    fireEvent.click(screen.getByRole('button', { name: 'Outside' }))
    expect(onCancel).toHaveBeenCalledOnce()
    expect(outsideClick).toHaveBeenCalledOnce()
  })
})

it('appends dictation to the comment without submitting it', () => {
  const onSubmit = vi.fn(() => true)
  render(
    <PreviewTextCommentCard
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      owner={{ connectionId: 'local', profile: 'writer' }}
      paneHeight={400}
      paneWidth={400}
      rect={{ x: 0, y: 0, width: 50, height: 20 }}
      selectedText="Heading"
    />
  )
  const field = screen.getByRole('textbox') as HTMLTextAreaElement
  fireEvent.change(field, { target: { value: 'Please' } })
  fireEvent.click(screen.getByRole('button', { name: 'Voice dictation' }))
  expect(recorder.dictate).toHaveBeenCalledOnce()
  act(() => recorder.onTranscript('shorten this'))
  expect(field.value).toBe('Please shorten this')
  expect(onSubmit).not.toHaveBeenCalled()
})
