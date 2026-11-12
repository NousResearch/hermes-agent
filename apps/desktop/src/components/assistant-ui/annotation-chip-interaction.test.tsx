import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { formatResponseAnnotations, onResponseAnnotationsOpen } from '@/lib/response-annotations'

import { DirectiveContent } from './directive-text'

afterEach(cleanup)

it('opens a sent annotation preview without activating the containing message editor', () => {
  const edit = vi.fn()
  const prepareEdit = vi.fn()
  const preview = vi.fn()
  const unsubscribe = onResponseAnnotationsOpen(preview)

  const text = formatResponseAnnotations({
    type: 'response-annotations', version: 1, label: '1 annotation',
    annotations: [{ id: 'feedback', text: 'Launch', comment: 'Make a checklist.' }]
  })

  try {
    render(<div onClick={edit} onPointerDown={prepareEdit}><DirectiveContent text={text} /></div>)
    const chip = screen.getByRole('button', { name: '1 annotation' })
    fireEvent.pointerDown(chip)
    fireEvent.click(chip)
    expect(preview).toHaveBeenCalledWith(expect.objectContaining({ editable: false }))
    expect(prepareEdit).not.toHaveBeenCalled()
    expect(edit).not.toHaveBeenCalled()
  } finally { unsubscribe() }
})
