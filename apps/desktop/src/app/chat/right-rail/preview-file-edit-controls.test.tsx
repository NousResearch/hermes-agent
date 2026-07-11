import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { EditControls } from './preview-file'

afterEach(cleanup)

describe('file editor controls', () => {
  it('exposes a visible Find and Replace action', () => {
    const onFindReplace = vi.fn()

    render(
      <EditControls dirty={false} onCancel={vi.fn()} onFindReplace={onFindReplace} onSave={vi.fn()} saving={false} />
    )

    const button = screen.getByRole('button', { name: 'Find and Replace' })

    fireEvent.click(button)
    expect(onFindReplace).toHaveBeenCalledTimes(1)
  })
})
