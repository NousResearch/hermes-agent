import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $screenTutor } from '@/store/screen-tutor'

import { ScreenGuideCard } from './screen-guide-card'

const guide = {
  id: 'excel-pivot',
  instruction: 'Open the Insert tab',
  step: 1,
  successCheck: 'The Insert ribbon is visible',
  title: 'Build a pivot table',
  total: 4
}

describe('ScreenGuideCard', () => {
  beforeEach(() => {
    act(() => {
      $screenTutor.set({
        armedTarget: null,
        error: null,
        overlay: { count: 2, frozen: true, guide, visible: true },
        status: 'idle'
      })
    })
  })

  it('waits for an explicit check before submitting the verification turn', async () => {
    const onSubmit = vi.fn().mockResolvedValue(true)
    render(<ScreenGuideCard busy={false} disabled={false} onSubmit={onSubmit} target="main" />)

    expect(screen.getByText('Open the Insert tab')).toBeTruthy()
    expect(onSubmit).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: /check my step/i }))

    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce())
    expect(onSubmit.mock.calls[0][0]).toContain('Advance only if it is visibly satisfied')
    expect(onSubmit.mock.calls[0][1]).toEqual({ displayText: 'Check step 1' })
    expect($screenTutor.get().armedTarget).toBe('main')
  })

  it('stops locally without sending another agent turn', () => {
    const onSubmit = vi.fn()
    render(<ScreenGuideCard busy={false} disabled={false} onSubmit={onSubmit} target="main" />)

    fireEvent.click(screen.getByRole('button', { name: 'Stop guide' }))

    expect(onSubmit).not.toHaveBeenCalled()
    expect($screenTutor.get().overlay.visible).toBe(false)
  })
})
