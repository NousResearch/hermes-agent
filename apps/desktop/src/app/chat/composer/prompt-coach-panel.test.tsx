import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $promptCoachPreviewBySession, openPromptCoachPreview } from '@/store/prompt-coach'

import { PromptCoachPanel } from './prompt-coach-panel'

const preview = {
  generatedBy: 'ai' as const,
  hasPotentialSecret: false,
  missing: ['target', 'success'] as const,
  reason: 'Missing target and success criteria',
  score: 25,
  suggestedPrompt: 'Goal:\nfix it\n\nTarget:\n[Specify target]'
}

describe('PromptCoachPanel', () => {
  beforeEach(() => {
    $promptCoachPreviewBySession.set({})
    window.localStorage.clear()
  })

  it('shows an original-versus-suggested preview and replaces without sending', () => {
    const onApply = vi.fn()
    const onSendOriginal = vi.fn()

    act(() => openPromptCoachPreview('s1', 'fix it', { ...preview, missing: [...preview.missing] }))
    render(<PromptCoachPanel onApply={onApply} onSendOriginal={onSendOriginal} sessionId="s1" />)

    expect(screen.getByText('Original')).toBeTruthy()
    expect(screen.getByText('Suggested')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Replace' }))

    expect(onApply).toHaveBeenCalledWith(preview.suggestedPrompt)
    expect(onSendOriginal).not.toHaveBeenCalled()
    expect(screen.queryByLabelText('Prompt Coach preview')).toBeNull()
  })

  it('allows editing before applying', () => {
    const onApply = vi.fn()

    act(() => openPromptCoachPreview('s1', 'fix it', { ...preview, missing: [...preview.missing] }))
    render(<PromptCoachPanel onApply={onApply} onSendOriginal={vi.fn()} sessionId="s1" />)

    fireEvent.click(screen.getByRole('button', { name: 'Edit' }))
    fireEvent.change(screen.getByLabelText('Edit improved prompt'), { target: { value: 'Goal:\nrepair the app' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply edited' }))

    expect(onApply).toHaveBeenCalledWith('Goal:\nrepair the app')
  })

  it('sends the untouched original only after an explicit click', () => {
    const onApply = vi.fn()
    const onSendOriginal = vi.fn()

    act(() => openPromptCoachPreview('s1', 'fix it', { ...preview, missing: [...preview.missing] }))
    render(<PromptCoachPanel onApply={onApply} onSendOriginal={onSendOriginal} sessionId="s1" />)

    fireEvent.click(screen.getByRole('button', { name: 'Send original' }))

    expect(onApply).not.toHaveBeenCalled()
    expect(onSendOriginal).toHaveBeenCalledOnce()
  })
})
