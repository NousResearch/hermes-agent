import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { HighlightMatches } from './highlight-matches'

const marksOf = (container: HTMLElement) => Array.from(container.querySelectorAll('mark')).map(m => m.textContent)

describe('HighlightMatches with separator folding', () => {
  it('HYPHEN query marks the SPACED label (the reported defect — fails pre-fix)', () => {
    const { container } = render(<HighlightMatches query="qwen3.8-flash" text="Qwen3.8 Flash" />)

    expect(marksOf(container)).toEqual(['Qwen3.8 Flash'])
  })

  it('SPACE query marks the HYPHENATED text (model-picker polarity)', () => {
    const { container } = render(<HighlightMatches query="qwen3 8 flash" text="qwen3.8-flash" />)

    expect(marksOf(container)).toEqual(['qwen3.8-flash'])
  })

  it('folding must not shift mark indices: slices return the ORIGINAL characters', () => {
    const { container } = render(<HighlightMatches query="gpt 5" text="GPT-5 Turbo" />)

    // mark covers "GPT-5" exactly — original characters, not folded ones.
    expect(marksOf(container)).toEqual(['GPT-5'])
    expect(container.textContent).toBe('GPT-5 Turbo')
  })

  it('literal spaces in the label are marked as part of the range', () => {
    const { container } = render(<HighlightMatches query="3 8 f" text="Qwen3.8 Flash" />)

    expect(marksOf(container)).toEqual(['3.8 F'])
  })

  it('prefix queries still mark (existing contract intact)', () => {
    const { container } = render(<HighlightMatches query="qwen3.8" text="Qwen3.8 Flash" />)

    expect(marksOf(container)).toEqual(['Qwen3.8'])
  })

  it('non-matching queries render plain text', () => {
    const { container } = render(<HighlightMatches query="zzz-9" text="Qwen3.8 Flash" />)

    expect(container.querySelector('mark')).toBeNull()
    expect(container.textContent).toBe('Qwen3.8 Flash')
  })
})
