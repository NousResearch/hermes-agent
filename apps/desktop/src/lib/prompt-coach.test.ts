import { describe, expect, it } from 'vitest'

import { analyzePromptDraft, redactPromptSecrets } from './prompt-coach'

describe('Prompt Coach local analysis', () => {
  it('offers help for ambiguous execution prompts and names the missing fields', () => {
    const analysis = analyzePromptDraft('fix it and make it better')

    expect(analysis).not.toBeNull()
    expect(analysis?.missing).toEqual(['target', 'constraints', 'success'])
    expect(analysis?.reason).toBe('Missing target, constraints and success criteria')
    expect(analysis?.suggestedPrompt).toContain('Goal:\nfix it and make it better')
    expect(analysis?.suggestedPrompt).toContain(
      '[Specify the project, workspace, file, component, or service to change.]'
    )
  })

  it('keeps concrete small edits, clear questions, commands, and structured prompts quiet', () => {
    expect(analyzePromptDraft('fix typo in README.md')).toBeNull()
    expect(analyzePromptDraft('How do I reset a Git branch safely?')).toBeNull()
    expect(analyzePromptDraft('/status')).toBeNull()
    expect(analyzePromptDraft('Goal:\nBuild the widget\n\nDone when:\nTests pass')).toBeNull()
  })

  it('catches ambiguous short requests without treating typing mistakes as the problem', () => {
    expect(analyzePromptDraft('givme that')?.missing).toEqual(['target', 'constraints', 'success'])
    expect(analyzePromptDraft('is it done?')?.reason).toBe('Missing target, constraints and success criteria')
    expect(analyzePromptDraft('hwo to clean it')?.reason).toBe('Missing target, constraints and success criteria')
    expect(analyzePromptDraft('hwo to clean it')?.suggestedPrompt).toContain('Goal:\nhwo to clean it')
  })

  it('stays quiet for grammar, spelling, capitalization, and punctuation alone', () => {
    expect(analyzePromptDraft('can yo tell me somehting about mars')).toBeNull()
    expect(analyzePromptDraft('whta is mars')).toBeNull()
    expect(analyzePromptDraft('tell me about mars')).toBeNull()
    expect(analyzePromptDraft('Can you tell me something about Mars?')).toBeNull()
  })

  it('preserves stated intent and adds placeholders instead of invented facts', () => {
    const original = 'Build a prompt widget in the desktop app'
    const analysis = analyzePromptDraft(original)

    expect(analysis?.suggestedPrompt).toContain(`Goal:\n${original}`)
    expect(analysis?.suggestedPrompt).not.toMatch(/D:\\|team-hermes-desktop|React|Electron/)
    expect(analysis?.missing).toEqual(['constraints', 'success'])
  })

  it('redacts likely credentials from the generated suggestion', () => {
    const token = `sk-${'a'.repeat(24)}`
    const analysis = analyzePromptDraft(`build this integration using ${token}`)

    expect(analysis?.hasPotentialSecret).toBe(true)
    expect(analysis?.suggestedPrompt).toContain('[REDACTED SECRET]')
    expect(analysis?.suggestedPrompt).not.toContain(token)
  })

  it('redacts assignment and bearer credential shapes without changing surrounding text', () => {
    expect(redactPromptSecrets('token=abc123xyz build it').redacted).toBe('token=[REDACTED SECRET] build it')
    expect(redactPromptSecrets('Authorization: Bearer abc.def.ghi').redacted).toBe(
      'Authorization: Bearer [REDACTED SECRET]'
    )
  })
})
