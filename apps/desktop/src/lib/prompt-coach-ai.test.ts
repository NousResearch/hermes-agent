import { describe, expect, it, vi } from 'vitest'

import { analyzePromptDraft } from './prompt-coach'
import { enhancePromptCoachWithAI } from './prompt-coach-ai'

describe('AI Prompt Coach enhancement', () => {
  it('uses the active Hermes session for questions while preserving the original wording', async () => {
    const requester = vi.fn().mockResolvedValue(
      JSON.stringify({
        constraints: 'What should not be deleted or changed while cleaning?',
        success: 'How should Hermes verify that cleaning completed safely?',
        target: 'What exactly does "it" refer to?'
      })
    )

    const analysis = analyzePromptDraft('hwo to clean it')!
    const enhanced = await enhancePromptCoachWithAI('hwo to clean it', analysis, 'codex-session', requester)

    expect(requester).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: 'codex-session', task: 'prompt_coach', temperature: 0.1 })
    )
    expect(enhanced.generatedBy).toBe('ai')
    expect(enhanced.suggestedPrompt).toContain('Request (kept exactly as written):\nhwo to clean it')
    expect(enhanced.suggestedPrompt).toContain('[What exactly does "it" refer to?]')
    expect(enhanced.suggestedPrompt).not.toContain('How to clean it?')
  })

  it('redacts secrets before the AI call and in the suggested copy', async () => {
    const requester = vi.fn().mockResolvedValue('{}')
    const secret = `sk-${'a'.repeat(24)}`
    const original = `build this using ${secret}`
    const enhanced = await enhancePromptCoachWithAI(original, analyzePromptDraft(original)!, null, requester)

    expect(requester.mock.calls[0]?.[0].input).toContain('[REDACTED SECRET]')
    expect(requester.mock.calls[0]?.[0].input).not.toContain(secret)
    expect(enhanced.suggestedPrompt).not.toContain(secret)
  })

  it('falls back locally when Hermes AI is unavailable', async () => {
    const requester = vi.fn().mockRejectedValue(new Error('offline'))
    const analysis = analyzePromptDraft('fix it and make it better')!
    const enhanced = await enhancePromptCoachWithAI('fix it and make it better', analysis, 'session', requester)

    expect(enhanced.generatedBy).toBe('local')
    expect(enhanced.suggestedPrompt).toContain('Goal:\nfix it and make it better')
  })
})
