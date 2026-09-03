import { type OneShotRequest, requestOneShot } from '@/lib/oneshot'

import {
  buildPromptCoachSuggestion,
  type PromptCoachAnalysis,
  type PromptCoachField,
  redactPromptSecrets
} from './prompt-coach'

const DEFAULT_QUESTIONS: Record<PromptCoachField, string> = {
  constraints: 'What must be preserved, avoided, or approved before making changes?',
  success: 'What observable result or verification will prove the task is complete?',
  target: 'What exact project, workspace, file, component, service, or item should this apply to?'
}

const PROMPT_COACH_INSTRUCTIONS = `You are Hermes Prompt Coach. Help make a task executable before it reaches the main agent.

Rules:
- Do NOT correct, comment on, or score spelling, grammar, capitalization, or punctuation.
- Do NOT rewrite the user's request and do NOT change its intent.
- Do NOT invent project names, files, constraints, facts, or acceptance criteria.
- Generate only concise questions for the missing task-readiness fields supplied to you.
- Return strict JSON only, with keys "target", "constraints", and "success".
- Each value must be one short question. Use an empty string for fields that are not missing.
- Never repeat or reconstruct redacted secrets.`

type PromptCoachRequester = (request: OneShotRequest) => Promise<string>

function parseQuestions(raw: string): Partial<Record<PromptCoachField, string>> {
  const json = raw.match(/\{[\s\S]*\}/)?.[0]

  if (!json) {
    return {}
  }

  try {
    const parsed = JSON.parse(json) as Record<string, unknown>
    const questions: Partial<Record<PromptCoachField, string>> = {}

    for (const field of ['target', 'constraints', 'success'] as const) {
      if (typeof parsed[field] === 'string' && parsed[field].trim()) {
        questions[field] = parsed[field].trim().replace(/\s+/g, ' ')
      }
    }

    return questions
  } catch {
    return {}
  }
}

function buildAISuggestion(
  original: string,
  analysis: PromptCoachAnalysis,
  questions: Partial<Record<PromptCoachField, string>>
): string {
  const { found, redacted } = redactPromptSecrets(original.trim())
  const sections = [`Request (kept exactly as written):\n${redacted}`]

  if (analysis.missing.includes('target')) {
    sections.push(`Target:\n[${questions.target ?? DEFAULT_QUESTIONS.target}]`)
  }

  if (analysis.missing.includes('constraints')) {
    sections.push(`Constraints:\n- [${questions.constraints ?? DEFAULT_QUESTIONS.constraints}]`)
  }

  if (analysis.missing.includes('success')) {
    sections.push(
      `Done when:\n- [${questions.success ?? DEFAULT_QUESTIONS.success}]\n- Report what changed, what was verified, and anything blocked.`
    )
  }

  if (found) {
    sections.push(
      'Security:\n- A possible secret was removed from this suggested copy. Supply credentials only through an approved secure flow.'
    )
  }

  return sections.join('\n\n')
}

/**
 * Ask the model already attached to this Hermes session for context-aware
 * readiness questions. The original task is inserted locally and verbatim;
 * the model never gets authority to rewrite it. Any failure returns the local
 * deterministic analysis so Prompt Coach cannot block chat.
 */
export async function enhancePromptCoachWithAI(
  original: string,
  analysis: PromptCoachAnalysis,
  sessionId?: null | string,
  requester: PromptCoachRequester = requestOneShot
): Promise<PromptCoachAnalysis> {
  const { redacted } = redactPromptSecrets(original.trim())

  try {
    const response = await requester({
      input: JSON.stringify({ draft: redacted, missing_fields: analysis.missing }),
      instructions: PROMPT_COACH_INSTRUCTIONS,
      maxTokens: 220,
      sessionId,
      task: 'prompt_coach',
      temperature: 0.1
    })

    const questions = parseQuestions(response)

    return {
      ...analysis,
      generatedBy: 'ai',
      suggestedPrompt: buildAISuggestion(original, analysis, questions)
    }
  } catch {
    const fallback = buildPromptCoachSuggestion(original, analysis.missing)

    return {
      ...analysis,
      generatedBy: 'local',
      hasPotentialSecret: fallback.hasPotentialSecret,
      suggestedPrompt: fallback.suggestedPrompt
    }
  }
}
