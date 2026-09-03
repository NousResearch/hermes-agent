export type PromptCoachField = 'constraints' | 'success' | 'target'

export interface PromptCoachAnalysis {
  generatedBy: 'ai' | 'local' | 'pending'
  hasPotentialSecret: boolean
  missing: PromptCoachField[]
  reason: string
  score: number
  suggestedPrompt: string
}

const ACTION_RE =
  /\b(?:add|automate|build|change|clean|configure|create|debug|deploy|design|diagnose|fix|implement|improve|integrate|make|migrate|refactor|remove|repair|restart|set\s*up|test|update|verify|wire)\b/i

const COMPLEX_ACTION_RE =
  /\b(?:automate|build|clean|configure|deploy|design|implement|integrate|migrate|refactor|repair|set\s*up|wire)\b/i

const AMBIGUOUS_REFERENCE_RE = /\b(?:above|it|something|that|the issue|the problem|this)\b/i

const SHORT_REQUEST_RE =
  /^(?:can|check|could|do|fix|get|give|givme|how|is|look|make|please|run|send|show|tell|try|use|was|what|where|why|would)\b/i

const QUESTION_START_RE = /^(?:can|could|did|do|does|how|is|should|tell|was|what|when|where|which|who|why|would)\b/i

const TARGET_RE =
  /(?:[A-Za-z]:\\|\/[^\s]+|\b[\w.-]+\.(?:c|cpp|css|go|html|js|json|jsx|md|mjs|py|rs|ts|tsx|yaml|yml)\b|\b(?:api|app|application|component|desktop|file|folder|gateway|module|plugin|project|repo|repository|service|site|widget|workspace)\b)/i

const CONSTRAINT_RE = /\b(?:ask before|avoid|do not|don't|keep|must|never|only|preserve|retain|without)\b/i

const SUCCESS_RE =
  /\b(?:acceptance|confirm|done when|evidence|make sure|passes?|proof|success|test|tested|verify|verified|works?|working)\b/i

const STRUCTURED_RE = /^(?:goal|context|target|requirements?|constraints?|done when|acceptance|report):/gim

const SECRET_PATTERNS = [
  /\b(?:sk|ghp|github_pat|xox[baprs])-[-_A-Za-z0-9]{12,}\b/g,
  /\b((?:api[_ -]?key|password|secret|token)\s*[:=]\s*)([^\s,;]+)/gi,
  /\b(Bearer\s+)[-._~+/A-Za-z0-9]+=*/gi
] as const

export function redactPromptSecrets(text: string): { redacted: string; found: boolean } {
  let found = false
  let redacted = text

  redacted = redacted.replace(SECRET_PATTERNS[0], () => {
    found = true

    return '[REDACTED SECRET]'
  })
  redacted = redacted.replace(SECRET_PATTERNS[1], (_match, prefix: string) => {
    found = true

    return `${prefix}[REDACTED SECRET]`
  })

  redacted = redacted.replace(SECRET_PATTERNS[2], (_match, prefix: string) => {
    found = true

    return `${prefix}[REDACTED SECRET]`
  })

  return { found, redacted }
}

function fieldLabel(field: PromptCoachField): string {
  switch (field) {
    case 'target':
      return 'target'

    case 'constraints':
      return 'constraints'

    case 'success':
      return 'success criteria'
  }
}

export function buildPromptCoachSuggestion(
  original: string,
  missing: readonly PromptCoachField[]
): {
  hasPotentialSecret: boolean
  suggestedPrompt: string
} {
  const { found, redacted } = redactPromptSecrets(original.trim())
  const sections = [`Goal:\n${redacted}`]

  if (missing.includes('target')) {
    sections.push('Target:\n[Specify the project, workspace, file, component, or service to change.]')
  }

  if (missing.includes('constraints')) {
    sections.push(
      'Constraints:\n- [Specify behavior that must be preserved.]\n- [Specify actions that require approval or must not be taken.]'
    )
  }

  if (missing.includes('success')) {
    sections.push(
      'Done when:\n- [Specify observable success evidence.]\n- Report what changed, what was verified, and anything blocked.'
    )
  }

  if (found) {
    sections.push(
      'Security:\n- A possible secret was removed from this suggested copy. Supply credentials only through an approved secure flow.'
    )
  }

  return { hasPotentialSecret: found, suggestedPrompt: sections.join('\n\n') }
}

/**
 * High-precision, local readiness check. It deliberately stands down for
 * slash commands, already-structured prompts, and small concrete edits. Short
 * requests with an ambiguous reference are intentionally included: these are
 * the prompts most likely to create an avoidable clarification round trip.
 */
export function analyzePromptDraft(text: string): PromptCoachAnalysis | null {
  const trimmed = text.trim()

  if (trimmed.length < 4 || trimmed.startsWith('/') || (trimmed.match(STRUCTURED_RE)?.length ?? 0) >= 2) {
    return null
  }

  // Prompt Coach evaluates task readiness only. Spelling, grammar and
  // punctuation are deliberately ignored and the user's words stay verbatim.
  const analyzedText = trimmed
  const hasAction = ACTION_RE.test(analyzedText)
  const hasAmbiguousReference = AMBIGUOUS_REFERENCE_RE.test(analyzedText)

  const clearInformationalQuestion =
    QUESTION_START_RE.test(analyzedText) &&
    (!hasAmbiguousReference || /\babout\s+(?!it\b|that\b|this\b)\S+/i.test(analyzedText))

  if (clearInformationalQuestion) {
    return null
  }

  const isShortAmbiguousRequest =
    hasAmbiguousReference && analyzedText.split(/\s+/).length <= 12 && SHORT_REQUEST_RE.test(analyzedText)

  if (!hasAction && !isShortAmbiguousRequest) {
    return null
  }

  const hasTarget = TARGET_RE.test(analyzedText)
  const hasConstraints = CONSTRAINT_RE.test(analyzedText)
  const hasSuccess = SUCCESS_RE.test(analyzedText)
  const missing: PromptCoachField[] = []

  if (!hasTarget) {
    missing.push('target')
  }

  if (!hasConstraints) {
    missing.push('constraints')
  }

  if (!hasSuccess) {
    missing.push('success')
  }

  const deservesCoach =
    missing.length >= 2 && (COMPLEX_ACTION_RE.test(analyzedText) || hasAmbiguousReference || isShortAmbiguousRequest)

  if (!deservesCoach) {
    return null
  }

  const score = 25 + (hasTarget ? 25 : 0) + (hasConstraints ? 20 : 0) + (hasSuccess ? 30 : 0)
  const suggestion = buildPromptCoachSuggestion(analyzedText, missing)
  const labels = missing.map(fieldLabel)

  const reason =
    labels.length === 1 ? `Missing ${labels[0]}` : `Missing ${labels.slice(0, -1).join(', ')} and ${labels.at(-1)}`

  return {
    generatedBy: 'local',
    hasPotentialSecret: suggestion.hasPotentialSecret,
    missing,
    reason,
    score,
    suggestedPrompt: suggestion.suggestedPrompt
  }
}
