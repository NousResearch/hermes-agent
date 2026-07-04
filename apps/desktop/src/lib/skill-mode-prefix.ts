export interface UltraModeFlags {
  agentFleet?: boolean
  ultrawork: boolean
  ultraresearch: boolean
}

export function ultraModeSkills(modes: UltraModeFlags): string[] {
  return [
    modes.agentFleet ? 'hq-agent-collaboration' : '',
    modes.ultrawork ? 'ulw' : '',
    modes.ultraresearch ? 'ultraresearch' : ''
  ].filter(Boolean)
}

const EXPLICIT_SKILL_MODE_RE = /^(?:agent\s+fleet|hq-agent-collaboration|ulw|ultrawork|ulr|ultraresearch)(?:\s|:|$)/i

/**
 * Applies desktop status-bar ultra-mode toggles to the prompt text that is sent
 * to the backend. The visible optimistic user message stays unchanged; this is
 * only a transport prefix so Hermes' normal skill trigger path can load the
 * corresponding mode without mutating the running system prompt.
 */
export function applyUltraModePrefix(text: string, modes: UltraModeFlags): string {
  const body = text.trim()

  if (!body || EXPLICIT_SKILL_MODE_RE.test(body)) {
    return text
  }

  const prefixes = [
    modes.agentFleet ? 'agent fleet' : '',
    modes.ultrawork ? 'ulw' : '',
    modes.ultraresearch ? 'ultraresearch' : ''
  ].filter(Boolean)

  if (!prefixes.length) {
    return text
  }

  return `${prefixes.join(' ')} ${body}`
}
