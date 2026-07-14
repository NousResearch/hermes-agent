export type AgentLane = 'claude' | 'codex' | 'gemini'

export interface ParsedAgentPrompt {
  lane: AgentLane
  prompt: string
}

const AGENT_TAG_RE = /^@(claude|codex|gemini)(?:\s+([\s\S]*))?$/i

export const AGENT_LANES: readonly AgentLane[] = ['claude', 'codex', 'gemini']

export function parseAgentTagPrompt(text: string): ParsedAgentPrompt | null {
  const match = AGENT_TAG_RE.exec(text.trim())

  if (!match) {
    return null
  }

  return {
    lane: match[1]!.toLowerCase() as AgentLane,
    prompt: (match[2] ?? '').trim()
  }
}

export function agentLaneTitle(lane: AgentLane): string {
  switch (lane) {
    case 'claude':
      return 'Claude'
    case 'codex':
      return 'Codex'
    case 'gemini':
      return 'Gemini'
  }
}
