import { describe, expect, it } from 'vitest'

import { agentLaneTitle, parseAgentTagPrompt } from './agent-lanes'

describe('agent lane tags', () => {
  it('parses leading Claude, Codex, and Gemini mentions', () => {
    expect(parseAgentTagPrompt('@claude fix this')).toEqual({ lane: 'claude', prompt: 'fix this' })
    expect(parseAgentTagPrompt('@codex\nreview diff')).toEqual({ lane: 'codex', prompt: 'review diff' })
    expect(parseAgentTagPrompt('@Gemini explain repo')).toEqual({ lane: 'gemini', prompt: 'explain repo' })
  })

  it('ignores non-agent refs', () => {
    expect(parseAgentTagPrompt('@file:README.md')).toBeNull()
    expect(parseAgentTagPrompt('hello @gemini')).toBeNull()
  })

  it('renders lane titles', () => {
    expect(agentLaneTitle('claude')).toBe('Claude')
    expect(agentLaneTitle('codex')).toBe('Codex')
    expect(agentLaneTitle('gemini')).toBe('Gemini')
  })
})
