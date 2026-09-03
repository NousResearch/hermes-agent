import { describe, expect, it } from 'vitest'

import { agentEmoji, DEFAULT_AGENT_EMOJI } from './agent-emoji'

describe('agentEmoji', () => {
  it('reads the role out of the title first', () => {
    expect(agentEmoji('gary', 'Gary', 'Chief Marketing Officer')).toBe('📣')
    expect(agentEmoji('warren', 'Warren', 'CFO')).toBe('💰')
    expect(agentEmoji('sales', 'Rae', 'Chief Revenue Officer')).toBe('🤝')
    expect(agentEmoji('jarvis', 'Jarvis', 'CTO')).toBe('🛠️')
    expect(agentEmoji('ceo', 'Nova', 'CEO')).toBe('👑')
  })

  it('falls back to the name, then the profile key', () => {
    expect(agentEmoji('x', 'Jarvis-CTO')).toBe('🛠️')
    expect(agentEmoji('warren_cfo')).toBe('💰')
    expect(agentEmoji('repokeeper', 'Linus')).toBe('⚙️')
    expect(agentEmoji('balen', 'Balen-PM')).toBe('📋')
  })

  it('gives the default profile its own glyph and unknowns the robot', () => {
    expect(agentEmoji('default', 'Hermes')).toBe('🪽')
    expect(agentEmoji('zeta', 'Zeta', '')).toBe(DEFAULT_AGENT_EMOJI)
  })

  it('does not match role words inside other words', () => {
    // "pmail" must not read as PM; "opsy" must not read as ops.
    expect(agentEmoji('pmail', 'Pmail')).toBe(DEFAULT_AGENT_EMOJI)
    expect(agentEmoji('opsy', 'Opsy')).toBe(DEFAULT_AGENT_EMOJI)
  })
})
