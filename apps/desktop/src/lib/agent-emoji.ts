/**
 * One glyph per agent for the compact surfaces (HUD agent pill, launcher
 * rows) where a colour dot says nothing. Derived from the bot's ROLE — its
 * Bot Mode title first, then its name and profile key — so a renamed agent
 * keeps the right glyph and an untitled one still gets something honest.
 */

const ROLE_GLYPHS: readonly [RegExp, string][] = [
  [/\b(ceo|founder|chief executive|boss|principal)\b/, '👑'],
  [/\b(cto|chief technology|architect|tech lead)\b/, '🛠️'],
  [/\b(cfo|chief financial|finance|accountant|treasur)/, '💰'],
  [/\b(cmo|chief marketing|marketing|growth|brand|social)\b/, '📣'],
  [/\b(cro|chief revenue|sales|revenue|deals?|partnerships?)\b/, '🤝'],
  [/\b(coo|operations|ops|devops|sre|infra)\b/, '🧰'],
  [/\b(pm|product manager|product|program manager|project manager|planner)\b/, '📋'],
  [/\b(engineer|developer|dev|coder|programmer|repokeeper|maintainer|linus)\b/, '⚙️'],
  [/\b(design|designer|ux|ui)\b/, '🎨'],
  [/\b(research|researcher|analyst|scientist)\b/, '🔬'],
  [/\b(writer|content|editor|copy|author)\b/, '✍️'],
  [/\b(security|sec|guard|cso|ciso)\b/, '🛡️'],
  [/\b(support|helpdesk|concierge|assistant|ea|secretary|cipher)\b/, '🗂️'],
  [/\b(trader|trading|quant|market)\b/, '📈'],
  [/\b(legal|lawyer|counsel|compliance)\b/, '⚖️'],
  [/\b(hermes|default)\b/, '🪽']
]

export const DEFAULT_AGENT_EMOJI = '🤖'

/** Normalise "Jarvis-CTO" / "warren_cfo" / "Gary (CMO)" into matchable words. */
function words(value: string | undefined): string {
  return (value ?? '')
    .toLowerCase()
    .replace(/[-_./()[\]:,]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

/**
 * The glyph for an agent. `title` is the Bot Mode title ("Chief Marketing
 * Officer"); `name` the display name; `profile` the routing key. Each is
 * tried in that order so the most deliberate description wins.
 */
export function agentEmoji(profile: string, name?: string, title?: string): string {
  for (const source of [title, name, profile]) {
    const text = words(source)

    if (!text) {
      continue
    }

    for (const [pattern, glyph] of ROLE_GLYPHS) {
      if (pattern.test(text)) {
        return glyph
      }
    }
  }

  return DEFAULT_AGENT_EMOJI
}
