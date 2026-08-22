import type { ProfileScope } from '@/api/client'
import { profileScopeKey } from '@/api/client'

// Bot-level skill assignment (#88973): an inverted view over the existing
// per-profile skill management. Pure helpers only — the Skills page feeds it
// scope-selector options and per-bot enabled-skill reads; these functions own
// the decoding and the skill→bots mapping.

/** One bot column in the Skills page's assignment matrix. */
export interface SkillBotColumn {
  /** Stable id — profileScopeKey(scope). Cache key + React key. */
  id: string
  /** Display name ("researcher", "Hermes (default)", "inbox-bot — Homelab"). */
  label: string
  /** Where this bot's reads/writes route. */
  scope: ProfileScope
}

/** Decode a Capabilities scope-selector option value back into its routing
 *  scope. Option values are bare profile names on the legacy path and
 *  `connectionId::profile` roster picks on multi-connection desktops — exactly
 *  what SkillsView.changeScope consumes, so matrix writes route to the same
 *  backend a selector pick would. */
export function scopeFromOptionValue(value: string): ProfileScope {
  const sep = value.indexOf('::')

  return sep >= 0 ? { connectionId: value.slice(0, sep), profile: value.slice(sep + 2) } : value
}

/** Build a matrix column from a selector option. The id is derived from the
 *  DECODED scope via profileScopeKey — never the raw option string — so cache
 *  keys stay identical to the ones the rest of the Capabilities surface uses
 *  (a roster pick for the local pool must share the bare-profile key). */
export function skillBotColumn(optionValue: string, label: string): SkillBotColumn {
  const scope = scopeFromOptionValue(optionValue)

  return { id: profileScopeKey(scope), label, scope }
}

/**
 * Invert per-bot enabled-skill sets into a per-skill assignment row.
 *
 * @param skillNames The Skills page's own rows (the current scope's list).
 *   Cross-profile-only skills are intentionally absent — this page manages
 *   THIS scope's skills and shows where else each one is assigned.
 * @param bots Columns to map, in display order.
 * @param enabledSkillsByBotId Bot id → that profile's enabled-skill names.
 *   A missing entry means the read hasn't landed yet.
 * @returns skill name → bot id → enabled, with `null` marking a column whose
 *   list is still loading or failed (rendered as a dimmed chip, not a guess).
 */
export function buildSkillBotMatrix(
  skillNames: readonly string[],
  bots: readonly SkillBotColumn[],
  enabledSkillsByBotId: ReadonlyMap<string, ReadonlySet<string>>
): Map<string, Map<string, boolean | null>> {
  const matrix = new Map<string, Map<string, boolean | null>>()

  if (bots.length === 0) {
    return matrix
  }

  for (const name of skillNames) {
    const row = new Map<string, boolean | null>()

    for (const bot of bots) {
      const enabled = enabledSkillsByBotId.get(bot.id)

      row.set(bot.id, enabled ? enabled.has(name) : null)
    }

    matrix.set(name, row)
  }

  return matrix
}
