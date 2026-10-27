import { JSON_SCHEMA, load } from 'js-yaml'

/** Display only: soft links never activate skills or change curator policy. */
export function skillRelations(content: string): string[] {
  const block = /^\uFEFF?---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(content)
  if (!block) return []

  try {
    const data = load(block[1], { schema: JSON_SCHEMA }) as Record<string, unknown> | null
    if (!data || typeof data !== 'object') return []
    const metadata = data.metadata as { hermes?: { related_skills?: unknown } } | undefined
    // Match skill_view's namespaced-first preference.
    const raw = metadata?.hermes?.related_skills || data.related_skills
    const values = typeof raw === 'string' ? raw.replace(/^\[|\]$/g, '').split(',') : raw
    if (!Array.isArray(values)) return []
    return [
      ...new Set(
        values
          .filter((value): value is string => typeof value === 'string')
          .map(value => value.trim())
          .filter(Boolean)
      )
    ]
  } catch {
    // Malformed frontmatter must not hide the readable skill body.
    return []
  }
}
