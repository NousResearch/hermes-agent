import type { McpCatalogEntry } from '@/types/hermes'

export interface OnboardingInterests {
  apps?: readonly string[]
  context?: string
}

export interface OnboardingRecommendation {
  name: string
  description: string
  examples: string[]
  detectedApps: string[]
  readiness: 'configured_unverified' | 'setup_required'
  requiresApp: boolean
  authType: string
  setupAction: 'install' | 'enable' | null
}

const words = (text: string): string =>
  text
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean)
    .join(' ')

/** Rank evidence, not a fixed list of products. The model derives outcomes from catalog descriptions; curated examples are optional. */
export function onboardingRecommendations(
  entries: readonly McpCatalogEntry[],
  { apps = [], context = '' }: OnboardingInterests = {}
): OnboardingRecommendation[] {
  const selected = new Set(apps.map(words))
  const subject = ` ${words(context)} `

  const candidates = entries.flatMap(entry => {
    const examples = [...new Set(entry.suggest?.examples ?? [])].filter(text => text.trim())

    const terms = [entry.name, ...(entry.suggest?.keywords ?? []), ...(entry.suggest?.applications ?? [])]
      .map(words)
      .filter(Boolean)

    const preferred = terms.some(term => selected.has(term))
    const topical = terms.some(term => subject.includes(` ${term} `))
    const detectedApps = entry.detected_apps ?? []
    const configured = entry.installed && entry.enabled

    // An explicit task can request a disabled integration; mere discovery must not undo that choice.
    if (entry.installed && !entry.enabled && !preferred && !topical) {
      return []
    }

    if (entry.suggest?.requires_app && !detectedApps.length && !entry.installed) {
      return []
    }

    if (!configured && !detectedApps.length && !preferred && !topical) {
      return []
    }

    const recommendation: OnboardingRecommendation = {
      name: entry.name,
      description: entry.description,
      examples: examples.slice(0, preferred || topical ? 3 : 1),
      detectedApps,
      readiness: configured ? 'configured_unverified' : 'setup_required',
      requiresApp: entry.suggest?.requires_app === true,
      authType: entry.auth_type,
      setupAction: !entry.installed ? 'install' : !entry.enabled ? 'enable' : null
    }

    return [{ recommendation, topical, preferred, configured, detected: detectedApps.length > 0 }]
  })

  return candidates
    .sort(
      (a, b) =>
        Number(b.topical) - Number(a.topical) ||
        Number(b.preferred) - Number(a.preferred) ||
        Number(b.configured) - Number(a.configured) ||
        Number(b.detected) - Number(a.detected) ||
        a.recommendation.name.localeCompare(b.recommendation.name)
    )
    .slice(0, 6)
    .map(candidate => candidate.recommendation)
}
