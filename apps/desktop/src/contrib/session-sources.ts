import { normalize } from '@/lib/text'

import { registry } from './registry'
import type { Contribution } from './types'

/** Presentation only: the stored source remains the backend query key. */
export const SESSION_SOURCE_LABELS_AREA = 'session.sourceLabels'

export interface SessionSourceLabelContribution {
  source: string
  label: string
  /** Exact owning profile. Omit to apply across profiles. */
  profile?: string
  /** Match the durable lineage root, preserving grouping after compression. */
  sessionIdPrefix?: string
}

export interface SessionSourceLabelContext {
  id?: string
  profile?: string
  _lineage_root_id?: null | string
}

interface SourceLabelRule extends SessionSourceLabelContribution {
  contributionId: string
}

function sourceLabelRules(contributions: readonly Contribution[]): SourceLabelRule[] {
  return contributions.flatMap(contribution => {
    const data = contribution.data as Partial<SessionSourceLabelContribution> | undefined

    if (
      contribution.area !== SESSION_SOURCE_LABELS_AREA ||
      !data ||
      typeof data.source !== 'string' ||
      !normalize(data.source) ||
      typeof data.label !== 'string' ||
      !data.label.trim() ||
      (data.profile !== undefined && (typeof data.profile !== 'string' || !data.profile.trim())) ||
      (data.sessionIdPrefix !== undefined && (typeof data.sessionIdPrefix !== 'string' || !data.sessionIdPrefix.trim()))
    ) {
      return []
    }

    return [
      {
        ...data,
        source: normalize(data.source),
        label: data.label.trim(),
        contributionId: contribution.id
      } as SourceLabelRule
    ]
  })
}

function matchesProfile(rule: SourceLabelRule, context?: SessionSourceLabelContext): boolean {
  return rule.profile === undefined || (context !== undefined && rule.profile === (context.profile || 'default'))
}

/** First matching contribution in registry order wins; unload restores defaults. */
export function resolveSessionSourceLabel(
  source: string,
  context?: SessionSourceLabelContext,
  contributions: readonly Contribution[] = registry.getArea(SESSION_SOURCE_LABELS_AREA)
): { label: string; groupId: string; filtered: boolean } | null {
  const id = normalize(source)

  const rule = sourceLabelRules(contributions).find(
    candidate =>
      candidate.source === id &&
      matchesProfile(candidate, context) &&
      (candidate.sessionIdPrefix === undefined ||
        Boolean((context?._lineage_root_id || context?.id)?.startsWith(candidate.sessionIdPrefix)))
  )

  if (!rule) {
    return null
  }

  const filtered = rule.sessionIdPrefix !== undefined || rule.profile !== undefined

  return {
    label: rule.label,
    groupId: filtered ? `${id}:${rule.contributionId}` : id,
    filtered
  }
}

/** Source totals cannot describe a display subgroup, including unloaded rows. */
export function hasSessionSourcePartition(
  source: string,
  profile: string | undefined,
  contributions: readonly Contribution[] = registry.getArea(SESSION_SOURCE_LABELS_AREA)
): boolean {
  const id = normalize(source)

  return sourceLabelRules(contributions).some(
    rule =>
      rule.source === id &&
      (profile === 'all' || matchesProfile(rule, { profile })) &&
      (rule.sessionIdPrefix !== undefined || (profile === 'all' && rule.profile !== undefined))
  )
}
