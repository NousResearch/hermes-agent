/**
 * Skill ↔ MCP wiring — pure helpers that map skills (org playbooks, user
 * drafts, templates) to the MCP tiles they call, so the Skills tab can group
 * skills by tool and show which MCPs a skill spans.
 *
 * Matching is text-derived: a skill references a tile when its markdown
 * mentions the tile id (word-boundary, case-insensitive, `-`/`_`/space
 * tolerant) or a known alias (e.g. "openbsp" → whatsapp tile). This works
 * for the org catalog, the built-in templates AND freshly typed drafts
 * without any extra metadata to maintain.
 */
import mcpTilesData from './data/mcp-tiles.json'
import type { IxMcpTileItem } from './types'

export const MCP_TILES: IxMcpTileItem[] = Array.isArray(mcpTilesData.items)
  ? (mcpTilesData.items as IxMcpTileItem[])
  : []

/** Extra names skills use for a tile that differ from the tile id. */
const TILE_ALIASES: Record<string, string[]> = {
  whatsapp: ['openbsp'],
  twenty: ['crm'],
  fonoster: ['voice call', 'phone call'],
  'stripe-revenue': ['stripe'],
  'revenuecat-revenue': ['revenuecat'],
  'ga4-analytics': ['ga4'],
  'gsc-search': ['gsc', 'search console'],
  'nakama-console': ['nakama'],
  'appsflyer-attribution': ['appsflyer'],
  'didit-kyc': ['didit'],
  'veriff-kyc': ['veriff'],
  'quickbooks': ['qbo'],
  'aws-costs': ['aws cost', 'cloud cost']
}

/** Tiles that are junk matches in prose (common English words). */
const AMBIGUOUS_IDS = new Set(['quests'])

function patternFor(term: string): RegExp {
  // "aws-costs" should match "aws costs", "aws_costs" and "aws-costs".
  const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/[-_ ]/g, '[-_ ]')

  return new RegExp(`(^|[^a-z0-9])${escaped}([^a-z0-9]|$)`, 'i')
}

export interface SkillTextLike {
  id?: string
  title?: string
  description?: string
  content?: string
}

/** Tile ids referenced by a skill's text, in tile-registry order. */
export function skillMcpIds(skill: SkillTextLike, tiles: IxMcpTileItem[] = MCP_TILES): string[] {
  const haystack = [skill.id, skill.title, skill.description, skill.content].filter(Boolean).join('\n')

  if (!haystack.trim()) {
    return []
  }

  return tiles
    .filter(tile => {
      const terms = [
        ...(AMBIGUOUS_IDS.has(tile.id) ? [] : [tile.id]),
        ...(TILE_ALIASES[tile.id] ?? [])
      ]

      return terms.some(term => patternFor(term).test(haystack))
    })
    .map(tile => tile.id)
}

/** Mirror of the electron-side direct-wire rule: public HTTPS on our domains
 *  gets a first-class entry in the generated Hermes config; everything else
 *  is reachable via the admin-mcp gateway's admin_call_mcp. */
export function tileWiring(tile: IxMcpTileItem): 'direct' | 'gateway' {
  const url = tile.mcpUrl || ''

  if (tile.id !== 'admin-mcp' && url.startsWith('https://')) {
    try {
      const host = new URL(url).hostname

      if (host.endsWith('.intelli-verse-x.ai') || host.endsWith('.toba-tech.ai')) {
        return 'direct'
      }
    } catch {
      // fall through to gateway
    }
  }

  return 'gateway'
}

/* ── Pods — the four org pods skills and MCPs roll up to ──────────────────── */

export const IX_PODS = ['content', 'growth', 'product', 'engineering'] as const

export type IxPod = (typeof IX_PODS)[number]

export const POD_LABELS: Record<IxPod, string> = {
  content: 'Content pod',
  growth: 'Growth pod',
  product: 'Product pod',
  engineering: 'Engineering pod'
}

/** Explicit tile → pod assignment. The registry's own groups are too coarse
 *  ("Measure & monitor" mixes ad attribution, product analytics and infra),
 *  so every tile is placed deliberately; the group map below only catches
 *  tiles added to the registry after this list was written. */
const TILE_PODS: Record<string, IxPod> = {
  // Content — making and shipping content.
  postiz: 'content',
  beehiiv: 'content',
  stitch: 'content',
  'open-seo': 'content',
  firecrawl: 'content',
  'agent-mcp': 'content',
  'youtube-stats': 'content',
  // Growth — outreach, campaigns, CRM, ads and acquisition analytics.
  notifuse: 'growth',
  whatsapp: 'growth',
  fonoster: 'growth',
  telnyx: 'growth',
  twenty: 'growth',
  chatwoot: 'growth',
  'appsflyer-attribution': 'growth',
  adjust: 'growth',
  'ga4-analytics': 'growth',
  'gsc-search': 'growth',
  // Product — games, product analytics, revenue, compliance, contracts.
  'nakama-console': 'product',
  quests: 'product',
  posthog: 'product',
  'stripe-revenue': 'product',
  'revenuecat-revenue': 'product',
  quickbooks: 'product',
  documenso: 'product',
  'appstore-insights': 'product',
  'playstore-insights': 'product',
  'didit-kyc': 'product',
  'veriff-kyc': 'product',
  // Engineering — infra, observability, automation, agent plumbing.
  grafana: 'engineering',
  'aws-costs': 'engineering',
  n8n: 'engineering',
  leantime: 'engineering',
  dataviz: 'engineering',
  'admin-mcp': 'engineering',
  gastown: 'engineering',
  'intelliverse-mcp': 'engineering'
}

/** Fallback for registry tiles that appear after TILE_PODS was written. */
const GROUP_PODS: Record<string, IxPod> = {
  'Talk to people': 'growth',
  'Make content': 'content',
  'Game ops': 'product',
  Commerce: 'product',
  'Plan & track work': 'engineering',
  'Measure & monitor': 'product',
  Agents: 'engineering'
}

export function tilePod(tile: IxMcpTileItem): IxPod {
  return TILE_PODS[tile.id] ?? GROUP_PODS[tile.group ?? ''] ?? 'engineering'
}

/** Pods a skill rolls up to — the pods of every MCP it references, in the
 *  canonical pod order. Cross-pod skills legitimately return several. */
export function skillPods(skill: SkillTextLike, tiles: IxMcpTileItem[] = MCP_TILES): IxPod[] {
  const matched = new Set(skillMcpIds(skill, tiles))
  const pods = new Set(tiles.filter(tile => matched.has(tile.id)).map(tilePod))

  return IX_PODS.filter(pod => pods.has(pod))
}

export interface PodSkillGroup<S> {
  label: string
  pod: IxPod
  skills: S[]
}

/** Group skills by pod (via their MCPs' pods). A skill whose tools span pods
 *  appears under each; skills naming no tile land in `ungrouped`. Pods keep
 *  their canonical order and empty pods are dropped. */
export function groupSkillsByPod<S extends SkillTextLike>(
  skills: S[],
  tiles: IxMcpTileItem[] = MCP_TILES
): { pods: PodSkillGroup<S>[]; ungrouped: S[] } {
  const byPod = new Map<IxPod, S[]>()
  const ungrouped: S[] = []

  for (const skill of skills) {
    const pods = skillPods(skill, tiles)

    if (!pods.length) {
      ungrouped.push(skill)

      continue
    }

    for (const pod of pods) {
      const bucket = byPod.get(pod)

      if (bucket) {
        bucket.push(skill)
      } else {
        byPod.set(pod, [skill])
      }
    }
  }

  const pods = IX_PODS.filter(pod => byPod.has(pod)).map(pod => ({
    pod,
    label: POD_LABELS[pod],
    skills: byPod.get(pod) as S[]
  }))

  return { pods, ungrouped }
}

export interface McpSkillGroup<S> {
  tile: IxMcpTileItem
  skills: S[]
}

/** Group skills by the MCP tiles they reference. A skill spanning several
 *  MCPs appears under EACH of its tiles (that's the point — cross-MCP skills
 *  are the default posture). Groups are ordered by skill count, then by the
 *  tile's registry position; `ungrouped` collects skills that name no tile. */
export function groupSkillsByMcp<S extends SkillTextLike>(
  skills: S[],
  tiles: IxMcpTileItem[] = MCP_TILES
): { groups: McpSkillGroup<S>[]; ungrouped: S[] } {
  const byTile = new Map<string, S[]>()
  const ungrouped: S[] = []

  for (const skill of skills) {
    const ids = skillMcpIds(skill, tiles)

    if (!ids.length) {
      ungrouped.push(skill)

      continue
    }

    for (const id of ids) {
      const bucket = byTile.get(id)

      if (bucket) {
        bucket.push(skill)
      } else {
        byTile.set(id, [skill])
      }
    }
  }

  const groups = tiles
    .filter(tile => byTile.has(tile.id))
    .map(tile => ({ tile, skills: byTile.get(tile.id) as S[] }))
    .sort((a, b) => b.skills.length - a.skills.length)

  return { groups, ungrouped }
}
