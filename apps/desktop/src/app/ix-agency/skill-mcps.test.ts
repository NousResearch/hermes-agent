/**
 * Tests for skill-mcps.ts — the skill↔MCP text mapping behind the Skills
 * tab's "By tool" grouping and the wired-MCP chips.
 */
import { describe, expect, test } from 'vitest'

import skillsData from './data/skills.json'
import {
  groupSkillsByMcp,
  groupSkillsByPod,
  IX_PODS,
  MCP_TILES,
  skillMcpIds,
  skillPods,
  tilePod,
  tileWiring
} from './skill-mcps'
import type { IxMcpTileItem } from './types'

const tile = (id: string, mcpUrl = ''): IxMcpTileItem => ({ id, label: id, mcpUrl, domain: '' })

const TILES = [
  tile('notifuse', 'https://notifuse-mcp.intelli-verse-x.ai/mcp'),
  tile('whatsapp', 'https://openbsp-mcp.intelli-verse-x.ai/'),
  tile('grafana', 'https://grafana-mcp.intelli-verse-x.ai/'),
  tile('stripe-revenue', 'http://stripe-mcp.aicart.svc.cluster.local/'),
  tile('posthog', 'https://mcp.posthog.com/mcp')
]

describe('skillMcpIds', () => {
  test('matches tile ids with word boundaries, case-insensitive', () => {
    expect(skillMcpIds({ content: 'Send via Notifuse, check grafana panels.' }, TILES)).toEqual([
      'notifuse',
      'grafana'
    ])
  })

  test('does not match substrings inside other words', () => {
    expect(skillMcpIds({ content: 'the grafanapocalypse notifuser' }, TILES)).toEqual([])
  })

  test('aliases: openbsp maps to the whatsapp tile, stripe to stripe-revenue', () => {
    expect(skillMcpIds({ content: 'broadcast on openbsp; reconcile stripe payouts' }, TILES)).toEqual([
      'whatsapp',
      'stripe-revenue'
    ])
  })

  test('separator tolerance: "stripe revenue" and "stripe_revenue" both match', () => {
    expect(skillMcpIds({ content: 'stripe revenue numbers' }, TILES)).toContain('stripe-revenue')
    expect(skillMcpIds({ content: 'stripe_revenue numbers' }, TILES)).toContain('stripe-revenue')
  })

  test('empty text yields no matches', () => {
    expect(skillMcpIds({}, TILES)).toEqual([])
  })
})

describe('groupSkillsByMcp', () => {
  test('cross-MCP skills land in every matching group; groups sort by count', () => {
    const skills = [
      { id: 'a', content: 'notifuse + grafana' },
      { id: 'b', content: 'grafana only' },
      { id: 'c', content: 'nothing relevant' }
    ]

    const { groups, ungrouped } = groupSkillsByMcp(skills, TILES)

    expect(groups.map(g => g.tile.id)).toEqual(['grafana', 'notifuse'])
    expect(groups[0].skills.map(s => s.id)).toEqual(['a', 'b'])
    expect(groups[1].skills.map(s => s.id)).toEqual(['a'])
    expect(ungrouped.map(s => s.id)).toEqual(['c'])
  })

  test('the bundled org catalog groups by real tiles and spans multiple MCPs', () => {
    // Real-data check against the shipped playbooks + registry snapshot: the
    // catalog must produce several tool groups, and a healthy share of skills
    // must span MORE than one MCP (the "skills across multiple MCPs" posture).
    const catalog = (skillsData.items as Array<{ content: string; description: string; id: string; title: string }>) ?? []
    const { groups } = groupSkillsByMcp(catalog)

    expect(groups.length).toBeGreaterThan(5)

    const multiMcp = catalog.filter(skill => skillMcpIds(skill).length > 1)

    expect(multiMcp.length).toBeGreaterThan(5)
  })
})

describe('pods', () => {
  test('skillPods rolls MCPs up to pods in canonical order, deduped', () => {
    // notifuse → growth, grafana → engineering, whatsapp → growth
    expect(skillPods({ content: 'notifuse + grafana + openbsp broadcast' }, TILES)).toEqual([
      'growth',
      'engineering'
    ])
    expect(skillPods({ content: 'nothing relevant' }, TILES)).toEqual([])
  })

  test('groupSkillsByPod buckets cross-pod skills under each pod', () => {
    const skills = [
      { id: 'a', content: 'notifuse campaign, grafana alerts' },
      { id: 'b', content: 'stripe revenue rollup' },
      { id: 'c', content: 'no tools named' }
    ]

    const { pods, ungrouped } = groupSkillsByPod(skills, TILES)

    expect(pods.map(g => g.pod)).toEqual(['growth', 'product', 'engineering'])
    expect(pods.find(g => g.pod === 'growth')?.skills.map(s => s.id)).toEqual(['a'])
    expect(pods.find(g => g.pod === 'product')?.skills.map(s => s.id)).toEqual(['b'])
    expect(pods.find(g => g.pod === 'engineering')?.skills.map(s => s.id)).toEqual(['a'])
    expect(ungrouped.map(s => s.id)).toEqual(['c'])
  })

  test('every registry tile lands in one of the four pods', () => {
    for (const t of MCP_TILES) {
      expect(IX_PODS).toContain(tilePod(t))
    }
  })
})

describe('tileWiring', () => {
  test('public https on our domains is direct; cluster-local and SaaS are gateway', () => {
    expect(tileWiring(TILES[0])).toBe('direct')
    expect(tileWiring(TILES[3])).toBe('gateway')
    expect(tileWiring(TILES[4])).toBe('gateway')
  })
})
