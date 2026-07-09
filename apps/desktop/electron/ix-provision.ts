/**
 * IX Agency provisioning — wires the WHOLE platform into local Hermes with
 * zero manual setup:
 *
 *  - MCP servers  : every EKS-hosted MCP deployment (notifuse, fonoster,
 *                   telnyx, chatwoot, twenty, n8n, nakama, grafana, …) is
 *                   generated straight into ~/.hermes/config.yaml from the
 *                   same registry snapshot the Tools tab shows
 *                   (src/app/ix-agency/data/mcp-tiles.json). The admin-mcp
 *                   gateway stays as the catch-all for in-cluster-only tiles.
 *  - Skills       : the portal's admin-skills catalog (skills.json — full
 *                   playbook markdown included) is materialized as native
 *                   SKILL.md folders under ~/.hermes/skills/ix-portal/, and
 *                   every platform skill repo present on disk is added as an
 *                   external skills dir.
 *  - Models       : the LiteLLM gateway's /v1/models is fetched live so the
 *                   chat picker offers EVERY model the key can route to, not
 *                   just the small static allowlist.
 *
 * Pure logic + node fs; unit-testable without electron (ix-provision.test.ts).
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import mcpTilesData from '../src/app/ix-agency/data/mcp-tiles.json' with { type: 'json' }
import ixSkillsData from '../src/app/ix-agency/data/skills.json' with { type: 'json' }

type IxMcpTile = {
  id: string
  label?: string
  blurb?: string
  mcpUrl?: string
  mcpAuthHint?: string
}

type IxPortalSkill = {
  id: string
  title?: string
  description?: string
  persona?: string
  content?: string
}

/* ── 1. Direct MCP servers from the EKS registry snapshot ─────────────────── */

const EKS_MCP_HOST_SUFFIXES = ['.intelli-verse-x.ai', '.toba-tech.ai']

/** Tiles that get a first-class mcp_servers entry: public HTTPS endpoints on
 *  our own domains. In-cluster-only URLs (svc.cluster.local) and external
 *  SaaS MCPs stay reachable through the admin-mcp gateway's admin_call_mcp. */
export function ixDirectMcpTiles(tiles: IxMcpTile[] = mcpTilesData.items as IxMcpTile[]): IxMcpTile[] {
  return tiles.filter(tile => {
    const url = tile.mcpUrl || ''

    if (tile.id === 'admin-mcp' || !url.startsWith('https://')) {
      return false
    }

    try {
      const host = new URL(url).hostname

      return EKS_MCP_HOST_SUFFIXES.some(suffix => host === suffix.slice(1) || host.endsWith(suffix))
    } catch {
      return false
    }
  })
}

function envVarForTile(id: string): string {
  const base = id.replace(/[^a-zA-Z0-9]+/g, '_').toUpperCase()

  return base.endsWith('_MCP') ? `${base}_TOKEN` : `${base}_MCP_TOKEN`
}

/** Whether direct calls to this tile expect a caller-supplied bearer token
 *  (vs a wrapper with server-side default credentials / no auth at all). */
export function tileNeedsBearer(tile: IxMcpTile): boolean {
  const hint = tile.mcpAuthHint || ''

  if (/no token/i.test(hint)) {
    return false
  }

  return /bearer/i.test(hint)
}

function yamlQuote(value: string): string {
  return `"${value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
}

/** YAML fragment (already indented for a top-level `mcp_servers:` map) with
 *  one entry per EKS MCP deployment. */
export function ixDirectMcpServersYaml(tiles: IxMcpTile[] = ixDirectMcpTiles()): string {
  const blocks = tiles.map(tile => {
    const lines: string[] = []
    const summary = [tile.label, tile.blurb].filter(Boolean).join(' — ')

    if (summary) {
      lines.push(`  # ${summary}`)
    }

    lines.push(`  ${tile.id}:`)
    lines.push(`    url: ${yamlQuote(tile.mcpUrl || '')}`)
    lines.push('    timeout: 60')

    if (tileNeedsBearer(tile)) {
      const envVar = envVarForTile(tile.id)

      lines.push('    headers:')
      lines.push(`      Authorization: "Bearer \${${envVar}}"`)
    }

    return lines.join('\n')
  })

  return blocks.join('\n\n')
}

/* ── 2. Platform skill repos present on this machine ──────────────────────── */

const PLATFORM_SKILL_DIRS = [
  'dev/hermes-deployment/skills',
  'dev/intelli-verse-x-agent-skills/skills',
  'dev/Intelli-verse-X-SDK/skills',
  'dev/nakama/.agents/skills',
  'dev/agent-skills/.agents/skills',
  'dev/Agentic-SEO-Skill/resources/skills',
  'dev/geo-optimizer-skill/src/geo_optimizer/skills',
  'dev/hyperframes/skills',
  // Curated packs from the hermes-agent fork's upstream skills library.
  'dev/hermes-agent/skills/research',
  'dev/hermes-agent/skills/creative',
  'dev/hermes-agent/skills/productivity',
  'dev/hermes-agent/skills/github',
  'dev/hermes-agent/skills/software-development',
  'dev/hermes-agent/skills/media',
  'dev/hermes-agent/skills/data-science',
  'dev/hermes-agent/skills/email'
]

/** Curated hermes skill packs bundled with the packaged app (electron-builder
 *  extraResources → <resources>/hermes-skills/<pack>) so coding/eval/GitHub
 *  loop capability survives on machines WITHOUT the dev checkouts. */
const PACKAGED_SKILL_PACKS = [
  'research',
  'creative',
  'productivity',
  'github',
  'software-development',
  'media',
  'data-science',
  'email'
]

function isDirectory(dir: string): boolean {
  try {
    return fs.statSync(dir).isDirectory()
  } catch {
    return false
  }
}

export function ixExternalSkillDirs(homedir: string = os.homedir(), resourcesPath?: string): string[] {
  const candidates = PLATFORM_SKILL_DIRS.map(rel => path.join(homedir, rel))

  if (resourcesPath) {
    for (const pack of PACKAGED_SKILL_PACKS) {
      // The dev checkout wins when present; the bundled copy is the fallback.
      if (!isDirectory(path.join(homedir, 'dev/hermes-agent/skills', pack))) {
        candidates.push(path.join(resourcesPath, 'hermes-skills', pack))
      }
    }
  }

  return candidates.filter(isDirectory)
}

/* ── 3. Full generated ~/.hermes/config.yaml ──────────────────────────────── */

/** Superset of ix-status.ts's minimalHermesConfigYaml: LiteLLM as the model
 *  provider, the admin-mcp gateway PLUS a direct entry for every EKS MCP
 *  deployment, and all platform skill dirs found on disk. */
export function fullHermesConfigYaml(
  litellmUrl: string,
  gatewayUrl: string,
  homedir: string = os.homedir(),
  resourcesPath?: string
): string {
  const base = String(litellmUrl || '').replace(/\/+$/, '')
  const baseUrl = /\/v1$/.test(base) ? base : `${base}/v1`
  const skillDirs = ixExternalSkillDirs(homedir, resourcesPath)

  const skillsYaml = skillDirs.length
    ? `\nskills:\n  # Platform skill repos found on this machine (SKILL.md trees).\n  external_dirs:\n${skillDirs.map(dir => `    - ${yamlQuote(dir)}`).join('\n')}\n  auto_load_threshold: 8\n`
    : ''

  return `# Generated by the Hermes desktop IX Agency setup. Every EKS MCP
# deployment is wired directly (same registry snapshot as the Tools tab);
# in-cluster-only and SaaS tiles remain reachable via admin-mcp's
# admin_call_mcp. Bearer placeholders resolve from ~/.hermes/.env.
model:
  default: "anthropic/claude-opus-4.6"
  provider: "custom"
  base_url: "${baseUrl}"

mcp_servers:
  # Admin gateway meta-tools (admin_list_tools, admin_call_mcp, …) — the
  # catch-all that fans out to every tile with server-side credentials.
  admin-mcp:
    url: "${gatewayUrl}"
    timeout: 60
    headers:
      Authorization: "Bearer \${ADMIN_MCP_TOKEN}"

${ixDirectMcpServersYaml()}
${skillsYaml}
memory:
  enabled: true

context_files:
  enabled: true
`
}

/* ── 4. Materialize portal admin-skills as native SKILL.md folders ────────── */

/** Writes ~/.hermes/skills/ix-portal/<id>/SKILL.md for every skill in the
 *  bundled portal catalog (full playbook markdown when available). Existing
 *  files are overwritten so re-running init picks up catalog updates.
 *  Returns the number of skills written. */
export function materializeIxPortalSkills(
  hermesHome: string,
  skills: IxPortalSkill[] = ixSkillsData.items as IxPortalSkill[]
): number {
  const root = path.join(hermesHome, 'skills', 'ix-portal')
  let written = 0

  for (const skill of skills) {
    const slug = String(skill.id || '').replace(/[^a-zA-Z0-9-_]+/g, '-').replace(/^-+|-+$/g, '')

    if (!slug) {
      continue
    }

    const description = (skill.description || skill.title || slug).replace(/\s+/g, ' ').trim()

    const body =
      skill.content?.trim() ||
      `# Skill: ${skill.title || slug}\n\n${skill.description || ''}\n`

    const frontmatter = [
      '---',
      `name: ${slug}`,
      `description: ${yamlQuote(description)}`,
      'author: Intelliverse-X portal catalog',
      ...(skill.persona ? [`metadata:`, `  ix:`, `    persona: ${yamlQuote(skill.persona)}`] : []),
      '---'
    ].join('\n')

    const dir = path.join(root, slug)

    fs.mkdirSync(dir, { recursive: true })
    fs.writeFileSync(path.join(dir, 'SKILL.md'), `${frontmatter}\n\n${body}\n`, 'utf8')
    written += 1
  }

  return written
}

/* ── 5. Live model list from the LiteLLM gateway ──────────────────────────── */

export type IxChatModelOption = { id: string; label: string }

let liteLlmModelsCache: { key: string; at: number; models: IxChatModelOption[] } | null = null

const LITELLM_MODELS_TTL_MS = 5 * 60_000

/** GET <litellm>/v1/models — every model the key is allowed to route to.
 *  Returns null on any failure so callers can fall back to the static list.
 *  Cached for 5 minutes per url+key. */
export async function fetchLiteLlmModels(
  litellmUrl: string,
  litellmKey: string,
  fetchImpl: typeof fetch = fetch
): Promise<IxChatModelOption[] | null> {
  const base = String(litellmUrl || '').replace(/\/+$/, '')

  if (!base) {
    return null
  }

  const url = `${/\/v1$/.test(base) ? base : `${base}/v1`}/models`
  const cacheKey = `${url}\n${litellmKey || ''}`

  if (liteLlmModelsCache && liteLlmModelsCache.key === cacheKey && Date.now() - liteLlmModelsCache.at < LITELLM_MODELS_TTL_MS) {
    return liteLlmModelsCache.models
  }

  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 5_000)

    const response = await fetchImpl(url, {
      headers: litellmKey ? { Authorization: `Bearer ${litellmKey}` } : {},
      signal: controller.signal
    })

    clearTimeout(timer)

    if (!response.ok) {
      return null
    }

    const payload = (await response.json()) as { data?: Array<{ id?: unknown }> }

    const models = (payload.data ?? [])
      .map(entry => (typeof entry.id === 'string' ? entry.id.trim() : ''))
      .filter(Boolean)
      .sort()
      .map(id => ({ id, label: id }))

    if (!models.length) {
      return null
    }

    liteLlmModelsCache = { key: cacheKey, at: Date.now(), models }

    return models
  } catch {
    return null
  }
}

export function resetLiteLlmModelsCache() {
  liteLlmModelsCache = null
}
