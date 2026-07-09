/**
 * Regenerates the bundled IX Agency fallback data shipped with the renderer:
 *
 *   src/app/ix-agency/data/skills.json     — IX Agency skills catalog,
 *     extracted from the Intelliverse portal's src/lib/admin-skills.ts
 *   src/app/ix-agency/data/mcp-tiles.json  — static MCP tile list, extracted
 *     from the admin-mcp gateway registry
 *     (intelli-verse-kube-infra/admin-mcp/registry.json)
 *
 * Both source repos are sibling checkouts of the hermes repo by default;
 * override with IX_FRONTEND_DIR / IX_INFRA_DIR. When a source is missing the
 * existing generated file is kept (fail-soft) so builds work on machines
 * without the sibling repos. Run manually: node scripts/generate-ix-agency-data.mjs
 */
import { build } from 'esbuild'
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const dataDir = path.join(appRoot, 'src/app/ix-agency/data')
mkdirSync(dataDir, { recursive: true })

const FRONTEND_DIR = process.env.IX_FRONTEND_DIR || path.resolve(appRoot, '../../../Intelliverse-X-Webfrontend')
const INFRA_DIR = process.env.IX_INFRA_DIR || path.resolve(appRoot, '../../../intelli-verse-kube-infra')

function ensureFallback(file) {
  const target = path.join(dataDir, file)

  if (!existsSync(target)) {
    writeFileSync(target, JSON.stringify({ generatedAt: null, items: [] }, null, 2))
  }
}

async function generateSkills() {
  const src = path.join(FRONTEND_DIR, 'src/lib/admin-skills.ts')

  if (!existsSync(src)) {
    console.warn(`[ix-agency-data] skills: ${src} not found — keeping existing skills.json`)
    ensureFallback('skills.json')

    return
  }

  const tmp = path.join(appRoot, '.tmp-admin-skills.mjs')

  await build({ entryPoints: [src], outfile: tmp, bundle: true, platform: 'neutral', format: 'esm', logLevel: 'silent' })

  try {
    const mod = await import(pathToFileURL(tmp).href)

    const skills = (mod.BUILT_IN_SKILLS ?? []).map(skill => ({
      id: skill.id,
      title: skill.label,
      description: skill.blurb ?? '',
      persona: skill.superAdminOnly
        ? 'super-admin only'
        : [
            skill.tiers?.length ? `tiers: ${skill.tiers.join('/')}` : 'all tiers',
            skill.bundles?.length ? `bundles: ${skill.bundles.join(', ')}` : null
          ]
            .filter(Boolean)
            .join(' · '),
      rank: skill.rank ?? null,
      superAdminOnly: Boolean(skill.superAdminOnly),
      // Full playbook prompt — powers the native chat's "Run natively"
      // (injected as an ACTIVE SKILL block, same as the portal does).
      content: skill.content ?? '',
      starterPrompts: Array.isArray(skill.starterPrompts) ? skill.starterPrompts : [],
      // Raw scoping metadata, kept for future scope filtering.
      tiers: Array.isArray(skill.tiers) ? skill.tiers : [],
      bundles: Array.isArray(skill.bundles) ? skill.bundles : [],
      appIds: Array.isArray(skill.appIds) ? skill.appIds : []
    }))

    writeFileSync(
      path.join(dataDir, 'skills.json'),
      JSON.stringify({ generatedAt: new Date().toISOString(), source: 'admin-skills.ts', items: skills }, null, 2)
    )
    console.log(`[ix-agency-data] skills.json: ${skills.length} skills`)
  } finally {
    rmSync(tmp, { force: true })
    rmSync(`${tmp}.map`, { force: true })
  }
}

function generateMcpTiles() {
  const src = path.join(INFRA_DIR, 'admin-mcp/registry.json')

  if (!existsSync(src)) {
    console.warn(`[ix-agency-data] mcp: ${src} not found — keeping existing mcp-tiles.json`)
    ensureFallback('mcp-tiles.json')

    return
  }

  const registry = JSON.parse(readFileSync(src, 'utf8'))
  const groups = new Map((registry.groups ?? []).map(group => [group.id, group.label]))

  const tiles = (registry.actions ?? [])
    .filter(action => action.mcpUrl)
    .map(action => {
      let domain = ''

      try {
        domain = new URL(action.mcpUrl).hostname
      } catch {
        domain = action.mcpUrl
      }

      return {
        id: action.id,
        label: action.label,
        blurb: action.blurb ?? '',
        group: groups.get(action.group) ?? action.group,
        mcpUrl: action.mcpUrl,
        domain,
        mcpAuthHint: action.mcpAuthHint ?? ''
      }
    })

  writeFileSync(
    path.join(dataDir, 'mcp-tiles.json'),
    JSON.stringify({ generatedAt: new Date().toISOString(), source: 'admin-mcp/registry.json', items: tiles }, null, 2)
  )
  console.log(`[ix-agency-data] mcp-tiles.json: ${tiles.length} MCP tiles`)
}

await generateSkills()
generateMcpTiles()
