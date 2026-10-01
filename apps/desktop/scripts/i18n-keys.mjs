#!/usr/bin/env node
// Emit the Desktop catalog's key set for language-pack validation.
//
// `hermes plugins validate` checks a pack's `<lang>.desktop.yaml` keys against
// `locales/_keys.desktop.json` (repo root, committed). This script derives that
// file from `src/i18n/en.ts` — every leaf's dotted path, sorted; function-
// valued entries are keys too (a pack overrides them with a positional
// `{0}`/`{1}` string). Run via `npm run i18n:keys` and commit the result (`--check` verifies);
// the build never writes it — a dirty checkout would break `hermes update`.
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import { isMain, repoRoot, workspaceTool } from '../../../scripts/build/frontend-common.mjs'

export const SURFACE = 'desktop'

/** Sorted dotted leaf paths. Mirrors `flattenMessageKeys` in apps/shared. */
export function flattenKeys(tree, prefix = '') {
  const isRecord = value => typeof value === 'object' && value !== null && !Array.isArray(value)
  if (!isRecord(tree)) return prefix ? [prefix] : []
  const keys = []
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (isRecord(value)) keys.push(...flattenKeys(value, path))
    else if (typeof value === 'function') {
      try {
        const result = value(...Array.from({ length: value.length }, (_, index) => `{${index}}`))
        if (typeof result === 'string') keys.push(path)
      } catch {
        // A function that cannot produce a string is not a valid pack leaf.
      }
    } else keys.push(path)
  }
  return keys.sort()
}

export function renderKeysFile(keys) {
  return `${JSON.stringify({ surface: SURFACE, keys }, null, 2)}\n`
}

/** Bundle en.ts (it imports app modules through the `@/` alias) into one ESM
 *  file the emitter can import in plain Node. */
async function loadEnglishCatalog(source) {
  const app = join(source, 'apps/desktop')
  const { build } = await import(pathToFileURL(workspaceTool(source, 'apps/desktop', 'esbuild')).href)
  const outdir = join(app, 'build/i18n-keys')
  mkdirSync(outdir, { recursive: true })
  const outfile = join(outdir, `en-${process.pid}.mjs`)
  const alias = {
    '@hermes/shared/ansi': join(source, 'apps/shared/src/ansi.ts'),
    '@hermes/shared/billing': join(source, 'apps/shared/src/billing-types.ts'),
    '@hermes/shared/color': join(source, 'apps/shared/src/color.ts'),
    '@hermes/shared/i18n': join(source, 'apps/shared/src/i18n.ts'),
    '@hermes/shared/translucency': join(source, 'apps/shared/src/translucency.ts'),
    '@hermes/shared': join(source, 'apps/shared/src/index.ts'),
    '@': join(app, 'src')
  }
  await build({
    entryPoints: [join(app, 'src/i18n/en.ts')],
    bundle: true,
    format: 'esm',
    platform: 'node',
    target: 'node22',
    jsx: 'automatic',
    alias,
    // Stylesheets and assets some transitive app modules import are irrelevant
    // to the catalog's shape.
    loader: { '.css': 'empty', '.svg': 'empty', '.png': 'empty', '.woff2': 'empty' },
    logLevel: 'silent',
    outfile
  })
  try {
    const mod = await import(pathToFileURL(outfile).href)
    return mod.en
  } finally {
    rmSync(outfile, { force: true })
  }
}

export async function emitDesktopKeys({ source = repoRoot, out, check = false } = {}) {
  source = resolve(source)
  out = resolve(out ?? join(source, 'locales', `_keys.${SURFACE}.json`))
  const en = await loadEnglishCatalog(source)
  const rendered = renderKeysFile(flattenKeys(en))
  if (check) {
    let current = ''
    try {
      current = readFileSync(out, 'utf8')
    } catch {
      // A missing file is drift too.
    }
    if (current !== rendered) {
      throw new Error(`${out} is out of date with src/i18n/en.ts — run \`npm run i18n:keys\` in apps/desktop and commit it`)
    }
    return out
  }
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, rendered)
  return out
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({
    options: { source: { type: 'string' }, out: { type: 'string' }, check: { type: 'boolean', default: false } }
  })
  const out = await emitDesktopKeys(values)
  console.log(`${values.check ? 'verified' : 'wrote'} ${out}`)
}
