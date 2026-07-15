/*
 * Post-build asset fix: the vendor renderer's built CSS/JS references the
 * @nous-research/ui font files (used for the "HERMES AGENT" hero headline,
 * font-family "Collapse") via a path that Vite left un-rebased for a
 * monorepo-hoisted node_modules layout, e.g.:
 *
 *   src: url(../../../node_modules/@nous-research/ui/dist/fonts/Collapse-Bold.woff2)
 *
 * When dist/assets/index-*.css (served from webroot /assets/…) resolves that
 * relative URL, the browser clamps the excess "../" segments at the root and
 * requests /node_modules/@nous-research/ui/dist/fonts/Collapse-Bold.woff2 —
 * a path that does not exist in the shipped bundle (dist/ has no
 * node_modules/ at all). Result: 404 → "OTS parsing error" → the hero
 * headline silently falls back to a default font.
 *
 * Fix: copy the font files out of vendor's node_modules into dist/ui-fonts/
 * and rewrite every reference (in any leading-"../" or absolute-"/" form) to
 * a relative path anchored at ui-fonts/, computed per-file so it keeps
 * working regardless of which assets/ subdirectory the referencing file
 * lives in.
 *
 * Idempotent: safe to re-run — once the rewrite has happened there is
 * nothing left to match, and the font copy is a plain overwrite.
 */
import { readFileSync, writeFileSync, mkdirSync, cpSync, existsSync, readdirSync, statSync } from 'node:fs'
import { resolve, dirname, relative, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dist = resolve(root, 'dist')
const distAssets = resolve(dist, 'assets')
const uiFontsDir = resolve(dist, 'ui-fonts')

const FONT_REL = '@nous-research/ui/dist/fonts'

// Candidate locations for the vendor's font directory: the package may be
// installed local to apps/desktop, or hoisted to the workspace root.
const candidates = [
  resolve(root, `vendor/apps/desktop/node_modules/${FONT_REL}`),
  resolve(root, `vendor/node_modules/${FONT_REL}`),
]

const fontsSrc = candidates.find((p) => existsSync(p))
if (!fontsSrc) {
  throw new Error(
    `fix-assets: could not find ${FONT_REL} under any of:\n` + candidates.map((p) => `  - ${p}`).join('\n'),
  )
}

mkdirSync(uiFontsDir, { recursive: true })
cpSync(fontsSrc, uiFontsDir, { recursive: true })
console.log(`Copied fonts: ${fontsSrc} -> ${uiFontsDir}`)

// Matches any of:
//   /node_modules/@nous-research/ui/dist/fonts/
//   ../../../node_modules/@nous-research/ui/dist/fonts/  (any number of ../)
//   node_modules/@nous-research/ui/dist/fonts/           (bare relative)
const REF_RE = /(?:\.\.\/)*\/?node_modules\/@nous-research\/ui\/dist\/fonts\//g

function walk(dirPath, out = []) {
  for (const entry of readdirSync(dirPath)) {
    const full = join(dirPath, entry)
    const stat = statSync(full)
    if (stat.isDirectory()) walk(full, out)
    else if (/\.(css|js)$/.test(entry)) out.push(full)
  }
  return out
}

const targets = existsSync(distAssets) ? walk(distAssets, []) : []
let rewritten = 0

for (const file of targets) {
  const original = readFileSync(file, 'utf8')
  if (!original.includes(`${FONT_REL}/`) && !REF_RE.test(original)) continue
  REF_RE.lastIndex = 0

  const relPrefix = relative(dirname(file), uiFontsDir).split('\\').join('/')
  const replaced = original.replace(REF_RE, (relPrefix || '.') + '/')

  if (replaced !== original) {
    writeFileSync(file, replaced)
    rewritten += 1
    console.log(`Rewrote font references in: ${relative(root, file)}`)
  }
}

console.log(`fix-assets: done (${rewritten} file(s) rewritten, fonts at ${relative(root, uiFontsDir)})`)
