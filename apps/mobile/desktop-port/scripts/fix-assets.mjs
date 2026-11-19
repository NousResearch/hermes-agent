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
 * Fix: copy the font files out of the renderer's node_modules into
 * dist/ui-fonts/ and rewrite every reference (in any leading-"../" or
 * absolute-"/" form) to a relative path anchored at ui-fonts/, computed
 * per-file so it keeps working regardless of which assets/ subdirectory the
 * referencing file lives in.
 *
 * The font source is derived from the renderer directory (apps/desktop) that
 * build.sh actually built, passed as argv[2]. This covers BOTH build modes:
 *   - standalone:  vendor/apps/desktop         (fonts under vendor/…)
 *   - in-tree:     $HERMES_AGENT_SRC/apps/desktop (no vendor/ exists)
 *
 * Idempotent: safe to re-run — once the rewrite has happened there is
 * nothing left to match, and the font copy is a plain overwrite.
 */
import { readFileSync, writeFileSync, mkdirSync, cpSync, existsSync, readdirSync, statSync } from 'node:fs'
import { resolve, dirname, relative, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const FONT_REL = '@nous-research/ui/dist/fonts'

// The renderer source dir whose node_modules holds the fonts. build.sh passes
// the directory it built (DESKTOP_DIR) as argv[2]. Fallback to the standalone
// vendor location keeps the script runnable with no argument.
export function resolveDesktopDir(argv, rootDir) {
  const arg = argv[2]
  return arg ? resolve(arg) : resolve(rootDir, 'vendor/apps/desktop')
}

// The font package may be installed local to apps/desktop or hoisted to the
// workspace root (two levels up from apps/desktop). Derived from the renderer
// dir so the in-tree build (which has no vendor/) resolves correctly.
export function fontCandidates(desktopDir) {
  return [
    resolve(desktopDir, 'node_modules', FONT_REL),
    resolve(desktopDir, '..', '..', 'node_modules', FONT_REL),
  ]
}

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

export function run(argv) {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
  const dist = resolve(root, 'dist')
  const distAssets = resolve(dist, 'assets')
  const uiFontsDir = resolve(dist, 'ui-fonts')

  const desktopDir = resolveDesktopDir(argv, root)
  const candidates = fontCandidates(desktopDir)

  const fontsSrc = candidates.find((p) => existsSync(p))
  if (!fontsSrc) {
    throw new Error(
      `fix-assets: could not find ${FONT_REL} under any of:\n` + candidates.map((p) => `  - ${p}`).join('\n'),
    )
  }

  mkdirSync(uiFontsDir, { recursive: true })
  cpSync(fontsSrc, uiFontsDir, { recursive: true })
  console.log(`Copied fonts: ${fontsSrc} -> ${uiFontsDir}`)

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
}

// Run only when invoked directly (not when imported by tests).
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run(process.argv)
}
