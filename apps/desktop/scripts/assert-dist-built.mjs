// Build-time guard: refuse to hand a half-built renderer to electron-builder.
//
// `npm run pack` / `npm run dist*` are `npm run build && npm run builder`.
// If the `build` step (tsc -b && vite build) fails but packaging proceeds
// anyway — a stale checkout that fails typecheck, an interrupted vite build,
// or npm not short-circuiting `&&` in some shells — electron-builder happily
// packages an app with an empty or missing `dist/`. The result launches but
// blank-pages with `ERR_FILE_NOT_FOUND` for dist/index.html, with no clue why.
//
// This runs at the tail of `build`, after vite build, so any packaging path
// inherits it. It fails loud and early instead of shipping a broken bundle.
// See issues #39484 (renderer blank page) and #41327 / #39472 (dashboard 404).

import { existsSync, readFileSync, statSync, readdirSync } from "fs"
import { join, resolve } from "path"
import { isMain } from "./utils.mjs"

// Pure check — returns { ok: true } or { ok: false, error: "..." }.
// Kept side-effect-free so it can be unit tested without spawning a process.
export function checkDistBuilt(distDir) {
  if (!existsSync(distDir) || !statSync(distDir).isDirectory()) {
    return { ok: false, error: `no dist directory at ${distDir}` }
  }

  const indexHtml = join(distDir, "index.html")
  if (!existsSync(indexHtml) || !statSync(indexHtml).isFile()) {
    return { ok: false, error: `dist/index.html is missing at ${indexHtml}` }
  }
  if (statSync(indexHtml).size === 0) {
    return { ok: false, error: `dist/index.html is empty at ${indexHtml}` }
  }

  // index.html alone isn't enough — vite emits hashed JS into dist/assets.
  // An index.html with no script bundle still blank-pages.
  const assetsDir = join(distDir, "assets")
  const assetNames =
    existsSync(assetsDir) &&
    statSync(assetsDir).isDirectory() &&
    readdirSync(assetsDir)
  const hasJsBundle = assetNames && assetNames.some(name => name.endsWith(".js"))
  if (!hasJsBundle) {
    return { ok: false, error: `dist/assets has no built JS bundle (expected vite output under ${assetsDir})` }
  }

  // A CSS file existing is not sufficient. Tailwind can emit its theme/base
  // layers while silently scanning zero renderer files (for example when a
  // local Git exclude hides apps/desktop/src). Electron then launches and the
  // React tree mounts, but every pane split computes to display:block. Require
  // a small set of structural utilities that the desktop shell always uses so
  // a source-scan failure stops the build before electron-builder installs it.
  const cssNames = assetNames.filter(name => name.endsWith(".css"))
  if (cssNames.length === 0) {
    return { ok: false, error: `dist/assets has no built CSS bundle (expected vite output under ${assetsDir})` }
  }

  const css = cssNames.map(name => readFileSync(join(assetsDir, name), "utf8")).join("\n")
  const requiredUtilities = [".flex{display:flex}", ".flex-row{flex-direction:row}", ".min-h-0{min-height:0}"]
  const missingUtilities = requiredUtilities.filter(rule => !css.includes(rule))
  if (missingUtilities.length > 0) {
    return {
      ok: false,
      error: `dist CSS is missing structural Tailwind utilities: ${missingUtilities.join(", ")}`
    }
  }

  return { ok: true }
}

function main() {
  const desktopRoot = resolve(import.meta.dirname, "..")
  const distDir = join(desktopRoot, "dist")
  const result = checkDistBuilt(distDir)

  if (!result.ok) {
    console.error(`\n✗ assert-dist-built: ${result.error}`)
    console.error("  The renderer bundle is missing or incomplete, so packaging")
    console.error("  would produce an app that launches to a blank page.")
    console.error("  Re-run the build and check the tsc/vite output above for the")
    console.error("  real failure, then package again:")
    console.error(`    cd ${desktopRoot} && npm run build\n`)
    process.exit(1)
  }

  console.log("✓ assert-dist-built: renderer JS + structural CSS present")
}

if (isMain(import.meta.url)) {
  main()
}

export default { checkDistBuilt }
