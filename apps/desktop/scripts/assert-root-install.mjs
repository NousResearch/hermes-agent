// Build-time guard: refuse to start a build the installed tree cannot finish.
//
// The desktop workspace's dependencies are hoisted to the repo-root
// `node_modules`, so a root install that only covers *part* of the workspace
// graph leaves this app importable-looking but unbuildable. The guard exists to
// turn that into one actionable line ("run npm ci from the repo root") instead
// of a failure deep inside vite.
//
// It runs from `prebuild`, ahead of `npm run clean`, so a tree that cannot
// build is rejected before the build starts deleting its own outputs. `build`
// re-runs it for anyone invoking the build steps directly; the check is pure
// filesystem lookups, so paying for it twice costs nothing.

import { existsSync, readFileSync } from "fs"
import { createRequire } from "module"
import { resolve, join, dirname } from "path"
import { isMain } from "./utils.mjs"

// Packages the build *consumes*, as opposed to merely declares. Each one is
// load-bearing for a distinct build step, and each one has been observed
// missing from a partial root install:
//
//   vite            — bundles the renderer (`vite build`).
//   katex           — `src/styles.css` imports `katex/dist/katex.min.css`, so
//                     the CSS transform fails before a single chunk is emitted.
//   electron        — the runtime electron-builder packages; without it `pack`
//                     cannot produce an unpacked app at all.
//   electron-builder — the packager `npm run builder` shells out to.
//
// Checking only `vite` (the original guard) passes a tree missing any of the
// others, which is how an incomplete install reached `vite build` and died on
// an unresolved `katex/dist/katex.min.css` with no hint that the install — not
// the source — was at fault (#86443).
const BUILD_CRITICAL_PACKAGES = ["vite", "katex", "electron", "electron-builder"]
export { BUILD_CRITICAL_PACKAGES }

// Resolve the way Node's own lookup does — walk `node_modules` upward — rather
// than through `require.resolve`. A package whose `exports` map does not expose
// `./package.json` is not resolvable by path even when correctly installed, and
// that must not read as "missing".
function packageIsInstalled(name, fromDir) {
  let dir = fromDir
  for (;;) {
    if (existsSync(join(dir, "node_modules", name, "package.json"))) return true
    const parent = dirname(dir)
    if (parent === dir) return false
    dir = parent
  }
}

// Pure check — returns { ok: true } or { ok: false, error: "..." }.
// Kept side-effect-free so it can be unit tested without spawning a process.
export function checkRootInstall(appDir, rootDir) {
  const missing = BUILD_CRITICAL_PACKAGES.filter(pkg => !packageIsInstalled(pkg, appDir))
  if (missing.length > 0) {
    return {
      ok: false,
      error:
        `the desktop build needs ${missing.join(", ")}, which the current install ` +
        `does not provide. A partial root install leaves the workspace looking ` +
        `present while the build cannot complete. Reinstall from the repo root: ` +
        `cd ${rootDir} && npm ci`
    }
  }

  // `vite.config.ts` aliases react/react-dom to whatever this workspace resolves,
  // and React refuses to run when the two come from different installed copies
  // ("Minified React error #527" — it throws before the first paint, so the app
  // window stays blank). npm stays silent about the split because the hoisted
  // react still satisfies react-dom's caret peer range. Fail the build loudly
  // instead of shipping a white screen.
  const requireFromApp = createRequire(join(appDir, "package.json"))
  const installedVersion = pkg =>
    JSON.parse(readFileSync(requireFromApp.resolve(`${pkg}/package.json`), "utf8")).version

  let react
  let reactDom
  try {
    react = installedVersion("react")
    reactDom = installedVersion("react-dom")
  } catch (err) {
    // Both are in BUILD_CRITICAL_PACKAGES' spirit but not its list: they are
    // checked by version, and an unreadable package.json is a broken install
    // rather than an absent one. Report it as such instead of throwing.
    return {
      ok: false,
      error: `could not read the installed react/react-dom versions (${err.message}). Reinstall from the repo root: cd ${rootDir} && npm ci`
    }
  }

  if (react !== reactDom) {
    return {
      ok: false,
      error:
        `react@${react} / react-dom@${reactDom} version mismatch — React would fail ` +
        `with error #527 and render a blank window. Pin both to the same version ` +
        `in ${join(appDir, "package.json")}, then reinstall: cd ${rootDir} && npm ci`
    }
  }

  return { ok: true }
}

function main() {
  const app = resolve(import.meta.dirname, "..")
  const root = resolve(app, "..", "..")
  const result = checkRootInstall(app, root)

  if (!result.ok) {
    console.error(`✗ assert-root-install: ${result.error}`)
    process.exit(1)
  }
}

if (isMain(import.meta.url)) {
  main()
}
