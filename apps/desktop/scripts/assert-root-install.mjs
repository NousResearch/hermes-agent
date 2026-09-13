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

import { existsSync, readFileSync, realpathSync } from "fs"
import { createRequire } from "module"
import { resolve, join, dirname } from "path"
import { isMain } from "./utils.mjs"
import { electronBundleOptions } from "./electron-bundle-options.mjs"

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
//
// These four are the documented floor — always checked, even when the app's
// package.json cannot be read. The full class is wider: EVERY non-optional
// package the workspace manifest declares is something the build may import
// (`vite.config.ts` pulls `@rolldown/plugin-babel`, `@vitejs/plugin-react`,
// `@tailwindcss/vite`; `bundle-electron-main.mjs` pulls `esbuild`; the renderer
// imports the rest). A hand-maintained list drifts the moment a new import
// lands, so `checkRootInstall` unions the floor with the manifest's declared
// `dependencies` + `devDependencies` — a partial install is refused whichever
// package it happened to drop. `optionalDependencies` are excluded by design:
// npm legitimately skips them (platform-gated natives like `get-windows`).
const BUILD_CRITICAL_PACKAGES = ["vite", "katex", "electron", "electron-builder"]
export { BUILD_CRITICAL_PACKAGES }

// Resolve the way Node's own lookup does — walk `node_modules` upward — rather
// than through `require.resolve`. A package whose `exports` map does not expose
// `./package.json` is not resolvable by path even when correctly installed, and
// that must not read as "missing". Scoped names (`@scope/name`) are a nested
// directory under `node_modules`, which `join` handles.
function packageIsInstalled(name, fromDir) {
  let dir = fromDir
  for (;;) {
    if (existsSync(join(dir, "node_modules", name, "package.json"))) return true
    const parent = dirname(dir)
    if (parent === dir) return false
    dir = parent
  }
}

// Every package the workspace manifest at `appDir` declares as required
// (`dependencies` + `devDependencies`; never `optionalDependencies`). An
// unreadable or malformed manifest yields [] — the floor still applies, and
// the build's own manifest read fails loudly on its own.
export function requiredPackages(appDir) {
  try {
    const manifest = JSON.parse(readFileSync(join(appDir, "package.json"), "utf8"))
    return [
      ...Object.keys(manifest.dependencies ?? {}),
      ...Object.keys(manifest.devDependencies ?? {}),
    ]
  } catch {
    return []
  }
}

// esbuild adopts a Yarn Plug'n'Play manifest from any ancestor of its working
// directory (the build runs from this app's directory) — including folders above
// the repo, such as a stray `~/.pnp.cjs` left by running `yarn` in a home
// directory. A real manifest that does not list this workspace makes every bare
// import fail (`Could not resolve "simple-git"`) over a complete npm install, and
// nothing in that error points outside the repo. An empty file, a directory, or an
// unrelated file with the same name is ignored by esbuild, so the filename alone
// must not fail the build: `findPnpManifest` is only a cheap pre-filter, and
// `checkPnpResolution` asks esbuild itself, using the real bundle options.
const PNP_MANIFESTS = [".pnp.cjs", ".pnp.js", ".pnp.data.json"]
export { PNP_MANIFESTS }

// First path named like a PnP manifest, walking from `fromDir` up to the filesystem root, or null.
export function findPnpManifest(fromDir) {
  let dir = fromDir
  for (;;) {
    for (const name of PNP_MANIFESTS) {
      const candidate = join(dir, name)
      if (existsSync(candidate)) return candidate
    }
    const parent = dirname(dir)
    if (parent === dir) return null
    dir = parent
  }
}

// Runs the real Electron bundles (electron-bundle-options.mjs) in memory from
// `appDir` — the same working directory, entrypoints and externals as
// bundle-electron-main.mjs, with nothing written — and collects every import that
// esbuild says a Plug'n'Play manifest blocked. A manifest can list some of the
// packages the bundles need and omit others, so probing a single package is not
// enough. Any other esbuild error is left for the build itself to report.
export async function checkPnpResolution(requestedAppDir, { bundleOptions } = {}) {
  if (!findPnpManifest(requestedAppDir)) return { ok: true }

  // The build's working directory is always a real path; esbuild matches the
  // manifest against real paths, so a symlinked spelling (macOS /var -> /private/var)
  // would silently skip PnP and hide the failure this check exists to catch.
  const appDir = realpathSync(requestedAppDir)
  const realEntry = entry => (existsSync(entry) ? realpathSync(entry) : entry)
  const { build } = await import("esbuild")
  const blocked = new Map() // package -> manifest esbuild used

  for (const options of bundleOptions ?? electronBundleOptions(appDir)) {
    try {
      await build({
        ...options,
        entryPoints: options.entryPoints.map(realEntry),
        absWorkingDir: appDir,
        write: false,
        logLevel: "silent"
      })
    } catch (err) {
      for (const message of err.errors ?? []) {
        const note = (message.notes ?? []).find(n => n.text.includes("Plug'n'Play"))
        if (!note) continue
        const pkg = note.text.match(/importing "([^"]+)"/)?.[1] ?? message.text
        blocked.set(pkg, note.location?.file ? resolve(appDir, note.location.file) : findPnpManifest(appDir))
      }
    }
  }

  if (blocked.size === 0) return { ok: true }
  const packages = [...blocked.keys()].map(name => `"${name}"`).join(", ")
  return {
    ok: false,
    error:
      `esbuild is using the Yarn Plug'n'Play manifest at ${[...new Set(blocked.values())].join(", ")}, ` +
      `which does not list ${packages} for this npm workspace, so the Electron bundle cannot ` +
      `resolve them. Move or delete that file — it is usually left over from running yarn ` +
      `in a parent folder — then rebuild.`
  }
}

// Pure check — returns { ok: true } or { ok: false, error: "..." }.
// Kept side-effect-free so it can be unit tested without spawning a process.
export function checkRootInstall(appDir, rootDir) {
  const wanted = [...new Set([...BUILD_CRITICAL_PACKAGES, ...requiredPackages(appDir)])]
  const missing = wanted.filter(pkg => !packageIsInstalled(pkg, appDir))
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

async function main() {
  const app = resolve(import.meta.dirname, "..")
  const root = resolve(app, "..", "..")
  // The PnP probe loads esbuild, so it only runs once the install check has
  // confirmed the declared toolchain is present.
  for (const result of [checkRootInstall(app, root), await checkPnpResolution(app)]) {
    if (!result.ok) {
      console.error(`✗ assert-root-install: ${result.error}`)
      process.exit(1)
    }
  }
}

if (isMain(import.meta.url)) {
  await main()
}
