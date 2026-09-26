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
import { spawnSync } from "child_process"
import { join, relative, resolve } from "path"
import { isMain } from "./utils.mjs"

const ROUTER_CONTEXT_ERROR = "may be used only in the context of a"

// @tanstack/react-query carries module-level React context (QueryClientContext).
// The entry's QueryClientProvider and every lazy chunk's useQuery must share ONE
// runtime instance; if a build ever emits a second copy, the provider's context
// is invisible to the other copy and useQuery throws "No QueryClient set" — the
// packaged app error-boundaries on launch (#95560). Same single-instance
// invariant as the react-router check above, same failure class.
const QUERY_CLIENT_CONTEXT_ERROR = "No QueryClient set, use QueryClientProvider to set one"

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
  const hasAssets =
    existsSync(assetsDir) &&
    statSync(assetsDir).isDirectory() &&
    readdirSync(assetsDir).some(name => name.endsWith(".js"))
  if (!hasAssets) {
    return { ok: false, error: `dist/assets has no built JS bundle (expected vite output under ${assetsDir})` }
  }

  const routerContextAssets = readdirSync(assetsDir)
    .filter(name => name.endsWith(".js"))
    .filter(name => readFileSync(join(assetsDir, name), "utf8").includes(ROUTER_CONTEXT_ERROR))

  if (routerContextAssets.length > 1) {
    return {
      ok: false,
      error: `react-router context invariant found in multiple JS assets: ${routerContextAssets.join(", ")}`
    }
  }

  const queryClientContextAssets = readdirSync(assetsDir)
    .filter(name => name.endsWith(".js"))
    .filter(name => readFileSync(join(assetsDir, name), "utf8").includes(QUERY_CLIENT_CONTEXT_ERROR))

  if (queryClientContextAssets.length > 1) {
    return {
      ok: false,
      error:
        `@tanstack/react-query context invariant found in multiple JS assets: ` +
        `${queryClientContextAssets.join(", ")} — duplicate react-query runtimes make the ` +
        `QueryClientProvider's context invisible to useQuery in other chunks (` +
        `"No QueryClient set" on launch, #95560)`
    }
  }

  // Parse-validate every emitted chunk as an ES module. Corrupted-silent-fail
  // bundles (a dropped identifier token mid-file) produce invalid syntax that
  // only explodes at module-evaluation time in Electron's renderer.
  const chunkParse = verifyChunksParse(assetsDir)
  if (!chunkParse.ok) {
    return chunkParse
  }

  return { ok: true }
}

// Renderer chunks are emitted as ESM (`<script type="module">` in index.html).
// A silent bundler failure can emit syntactically invalid chunks that parse fine
// as CJS-ish text but throw on module evaluation in Electron — the app then
// white-screens with `Uncaught SyntaxError` in the renderer console (observed
// 2026-09: the update-produced bundle was missing a 10-byte identifier token,
// `{$:n,}` vs `{categories:n,}`, leaving an invalid destructuring pattern).
// Parse each emitted chunk as an ES module before packaging so a corrupted
// build fails loudly and the update retry rebuilds instead of shipping it.
//
// All chunks are parsed in ONE node child: `new vm.SourceTextModule(src)`
// parse-validates a module (link/evaluate are never called, so the code never
// runs) and one spawn covers the whole assets tree. The previous shape — one
// `node --input-type=module --check` spawn per chunk — cost seconds per spawn
// on Windows and the renderer emits ~960 chunks, so the check ran silent for
// the better part of an hour and the update hand-off's 600s idle watchdog
// (scripts/desktop-update/windows.ps1) killed every Desktop update (#123216).
function verifyChunksParse(assetsDir) {
  const chunks = readdirSync(assetsDir).filter(name => name.endsWith(".js"))
  const nodeBin = process.env.NODE ||
    (process.execPath && process.execPath.endsWith("node") ? process.execPath : "node")

  // The chunk list travels over stdin as JSON: it can exceed Windows's ~32 KiB
  // environment-block limit, and `--` does not forward argv in stdin-script
  // mode. The child reads each file itself and reports failures as JSON.
  const probe = `
import { readFileSync } from "node:fs";
import vm from "node:vm";
const chunks = JSON.parse(readFileSync(0, "utf8"));
const failures = [];
for (const file of chunks) {
  try { new vm.SourceTextModule(readFileSync(file, "utf8")); }
  catch (e) { failures.push({ file, message: String((e && e.message) || e) }); }
}
if (failures.length > 0) {
  process.stdout.write(JSON.stringify(failures));
  process.exit(1);
}
`
  const result = spawnSync(nodeBin, ["--input-type=module", "--experimental-vm-modules", "-e", probe], {
    input: JSON.stringify(chunks.map(name => join(assetsDir, name))),
    maxBuffer: 64 * 1024 * 1024,
    timeout: 300_000,
  })
  if (result.error) {
    return {
      ok: false,
      error: `could not run node to syntax-check ${chunks.length} chunks: ${result.error.message}`,
    }
  }
  if (result.status !== 0) {
    const first = parseFirstFailure(result.stdout)
    const name = first ? relative(assetsDir, first.file) : "(unknown chunk)"
    const detail = first ? first.message : firstDiagnosticLine(result.stderr)
    return {
      ok: false,
      error: `built chunk is not valid ES module syntax: ${name} — ${detail}. ` +
        `A renderer chunk failed to parse, so packaging would ship an app that ` +
        `white-screens with "Uncaught SyntaxError" on launch. Re-run the build.`,
    }
  }
  return { ok: true }
}

function parseFirstFailure(stdout) {
  try {
    const failures = JSON.parse(String(stdout))
    return Array.isArray(failures) && failures.length > 0 ? failures[0] : null
  } catch {
    return null
  }
}

// Only reached when the child failed without a JSON report (crash, bad node);
// skip the ExperimentalWarning banner lines vm-modules prints on older node.
function firstDiagnosticLine(stderr) {
  const lines = String(stderr || "").split("\n").map(line => line.trim()).filter(Boolean)
  const diagnostic = lines.find(line => !line.startsWith("(") && !line.endsWith("Warning"))
  return diagnostic || lines[0] || "no diagnostics"
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

  console.log("✓ assert-dist-built: dist/index.html + assets present")
}

if (isMain(import.meta.url)) {
  main()
}

export default { checkDistBuilt }
