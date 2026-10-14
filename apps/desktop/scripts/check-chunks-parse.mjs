// Parse every emitted renderer chunk as an ES module in ONE node process.
//
// assert-dist-built.mjs used to spawn `node --input-type=module --check` once per
// chunk. With ~1000 emitted chunks the update's desktop build therefore depended on
// ~1000 successful process creations, and on Windows a single spawn can stall for
// minutes under AV scanning or memory pressure. One such stall aborted the whole
// update (2026-09-27: `could not run node to syntax-check nord-*.js: spawnSync
// ...node.exe ETIMEDOUT`, which failed the update with exit 1 even though vite had
// built the bundle cleanly). Measured on the failing tree: 960 chunks, ~80 s of
// sequential spawns, worst single spawn 0.45 s on an idle box.
//
// `vm.SourceTextModule` is the in-process ES module parser, so the same check now
// costs one spawn. It reports the same early errors as `node --check` (invalid
// destructuring patterns such as the `{,:n}` token-drop this guard was written for,
// duplicate declarations) and throws SyntaxError for them. The caller passes
// --experimental-vm-modules, which the vm module API requires.
//
// Usage: node --experimental-vm-modules check-chunks-parse.mjs <assetsDir>
// stdout: {"ok":true,"checked":N}
//       | {"ok":false,"kind":"bundle","name":...,"detail":...}   the chunk does not parse
//       | {"ok":false,"kind":"harness","name"?:...,"detail":...} this CHECK could not run
// `kind` is what lets the guard report the truth: only a SyntaxError means the bundle is
// defective. A missing vm API (a node build ignoring --experimental-vm-modules) or an
// unreadable chunk is a failure of the CHECK, and must not be reported as invalid bundle
// syntax — that misattribution is exactly what the guard's `harness` branch exists to stop.
import { readFileSync, readdirSync } from "fs"
import { join } from "path"
import vm from "vm"

function report(payload) {
  process.stdout.write(JSON.stringify(payload))
}

// The check could not run (no directory, unreadable chunk, vm API unavailable): say so
// without claiming anything about the bundle's syntax.
function harnessFailure(detail, name) {
  report(name ? { ok: false, kind: "harness", name, detail } : { ok: false, kind: "harness", detail })
  process.exit(2)
}

const assetsDir = process.argv[2]
if (!assetsDir) {
  harnessFailure("no assets directory given")
}

try {
  const chunks = readdirSync(assetsDir).filter(name => name.endsWith(".js"))
  for (const name of chunks) {
    let source
    try {
      source = readFileSync(join(assetsDir, name), "utf8")
    } catch (err) {
      harnessFailure(`${err.name || "Error"}: ${String(err.message || err)}`, name)
    }
    try {
      new vm.SourceTextModule(source, { identifier: name })
    } catch (err) {
      const detail = `${err.name || "Error"}: ${String(err.message || err)}`
        .split("\n")
        .slice(0, 4)
        .join(" / ")
      if (err instanceof SyntaxError) {
        report({ ok: false, kind: "bundle", name, detail })
        process.exit(1)
      }
      // e.g. `TypeError: vm.SourceTextModule is not a constructor` when this node build does
      // not honour --experimental-vm-modules: the check did not run, the bundle is not at fault.
      harnessFailure(detail, name)
    }
  }
  report({ ok: true, checked: chunks.length })
} catch (err) {
  harnessFailure(`${err.name || "Error"}: ${err.message || err}`)
}
