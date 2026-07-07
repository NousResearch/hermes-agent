'use strict'

// Dependency-free node_modules patcher, wired into the root postinstall so it
// re-applies after every `npm ci`/`npm install` (which rewrite node_modules).
// Deliberately not patch-package: that would need a lockfile entry and an
// extra dependency for what is a one-line containment fix.
//
// Each patch replaces an exact source string and stamps a marker comment so
// re-runs are no-ops. If the pinned package version changes shape (find string
// AND marker both missing), the install fails loudly so the patch gets
// refreshed or retired instead of silently reintroducing the bug.

const fs = require('node:fs')
const path = require('node:path')

const REPO_ROOT = path.join(__dirname, '..')

const PATCHES = [
  {
    // @assistant-ui/store 0.2.13 (pinned via root package.json overrides)
    // throws on index-out-of-bounds lookups. During high-frequency store
    // replacement (session switch mid-stream, gateway reconnect replay) stale
    // subscribers read past the new, shorter list and the throw escapes React
    // render entirely (store notification tick), blanking the desktop app.
    // Production signature: "tapClientLookup: Index N out of bounds
    // (length: 0)" in ~/.hermes/logs/desktop.log. Upstream:
    // https://github.com/assistant-ui/assistant-ui/issues/4051
    // Containment: clamp to the nearest live resource (stale-but-valid for
    // one tick, self-heals on the next snapshot); undefined when empty.
    file: 'node_modules/@assistant-ui/store/dist/tapClientLookup.js',
    marker: 'hermes-patch(assistant-ui#4051)',
    find:
      'if (lookup.index < 0 || lookup.index >= keys.length) throw new Error(`tapClientLookup: Index ${lookup.index} out of bounds (length: ${keys.length})`);',
    replace:
      '/* hermes-patch(assistant-ui#4051): clamp transient out-of-bounds lookups instead of throwing */ if (lookup.index < 0 || lookup.index >= keys.length) return keys.length === 0 ? undefined : resources[Math.min(Math.max(lookup.index, 0), keys.length - 1)].methods;'
  }
]

function applyPatchToSource(source, patch) {
  if (source.includes(patch.marker)) {
    return { source, status: 'already-applied' }
  }

  if (!source.includes(patch.find)) {
    return { source, status: 'target-not-found' }
  }

  return { source: source.replace(patch.find, patch.replace), status: 'applied' }
}

function main() {
  let failed = false

  for (const patch of PATCHES) {
    const filePath = path.join(REPO_ROOT, patch.file)

    if (!fs.existsSync(filePath)) {
      // Partial installs (e.g. `npm install --workspaces=false`) may not have
      // the package on disk; the next full install re-runs this script.
      console.log(`[apply-patches] skip (not installed): ${patch.file}`)
      continue
    }

    const source = fs.readFileSync(filePath, 'utf8')
    const result = applyPatchToSource(source, patch)

    if (result.status === 'target-not-found') {
      console.error(
        `[apply-patches] FAILED: ${patch.file} does not contain the expected source.\n` +
          `The installed package version has drifted — update or retire the patch in patches/apply-patches.cjs.`
      )
      failed = true
      continue
    }

    if (result.status === 'applied') {
      fs.writeFileSync(filePath, result.source)
    }

    console.log(`[apply-patches] ${result.status}: ${patch.file}`)
  }

  if (failed) {
    process.exitCode = 1
  }
}

module.exports = { PATCHES, applyPatchToSource }

if (require.main === module) {
  main()
}
