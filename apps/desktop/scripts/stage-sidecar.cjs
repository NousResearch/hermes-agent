'use strict'

/**
 * Stage the hermes-eats-world sidecar (Python source) for packaging.
 *
 * The composer's "attach app / dock" feature shells out to the sidecar. In dev
 * the desktop uses the working copy at ~/hermes-eats-world, but a packaged build
 * must not depend on a user-managed home-dir checkout. So at build time we copy
 * the sidecar's Python package into apps/desktop/build/native-deps/hermes-eats-world,
 * which ships via the existing `extraResources` entry (build/native-deps ->
 * resources/native-deps). main.cjs resolves SIDECAR_DIR from there when packaged.
 *
 * Source location: $HERMES_SIDECAR_SRC, else ~/hermes-eats-world (the sidecar is
 * a separate repo). The sidecar is Windows-only and optional, so a missing
 * source is a WARNING, not a build failure — cross-platform/CI builds that don't
 * ship it still succeed; the feature is gated off at runtime.
 *
 * Runs as part of `npm run build`. Idempotent.
 */

const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const APP_ROOT = path.resolve(__dirname, '..')
const STAGE_DEST = path.join(APP_ROOT, 'build', 'native-deps', 'hermes-eats-world')
const SIDECAR_SRC = process.env.HERMES_SIDECAR_SRC || path.join(os.homedir(), 'hermes-eats-world')

// Only the runtime-essential bits: the `sidecar` package plus dependency
// manifests (so an installer step can provision the Python deps). Everything
// else — tests, docs, .git, build caches, spikes — is left behind.
const INCLUDE_DIRS = ['sidecar']
const INCLUDE_FILES = ['pyproject.toml', 'requirements.txt', 'README.md']
const SKIP_DIR_NAMES = new Set(['__pycache__', '.git', 'node_modules', '.pytest_cache', '.venv', 'tests'])

function copyTree(srcDir, destDir) {
  let count = 0
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.isDirectory() && SKIP_DIR_NAMES.has(entry.name)) continue
    const src = path.join(srcDir, entry.name)
    const dest = path.join(destDir, entry.name)
    if (entry.isDirectory()) {
      fs.mkdirSync(dest, { recursive: true })
      count += copyTree(src, dest)
    } else if (entry.isFile() && (entry.name.endsWith('.py') || entry.name.endsWith('.toml') || entry.name.endsWith('.txt'))) {
      fs.mkdirSync(path.dirname(dest), { recursive: true })
      fs.copyFileSync(src, dest)
      count += 1
    }
  }
  return count
}

function main() {
  // Always (re)create the staging dir so the electron-builder extraResources
  // entry pointing at it never dangles — an empty dir ships as an empty dir
  // and main.ts's resolveSidecarDir() treats it as "not bundled".
  fs.rmSync(STAGE_DEST, { recursive: true, force: true })
  fs.mkdirSync(STAGE_DEST, { recursive: true })

  if (!fs.existsSync(SIDECAR_SRC)) {
    console.warn(
      `stage-sidecar: source not found at ${SIDECAR_SRC} — skipping (the ` +
        `attach-app feature will be unavailable in this build). Set ` +
        `HERMES_SIDECAR_SRC to stage it.`
    )
    return
  }

  let copied = 0
  for (const dir of INCLUDE_DIRS) {
    const src = path.join(SIDECAR_SRC, dir)
    if (fs.existsSync(src)) {
      fs.mkdirSync(path.join(STAGE_DEST, dir), { recursive: true })
      copied += copyTree(src, path.join(STAGE_DEST, dir))
    }
  }
  for (const file of INCLUDE_FILES) {
    const src = path.join(SIDECAR_SRC, file)
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(STAGE_DEST, file))
      copied += 1
    }
  }

  console.log(`stage-sidecar: staged ${copied} files from ${SIDECAR_SRC} -> ${STAGE_DEST}`)
}

main()
