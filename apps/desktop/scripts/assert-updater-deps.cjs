'use strict'

/**
 * Post-pack integrity gate for the staged electron-updater closure.
 *
 * WHY THIS EXISTS
 * ---------------
 * The shell self-updater require()s electron-updater at runtime from the staged
 * `updater-deps/vendor/node_modules/` subtree (see stage-updater-deps.cjs for
 * the full story). electron-builder does NOT validate extraResources inputs, so
 * a dropped or short-circuited stage ships a packaged app whose updater silently
 * disables itself -- exactly the hc-435 regression (0.16.1-0.16.3 shipped with
 * NO electron-updater at all, self-update dead, and CI caught nothing).
 *
 * The macOS workflow already had an inline shell gate for this, but it only
 * checked a SINGLE file (electron-updater/out/main.js) and lived only on mac.
 * The Windows workflow staged the identical 16-package / 202-file closure with
 * NO gate at all -- so a win-only stage breakage would sail through CI (hc-436).
 *
 * This is the shared, cross-platform (Node, no PowerShell) gate BOTH workflows
 * call after packaging. It reads the manifest that stage-updater-deps.cjs writes
 * alongside the staged tree (updater-deps-manifest.json, shipped by the same
 * extraResources mapping) and asserts that EVERY package the stager recorded --
 * and each package's require()-main entry -- actually survived into the packaged
 * resources. Driving the check off the emitted manifest (not a hand-maintained
 * list) means a newly added transitive dependency is covered automatically, the
 * same dynamic-closure philosophy as the stager itself.
 *
 * USAGE
 *   node scripts/assert-updater-deps.cjs <resourcesDir>
 * where <resourcesDir> is the packaged Resources dir:
 *   macOS  release/mac*<app>.app/Contents/Resources
 *   win    release/win-unpacked/resources
 * Exits 0 on a complete closure, 1 (with ::error:: annotations naming each
 * missing package/entry) otherwise.
 */

const fs = require('node:fs')
const path = require('node:path')

// Layout inside the packaged Resources dir. Mirrors stage-updater-deps.cjs:
//   <Resources>/updater-deps/updater-deps-manifest.json
//   <Resources>/updater-deps/vendor/node_modules/<pkg>/...
const UPDATER_DIR = 'updater-deps'
const MANIFEST_NAME = 'updater-deps-manifest.json'
const VENDOR_NODE_MODULES = path.join('vendor', 'node_modules')

// Load-bearing entry the shell self-updater require()s first. Checked
// explicitly (belt-and-suspenders) so even a manifest that somehow omitted the
// root package can never let a no-op updater ship. This is the exact file the
// pre-hc-436 mac gate asserted.
const ROOT_ENTRY_REL = path.join('electron-updater', 'out', 'main.js')

/**
 * Verify the staged updater closure inside a packaged Resources dir.
 * @param {string} resourcesDir packaged Resources dir (see USAGE).
 * @returns {{ ok: boolean, checked: number, missing: string[], manifestPath: string }}
 */
function checkUpdaterDeps(resourcesDir) {
  const missing = []
  const updaterRoot = path.join(resourcesDir, UPDATER_DIR)
  const vendorNm = path.join(updaterRoot, VENDOR_NODE_MODULES)
  const manifestPath = path.join(updaterRoot, MANIFEST_NAME)

  // 1) The manifest itself must have shipped -- its absence means the whole
  //    updater-deps extraResources copy was dropped (the hc-435 failure mode).
  if (!fs.existsSync(manifestPath)) {
    missing.push(`${MANIFEST_NAME} (updater-deps not packaged at all?)`)
    return { ok: false, checked: 0, missing, manifestPath }
  }

  let manifest
  try {
    manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  } catch (err) {
    missing.push(`${MANIFEST_NAME} unreadable/invalid JSON: ${err.message}`)
    return { ok: false, checked: 0, missing, manifestPath }
  }

  const packages = Array.isArray(manifest.packages) ? manifest.packages : []
  if (packages.length === 0) {
    missing.push(`${MANIFEST_NAME} lists zero packages (empty closure)`)
    return { ok: false, checked: 0, missing, manifestPath }
  }

  // 2) Every package the stager recorded, plus its require()-main entry, must
  //    exist on disk in the packaged vendor tree.
  let checked = 0
  for (const entry of packages) {
    checked += 1
    const relDir = String(entry.dir || '').split('/').join(path.sep)
    const pkgDir = path.join(vendorNm, relDir)
    if (!isDir(pkgDir)) {
      missing.push(`package dir: ${UPDATER_DIR}/${VENDOR_NODE_MODULES}/${entry.dir}`)
      continue // no point checking its entry file if the dir is gone
    }
    if (entry.entry) {
      const relEntry = String(entry.entry).split('/').join(path.sep)
      const entryPath = path.join(vendorNm, relEntry)
      if (!isFile(entryPath)) {
        missing.push(`entry file: ${UPDATER_DIR}/${VENDOR_NODE_MODULES}/${entry.entry}`)
      }
    }
  }

  // 3) Belt-and-suspenders: the root updater entry specifically.
  const rootEntryPath = path.join(vendorNm, ROOT_ENTRY_REL)
  const rootEntryRelPosix = ROOT_ENTRY_REL.split(path.sep).join('/')
  if (!isFile(rootEntryPath)) {
    const already = missing.some((m) => m.includes(rootEntryRelPosix))
    if (!already) {
      missing.push(`root entry: ${UPDATER_DIR}/${VENDOR_NODE_MODULES}/${rootEntryRelPosix}`)
    }
  }

  return { ok: missing.length === 0, checked, missing, manifestPath }
}

function isDir(p) {
  try {
    return fs.statSync(p).isDirectory()
  } catch {
    return false
  }
}

function isFile(p) {
  try {
    return fs.statSync(p).isFile()
  } catch {
    return false
  }
}

function main() {
  const resourcesDir = process.argv[2]
  if (!resourcesDir) {
    console.error('::error::assert-updater-deps: missing <resourcesDir> argument')
    process.exit(2)
  }
  if (!isDir(resourcesDir)) {
    console.error(`::error::assert-updater-deps: resources dir not found: ${resourcesDir}`)
    process.exit(2)
  }

  const { ok, checked, missing } = checkUpdaterDeps(resourcesDir)
  if (!ok) {
    console.error(
      '::error::updater-deps integrity gate FAILED — shell self-update would be ' +
        'disabled in this build. Missing:'
    )
    for (const m of missing) console.error(`::error::  - ${m}`)
    process.exit(1)
  }
  console.log(`[assert-updater-deps] OK — ${checked} staged packages present in ${resourcesDir}`)
}

if (require.main === module) {
  main()
}

module.exports = { checkUpdaterDeps, UPDATER_DIR, MANIFEST_NAME, VENDOR_NODE_MODULES }
