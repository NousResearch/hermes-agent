// Generates a bundled third-party license inventory for the desktop app's
// JavaScript dependencies.
//
// Scans the installed node_modules tree with license-checker, then emits
// dist/dependencies.txt — a plain-text file containing each package's
// name, version, license, repository URL, and full license text when
// available. This file is shipped in the electron-builder app bundle via
// extraResources and surfaced in the Settings → Licenses page.
//
// Why license-checker (standalone CLI) instead of rollup-plugin-license:
// Vite 8 uses rolldown under the hood, and rollup-plugin-license is a
// rollup plugin — compatibility is uncertain and the plugin approach
// only covers deps that end up in the bundle. license-checker scans the
// full node_modules tree, so the inventory is complete even for deps
// that are never imported at runtime (native modules, etc.).
import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
// __dirname = apps/desktop/scripts/
// repoRoot = apps/desktop/scripts/../../.. = repo root
const repoRoot = path.resolve(__dirname, '..', '..', '..')
const distDir = path.resolve(__dirname, '..', 'dist')

// We scan from the repo root, not the desktop workspace, because npm
// workspaces hoist all dependencies to the root node_modules/. Scanning
// from apps/desktop only finds the handful of deps that have their own
// node_modules (non-hoisted native modules etc.), while the repo root
// captures the full production dependency tree (77+ packages).
const SCAN_ROOT = repoRoot

function runLicenseChecker() {
  const bin = path.join(repoRoot, 'node_modules', '.bin', 'license-checker')
  // --production: exclude devDependencies — the shipped app only carries
  //   production deps in its bundle, so the license inventory matches.
  // --json: structured output for easy post-processing.
  // --relativeLicensePath: emit paths to LICENSE files so we can read them.
  const args = ['--production', '--json', '--relativeLicensePath']
  const raw = execFileSync(bin, args, {
    cwd: SCAN_ROOT,
    encoding: 'utf8',
    maxBuffer: 50 * 1024 * 1024,
    stdio: ['pipe', 'pipe', 'pipe'],
  })
  return JSON.parse(raw)
}

function readLicenseFile(relPath) {
  if (!relPath) {
    return null
  }

  // license-checker returns paths relative to the scan root (cwd).
  // In a workspace, these look like "node_modules/foo/LICENSE" and
  // resolve correctly from the workspace root.
  const full = path.resolve(SCAN_ROOT, relPath)
  if (!existsSync(full)) {
    return null
  }

  try {
    return readFileSync(full, 'utf8').trim()
  } catch {
    return null
  }
}

// Format the inventory as a human-readable plain-text file. Each package
// gets a delimited block with its metadata + full license text, so the
// result is a self-contained NOTICE-equivalent.
function generatedAt() {
  // Nix sets SOURCE_DATE_EPOCH for reproducible builds. Local builds retain a
  // useful wall-clock timestamp when that conventional variable is absent.
  const epoch = Number(process.env.SOURCE_DATE_EPOCH)
  return new Date(Number.isFinite(epoch) && epoch > 0 ? epoch * 1000 : Date.now()).toISOString()
}

function formatInventory(entries) {
  const lines = []

  lines.push('Third-Party Software Licenses')
  lines.push('=============================')
  lines.push('')
  lines.push('This file lists the open-source licenses of the npm dependencies')
  lines.push('bundled with the Hermes desktop application.')
  lines.push('')
  lines.push(`Generated: ${generatedAt()}`)
  lines.push(`Total packages: ${entries.length}`)
  lines.push('')

  for (const entry of entries) {
    lines.push('─'.repeat(70))
    lines.push(`Package:   ${entry.name}@${entry.version}`)
    lines.push(`License:   ${entry.license}`)
    if (entry.repository) {
      lines.push(`Repository: ${entry.repository}`)
    }
    if (entry.publisher) {
      lines.push(`Publisher: ${entry.publisher}`)
    }
    if (entry.email) {
      lines.push(`Email:     ${entry.email}`)
    }
    lines.push('')
    if (entry.licenseText) {
      lines.push(entry.licenseText)
    } else {
      lines.push('(No license text file found — see the package repository for details.)')
    }
    lines.push('')
  }

  return lines.join('\n')
}

function main() {
  mkdirSync(distDir, { recursive: true })

  let data
  try {
    data = runLicenseChecker()
  } catch (e) {
    // license-checker not installed or failed — write a stub so the build
    // doesn't fail and the Settings page can show a graceful message.
    const stub = [
      'Third-Party Software Licenses',
      '=============================',
      '',
      'WARNING: license-checker was not found or failed to run.',
      'The JavaScript dependency license inventory could not be generated.',
      '',
      `Error: ${e.message}`,
      '',
    ].join('\n')
    writeFileSync(path.join(distDir, 'dependencies.txt'), stub, 'utf8')
    console.warn(`[generate-js-licenses] license-checker failed: ${e.message}`)
    return
  }

  // Dedupe by name@version — the same package can appear once.
  const seen = new Map()

  for (const [pkgKey, info] of Object.entries(data)) {
    // pkgKey is "name@version"
    const atIdx = pkgKey.lastIndexOf('@')
    const name = atIdx > 0 ? pkgKey.slice(0, atIdx) : pkgKey
    const version = atIdx > 0 ? pkgKey.slice(atIdx + 1) : ''

    const dedupeKey = `${name}@${version}`
    if (seen.has(dedupeKey)) {
      continue
    }

    const licenseText = info.licenseFile
      ? readLicenseFile(info.licenseFile)
      : null

    seen.set(dedupeKey, {
      name,
      version,
      license: info.licenses || info.license || 'UNKNOWN',
      repository: info.repository || null,
      publisher: info.publisher || null,
      email: info.email || null,
      licenseText,
    })
  }

  const entries = [...seen.values()].sort((a, b) =>
    a.name.toLowerCase().localeCompare(b.name.toLowerCase()),
  )

  const output = formatInventory(entries)
  const outPath = path.join(distDir, 'dependencies.txt')
  writeFileSync(outPath, output, 'utf8')

  console.log(`[generate-js-licenses] Wrote ${entries.length} packages to ${outPath}`)
}

main()
