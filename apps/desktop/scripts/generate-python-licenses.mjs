// Generates a bundled third-party license inventory for the Hermes Agent
// Python dependencies.
//
// Shells out to `pip-licenses` against the active Python environment,
// emitting dist/dependencies-python.txt — a plain-text file containing
// each package's name, version, license, and full license text (when
// available in the installed dist-info).
//
// pip-licenses reads from the installed environment, so the inventory
// reflects whatever extras are actually installed on the build machine.
// --from=mixed is critical: it reads both the legacy License classifier
// AND the PEP 639 license_expression field, which is where modern
// packages (cryptography, pydantic, fastapi, etc.) declare their SPDX
// license. Without --from=mixed, ~30% of packages show as UNKNOWN.
import { execFileSync } from 'node:child_process'
import { mkdirSync, writeFileSync, existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
// __dirname = apps/desktop/scripts/; three parents is the repository root.
const repoRoot = path.resolve(__dirname, '..', '..', '..')
const distDir = path.resolve(__dirname, '..', 'dist')

// Locate the Python environment to scan. Resolution order:
//   1. HERMES_PYTHON (Nix dev shell / Nix desktop build)
//   2. VIRTUAL_ENV (explicit activated venv)
//   3. repo-root .venv or venv (local dev worktree)
//   4. HERMES_HOME/hermes-agent/venv (managed install, matches main.ts)
//   5. pip-licenses on PATH (fallback)
function findPipLicensesBin() {
  // Nix provides the exact Python interpreter through HERMES_PYTHON. It is
  // a store path rather than a traditional venv directory, so looking for
  // bin/pip-licenses next to it does not work; invoke the installed module.
  if (process.env.HERMES_PYTHON && existsSync(process.env.HERMES_PYTHON)) {
    return { type: 'module', path: process.env.HERMES_PYTHON, venv: null }
  }

  // pip-licenses is a console script installed into the venv's bin/Scripts
  // dir. We look for it relative to candidate venv roots.
  const binName = process.platform === 'win32' ? 'pip-licenses.exe' : 'pip-licenses'
  const scriptName = process.platform === 'win32' ? 'pip-licenses-script.py' : 'pip-licenses'

  const candidates = []

  if (process.env.VIRTUAL_ENV) {
    candidates.push(process.env.VIRTUAL_ENV)
  }

  candidates.push(path.join(repoRoot, '.venv'))
  candidates.push(path.join(repoRoot, 'venv'))

  if (process.env.HERMES_HOME) {
    candidates.push(path.join(process.env.HERMES_HOME, 'hermes-agent', 'venv'))
  }

  // Check $HOME/.hermes/hermes-agent/venv (the managed-install default)
  candidates.push(path.join(process.env.HOME || '', '.hermes', 'hermes-agent', 'venv'))

  for (const venv of candidates) {
    const binDir = process.platform === 'win32'
      ? path.join(venv, 'Scripts')
      : path.join(venv, 'bin')
    const binPath = path.join(binDir, binName)
    const scriptPath = path.join(binDir, scriptName)

    if (existsSync(binPath)) {
      return { type: 'bin', path: binPath, venv }
    }
    if (existsSync(scriptPath)) {
      // Need to run it with the venv's python
      const pythonBin = process.platform === 'win32'
        ? path.join(venv, 'Scripts', 'python.exe')
        : path.join(venv, 'bin', 'python')
      if (existsSync(pythonBin)) {
        return { type: 'module', path: pythonBin, venv }
      }
    }
  }

  // Fallback: assume pip-licenses is on PATH (CI/dev shells)
  return { type: 'path', path: 'pip-licenses', venv: null }
}

function runPipLicenses() {
  const { type, path: binPath } = findPipLicensesBin()

  // Flags:
  //   --from=mixed              — read PEP 639 license_expression AND legacy classifier
  //   --with-license-file       — include the full license text from the dist-info
  //   --with-notice-file        — include NOTICE files (Apache-2.0 §4(d) requirement)
  //   --no-license-path         — don't emit the on-disk path (we're bundling, not linking)
  //   --format=plain-vertical   — one package per block, readable as a NOTICE file
  //
  // Note: --with-license-file prints a warning about long fields with
  // plain-vertical format, but the output is still correct — the warning
  // is cosmetic. We suppress stderr to keep the build log clean.
  const args = [
    '--from=mixed',
    '--with-license-file',
    '--with-notice-file',
    '--no-license-path',
    '--format=plain-vertical',
  ]

  let result
  if (type === 'module') {
    // python -m piplicenses <args>
    result = execFileSync(binPath, ['-m', 'piplicenses', ...args], {
      encoding: 'utf8',
      maxBuffer: 50 * 1024 * 1024,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })
  } else {
    // Direct binary or PATH lookup
    result = execFileSync(binPath, args, {
      encoding: 'utf8',
      maxBuffer: 50 * 1024 * 1024,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })
  }

  return result
}

function generatedAt() {
  // Nix sets SOURCE_DATE_EPOCH for reproducible builds. Local builds retain a
  // useful wall-clock timestamp when that conventional variable is absent.
  const epoch = Number(process.env.SOURCE_DATE_EPOCH)
  return new Date(Number.isFinite(epoch) && epoch > 0 ? epoch * 1000 : Date.now()).toISOString()
}

function main() {
  mkdirSync(distDir, { recursive: true })

  let output
  try {
    output = runPipLicenses()
  } catch (e) {
    // pip-licenses not installed — write a stub so the build doesn't fail
    // and the Settings page can show a graceful "not available" message.
    output = [
      'Third-Party Software Licenses (Python)',
      '=======================================',
      '',
      'WARNING: pip-licenses was not found in the Python environment.',
      'The Python dependency license inventory could not be generated.',
      '',
      'To generate this file, install pip-licenses in the hermes venv:',
      '  pip install pip-licenses',
      '',
      `Error: ${e.message}`,
      '',
    ].join('\n')
    console.warn(`[generate-python-licenses] pip-licenses not available: ${e.message}`)
  }

  // Prepend a header so the file is self-documenting
  const header = [
    'Third-Party Software Licenses (Python)',
    '=======================================',
    '',
    'This file lists the open-source licenses of the Python dependencies',
    'of the Hermes Agent runtime bundled with the Hermes desktop app.',
    '',
    `Generated: ${generatedAt()}`,
    '',
  ].join('\n')

  const outPath = path.join(distDir, 'dependencies-python.txt')
  writeFileSync(outPath, header + output, 'utf8')

  // Count packages: in plain-vertical format, each block is name\nversion\nlicense\ntext
  // We approximate by counting version-like lines (X.Y.Z or X.Y patterns).
  const pkgCount = (output.match(/^\d+\.\d+/gm) || []).length
  console.log(`[generate-python-licenses] Wrote ~${pkgCount} packages to ${outPath}`)
}

main()
