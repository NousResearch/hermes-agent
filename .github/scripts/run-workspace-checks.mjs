// Run every workspace check at the same time and report all failures.
//
// The unit of work is a CHECK, and not a workspace. A package that declares
// `check:*` sub-scripts gives one unit for each sub-script. A package with a
// plain `check` gives that. This is the same selection rule the old CI matrix
// used, so the set of commands is unchanged. Only the schedule is different.
//
// This is not `npm run --ws check`, because that command is serial and stops
// at the first workspace that fails. This runs every unit and fails at the
// end with the full list.
//
// The output of each unit goes to a buffer and prints on completion inside a
// group that collapses. Children that write to one stdout together interleave
// their lines, and a failure is then hard to read.
//
// This also runs on a laptop: `node .github/scripts/run-workspace-checks.mjs`.
// `--concurrency N` sets the limit. `--list` prints the units and exits.
//
// Silent-skip guard: discovery is dynamic (`npm query .workspace`), so a
// workspace whose check scripts get renamed or removed — or a workspace that
// npm stops resolving (registry layout drift, a broken package.json) — drops
// out of the run SILENTLY and the job still reports green having checked one
// workspace fewer. `--expect-workspaces <pkg,...>` (wired in CI) pins the set:
// any listed workspace that contributes zero units is an error naming the
// workspace, so the skip can never pass unnoticed.

import { execFileSync, spawn } from 'node:child_process'
import { availableParallelism } from 'node:os'

const IS_CI = Boolean(process.env.GITHUB_ACTIONS)
const NPM = process.platform === 'win32' ? 'npm.cmd' : 'npm'

/**
 * @param {Record<string, import('node:child_process').ExecFileSyncOptionsWithStringEncoding>} [opts]
 */
function queryWorkspacePkgs(opts) {
  const raw = execFileSync(NPM, ['query', '.workspace'], {
    encoding: 'utf-8',
    shell: process.platform === 'win32',
    ...opts,
  })
  return /** @type {{location: string, scripts?: Record<string,string>}[]} */ (JSON.parse(raw))
}

/** @returns {{pkg: string, script: string}[]} */
function discoverUnits() {
  const pkgs = queryWorkspacePkgs()

  /** @type {{pkg: string, script: string}[]} */
  const units = []
  for (const pkg of pkgs) {
    const scripts = pkg.scripts || {}
    const subs = Object.keys(scripts).filter((s) => /^check:.+$/.test(s))
    if (subs.length > 0) {
      for (const script of subs) units.push({ pkg: pkg.location, script })
    } else if (scripts.check) {
      units.push({ pkg: pkg.location, script: 'check' })
    }
  }
  return units
}

/**
 * The silent-skip assertion: every workspace named in
 * `--expect-workspaces` must contribute at least one unit. Returns the
 * list of expected workspaces that contributed none (empty = pass).
 * @param {{pkg: string, script: string}[]} units
 * @param {string[]} expected
 */
function findSkippedExpectedWorkspaces(units, expected) {
  const contributors = new Set(units.map((u) => u.pkg))
  return expected.filter((pkg) => !contributors.has(pkg))
}

/** @param {{pkg: string, script: string}} unit */
function runUnit(unit) {
  return new Promise((resolve) => {
    const started = Date.now()
    const child = spawn(NPM, ['run', '--prefix', unit.pkg, unit.script], {
      // Buffer, and do not inherit. Children that share one stdout
      // interleave their lines, and a failure is then hard to read.
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: process.platform === 'win32',
    })
    /** @type {Buffer[]} */
    const chunks = []
    child.stdout.on('data', (c) => chunks.push(c))
    child.stderr.on('data', (c) => chunks.push(c))
    child.on('error', (err) => {
      chunks.push(Buffer.from(`failed to spawn: ${err.message}\n`))
      resolve({ unit, code: 1, output: Buffer.concat(chunks).toString('utf-8'), ms: Date.now() - started })
    })
    child.on('close', (code) => {
      resolve({
        unit,
        code: code ?? 1,
        output: Buffer.concat(chunks).toString('utf-8'),
        ms: Date.now() - started,
      })
    })
  })
}

async function main() {
  const argv = process.argv.slice(2)
  const units = discoverUnits()

  if (units.length === 0) {
    console.error(
      '::error::No workspace package declares a check script — refusing to report green having run nothing.',
    )
    process.exit(1)
  }

  // --expect-workspaces apps/desktop,web,… : the pinned workspace set. Any
  // listed workspace that contributed ZERO units fails before anything runs.
  const expectIdx = argv.indexOf('--expect-workspaces')
  if (expectIdx !== -1) {
    const expected = (argv[expectIdx + 1] || '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    if (expected.length === 0) {
      console.error('::error::--expect-workspaces given but the list is empty')
      process.exit(1)
    }
    const skipped = findSkippedExpectedWorkspaces(units, expected)
    if (skipped.length > 0) {
      for (const pkg of skipped) {
        console.error(
          `::error::workspace ${pkg} is expected to contribute checks but contributed NONE ` +
            '(its check scripts are missing, renamed, or npm no longer resolves it) ' +
            '— refusing to report green having silently skipped it',
        )
      }
      process.exit(1)
    }
    console.log(
      `workspace expectation holds: all ${expected.length} pinned workspaces contribute checks`,
    )
  }

  if (argv.includes('--list')) {
    for (const u of units) console.log(`${u.pkg} :: ${u.script}`)
    return
  }

  const flagIdx = argv.indexOf('--concurrency')
  const concurrency = Math.max(
    1,
    flagIdx !== -1 ? Number(argv[flagIdx + 1]) : Math.min(units.length, availableParallelism()),
  )

  console.log(`running ${units.length} checks, up to ${concurrency} at a time:`)
  for (const u of units) console.log(`  ${u.pkg} :: ${u.script}`)
  console.log('')

  const queue = [...units]
  /** @type {{unit: {pkg: string, script: string}, code: number, output: string, ms: number}[]} */
  const results = []

  async function worker() {
    for (;;) {
      const unit = queue.shift()
      if (!unit) return
      const res = await runUnit(unit)
      results.push(res)
      const label = `${res.unit.pkg} :: ${res.unit.script}`
      const secs = (res.ms / 1000).toFixed(1)
      const status = res.code === 0 ? 'PASS' : 'FAIL'
      if (IS_CI) console.log(`::group::${status} ${label} (${secs}s)`)
      else console.log(`----- ${status} ${label} (${secs}s) -----`)
      process.stdout.write(res.output.endsWith('\n') ? res.output : res.output + '\n')
      if (IS_CI) console.log('::endgroup::')
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, units.length) }, worker))

  const failed = results.filter((r) => r.code !== 0)
  console.log('\n=== summary ===')
  for (const r of [...results].sort((a, b) => b.ms - a.ms)) {
    console.log(
      `  ${r.code === 0 ? 'pass' : 'FAIL'}  ${(r.ms / 1000).toFixed(1).padStart(6)}s  ${r.unit.pkg} :: ${r.unit.script}`,
    )
  }

  if (failed.length > 0) {
    for (const r of failed) console.error(`::error::${r.unit.pkg} :: ${r.unit.script} failed`)
    console.error(`::error::${failed.length} of ${results.length} checks failed`)
    process.exit(1)
  }
  console.log(`\nall ${results.length} checks passed`)
}

await main()
