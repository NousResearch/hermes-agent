// Runs one shard of the UI vitest suite, deriving the shard index/count from
// the npm script NAME (npm_lifecycle_event), so the name and the flag can
// never disagree. A copy-paste slip like "check:test:ui:shard-2of3" running
// --shard=1/3 would silently skip a third of the suite while CI stays green;
// deriving from the name makes that impossible.
//
// It also validates that this package.json declares exactly the shard family
// 1..M for a single M, so a partial 3→4 migration (adding shard-4of4 without
// updating the siblings) fails loudly instead of dropping coverage.
import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const SHARD_RE = /^check:test:ui:shard-(\d+)of(\d+)$/

const scriptName = process.env.npm_lifecycle_event ?? ''
const match = scriptName.match(SHARD_RE)
if (!match) {
  console.error(
    `run-ui-shard: must be invoked via an npm script named check:test:ui:shard-<N>of<M> (got ${JSON.stringify(scriptName)})`,
  )
  process.exit(1)
}
const [, indexRaw, countRaw] = match
const index = Number(indexRaw)
const count = Number(countRaw)
if (!(index >= 1 && index <= count)) {
  console.error(`run-ui-shard: shard index ${index} out of range 1..${count}`)
  process.exit(1)
}

// The whole family must be exactly 1..M of one M — otherwise a rename or a
// partial count bump leaves a silently untested slice of the suite.
const pkgDir = dirname(dirname(fileURLToPath(import.meta.url)))
const pkg = JSON.parse(readFileSync(join(pkgDir, 'package.json'), 'utf8'))
const family = Object.keys(pkg.scripts ?? {})
  .map((name) => name.match(SHARD_RE))
  .filter(Boolean)
const counts = new Set(family.map((m) => Number(m[2])))
const indices = family.map((m) => Number(m[1])).sort((a, b) => a - b)
const expected = Array.from({ length: count }, (_, i) => i + 1)
if (counts.size !== 1 || indices.length !== count || indices.some((v, i) => v !== expected[i])) {
  console.error(
    `run-ui-shard: shard scripts must form exactly 1..M for a single M; found indices [${indices}] with counts {${[...counts]}}`,
  )
  process.exit(1)
}

// Delegate through test:ui so the vitest command stays single-sourced.
const result = spawnSync(
  'npm',
  ['run', 'test:ui', '--', `--shard=${index}/${count}`, ...process.argv.slice(2)],
  { stdio: 'inherit', cwd: pkgDir },
)
process.exit(result.status ?? 1)
