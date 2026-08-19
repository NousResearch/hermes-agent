import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, test } from 'vitest'

const POSIX = resolve(process.env.POSIX_SH_PATH ?? '../scripts/desktop-update/posix.sh')
const LIB = resolve(process.env.NODE_VERSION_CHECK_PATH ?? '../scripts/lib/node-version-check.sh')

/** Run posix.sh --self-test-node-gate and return the decision line. */
function gate(version: string): string {
  return execFileSync('bash', [POSIX, '--self-test-node-gate', '--node-version', version], {
    encoding: 'utf8',
  }).trim()
}

/** Source the shared lib directly and return its decision for a version. */
function libGate(version: string): string {
  const out = execFileSync('bash', ['-c', `source "$1" && node_satisfies_build "$2" && echo compatible || echo incompatible`, 'bash', LIB, version], {
    encoding: 'utf8',
  }).trim()

  return out
}

describe('posix.sh node build gate (--self-test-node-gate)', () => {
  test('accepts supported lines (22.22+, 24, 26+)', () => {
    expect(gate('v22.22.0')).toBe('compatible')
    expect(gate('v22.99.0')).toBe('compatible')
    expect(gate('v24.0.0')).toBe('compatible')
    expect(gate('v26.0.0')).toBe('compatible')
  })

  test('rejects odd-numbered releases (23, 25) and too-old 22.x', () => {
    expect(gate('v23.0.0')).toBe('incompatible')
    expect(gate('v25.9.0')).toBe('incompatible')
    expect(gate('v22.21.0')).toBe('incompatible')
    expect(gate('v20.19.0')).toBe('incompatible')
  })

  test('boundary matrix stays stable (#84397 alignment)', () => {
    // The install path (install.sh) gates on the same lines; keep the
    // boundary exact so the two paths can't drift apart.
    const expected: Record<string, string> = {
      'v22.21.0': 'incompatible',
      'v22.22.0': 'compatible',
      'v23.0.0': 'incompatible',
      'v24.0.0': 'compatible',
      'v25.0.0': 'incompatible',
      'v26.0.0': 'compatible',
    }

    for (const [v, want] of Object.entries(expected)) {
      expect(gate(v), `version ${v}`).toBe(want)
    }
  })
})

describe('posix.sh script exists', () => {
  test('desktop-update/posix.sh is present', () => {
    expect(existsSync(POSIX)).toBe(true)
  })
})

describe('shared lib node-version-check.sh (single source of truth)', () => {
  test('lib exists', () => {
    expect(existsSync(LIB)).toBe(true)
  })

  test('lib predicate matches posix.sh self-test gate', () => {
    const versions = ['v22.21.0', 'v22.22.0', 'v23.0.0', 'v24.0.0', 'v25.0.0', 'v26.0.0']

    for (const v of versions) {
      expect(libGate(v), `lib gate ${v}`).toBe(gate(v))
    }
  })
})

describe('prepare_node_for_build candidate selection', () => {
  const TMP = '/tmp/hermes-node-gate-test'

  /** Extract prepare_node_for_build into a temp file, then run it. */
  function runPrepare(prelude: string): string {
    const script = `
set -u
log() { echo "LOG: $1"; }
source "$1"
source "$3"
${prelude}
prepare_node_for_build
echo "FINAL_PATH=$PATH"
`

    // Extract the function body to a temp file so bash sources it as code
    // (command substitution would word-split the multi-line function).
    const fnFile = `${TMP}/prepare_node_for_build.sh`
    execFileSync('bash', ['-c', `sed -n '/^prepare_node_for_build() {/,/^}/p' "$1" > "$2"`, 'bash', POSIX, fnFile])

    return execFileSync('bash', ['-c', script, 'bash', LIB, POSIX, fnFile], { encoding: 'utf8' })
  }

  test('skips a candidate that has node but no npm', () => {
    execFileSync('bash', ['-c', `rm -rf ${TMP} && mkdir -p ${TMP}/node-only/bin`])
    execFileSync('bash', ['-c', `printf '#!/bin/sh\\necho v24.0.0\\n' > ${TMP}/node-only/bin/node && chmod +x ${TMP}/node-only/bin/node`])

    const out = runPrepare(`
      export HERMES_NODE_CANDIDATE_DIRS="${TMP}/node-only/bin"
      export PATH="/nonexistent:/usr/bin:/bin"
    `)

    expect(out).toContain('WARNING: no compatible Node found')
  })

  test('accepts a candidate that has both node and npm', () => {
    execFileSync('bash', ['-c', `rm -rf ${TMP} && mkdir -p ${TMP}/full/bin`])
    execFileSync('bash', ['-c', `printf '#!/bin/sh\\necho v24.0.0\\n' > ${TMP}/full/bin/node && printf '#!/bin/sh\\necho npm\\n' > ${TMP}/full/bin/npm && chmod +x ${TMP}/full/bin/node ${TMP}/full/bin/npm`])

    const out = runPrepare(`
      export HERMES_NODE_CANDIDATE_DIRS="${TMP}/full/bin"
      export PATH="/nonexistent:/usr/bin:/bin"
    `)

    expect(out).toContain(`using ${TMP}/full/bin`)
    expect(out).toContain(`FINAL_PATH=${TMP}/full/bin`)
  })
})
