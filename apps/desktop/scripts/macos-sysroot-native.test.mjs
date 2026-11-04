import { execFileSync, spawnSync } from 'node:child_process'
import { mkdtempSync, readdirSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, resolve } from 'node:path'
import { expect, test } from 'vitest'

import { macosSysroot } from './macos-sysroot.mjs'

const env = { ...process.env, SDKROOT: '' }
const xcrun = args => execFileSync('xcrun', args, { encoding: 'utf8', env }).trim()
const sdkVersion = sdk => xcrun(['--sdk', sdk, '--show-sdk-version']).split('.').map(Number)
const compareVersions = (a, b) => a.map((n, i) => n - (b[i] ?? 0)).find(d => d !== 0) ?? 0
const helpers = [
  ['build-command-screenshot-monitor.mjs', 'native/command-screenshot-monitor'],
  ['build-hud-modifier-monitor.mjs', 'native/darwin-universal/hud-modifier-monitor']
]

// An installed SDK other than the default, older so the active linker accepts
// it (a newer one is the very mismatch #113708 is about). Null when the host
// only has the default installed.
function olderInstalledSdk(defaultSdk) {
  const seen = new Set([realpathSync(defaultSdk)])
  const current = sdkVersion(defaultSdk)
  for (const entry of readdirSync(dirname(defaultSdk)).sort().reverse()) {
    if (!/^MacOSX.*\.sdk$/.test(entry)) continue
    const sdk = resolve(dirname(defaultSdk), entry)
    const real = realpathSync(sdk)
    if (seen.has(real)) continue
    seen.add(real)
    if (compareVersions(sdkVersion(sdk), current) < 0) return sdk
  }
  return null
}

test.skipIf(process.platform !== 'darwin')('builds both universal helpers under every SDK selection rung', () => {
  const dir = mkdtempSync(resolve(tmpdir(), 'hermes-sdk-builds-'))

  try {
    const sdk = macosSysroot(env) ?? xcrun(['--sdk', 'macosx', '--show-sdk-path'])
    const other = olderInstalledSdk(sdk)
    const selections = [
      ['default', ''],
      ['path', sdk],
      ['name', `macosx${sdkVersion(sdk).join('.')}`],
      // A stale SDKROOT must be reported and ignored, never fail the build.
      ['stale', resolve(dir, 'MacOSX99.sdk')],
      ['not-an-sdk', dir],
      ...(other ? [['other', other]] : [])
    ]
    for (const [name, SDKROOT] of selections) {
      for (const [script, relativeBinary] of helpers) {
        const dist = resolve(dir, name, script)
        const build = spawnSync(process.execPath, [resolve(import.meta.dirname, script), '--out-dir', dist], {
          encoding: 'utf8', env: { ...env, SDKROOT }, timeout: 60_000
        })
        expect(build.status, `${script} with ${name} SDK selection\n${build.stderr}`).toBe(0)
        const architectures = xcrun(['lipo', '-archs', resolve(dist, relativeBinary)]).split(/\s+/).sort()
        expect(architectures, `${script} with ${name} SDK selection`).toEqual(['arm64', 'x86_64'])
        if (name === 'stale' || name === 'not-an-sdk') expect(build.stderr).toContain(`SDKROOT=${SDKROOT} is not an installed SDK`)
      }
    }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}, 180_000)
