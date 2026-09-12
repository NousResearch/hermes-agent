#!/usr/bin/env bash
# Exercise the published launch contract after relocation. Network isolation is
# the runner's responsibility (Linux CI uses unshare --net), not a lazy-deps flag.
set -euo pipefail
PAYLOAD="${1:?usage: smoke-payload.sh <payload-dir>}"
command -v cygpath >/dev/null 2>&1 && PAYLOAD="$(cygpath -w "$PAYLOAD")"
node --input-type=module - "$PAYLOAD" <<'JS'
import { mkdtempSync, mkdirSync, readFileSync, realpathSync, renameSync, rmSync } from 'node:fs'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { spawnSync } from 'node:child_process'

const original = resolve(process.argv[2])
// Same filesystem: moving a multi-GB payload should not copy it.
const scratch = mkdtempSync(join(dirname(original), '.payload-smoke-'))
const moved = join(scratch, 'relocated payload')
let relocated = false
try {
  renameSync(original, moved)
  relocated = true
  const manifest = JSON.parse(readFileSync(join(moved, 'manifest.json'), 'utf8'))
  const command = manifest.runtime.commands.hermes
  if (!command) throw new Error('manifest has no hermes command')
  const executable = realpathSync(resolve(moved, command))
  const inside = relative(realpathSync(moved), executable)
  if (isAbsolute(command) || isAbsolute(inside) || inside === '..' || inside.startsWith(`..${sep}`)) {
    throw new Error('manifest command escapes the relocated payload')
  }
  const home = join(scratch, 'home')
  mkdirSync(home)
  const env = { HOME: home, USERPROFILE: home, HERMES_HOME: join(home, '.hermes'),
    PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', HERMES_DISABLE_LAZY_INSTALLS: '1',
    UV_OFFLINE: '1', npm_config_offline: 'true' }
  // Keep OS process necessities only; no checkout, Python, PM, or Node overrides.
  for (const [key, value] of Object.entries(process.env)) {
    if (/^(path|systemroot|windir|comspec|pathext|temp|tmp)$/i.test(key)) env[key] = value
  }
  // tools list crosses the real application/config/registry imports, unlike
  // version/help and PM's stdlib bootstrap fast paths.
  for (const args of [['--version'], ['--help'], ['tools', 'list'], ['pm', 'doctor']]) {
    console.log(`— published hermes ${args.join(' ')} (relocated) —`)
    const child = spawnSync(executable, args, {
      cwd: home, env, stdio: 'inherit', timeout: 120000,
    })
    if (child.error) throw child.error
    if (child.status !== 0) throw new Error(`hermes ${args.join(' ')} failed: ${child.status ?? child.signal}`)
  }
  console.log('SMOKE OK')
} finally {
  // Restore even after a command fails, so its artifact remains inspectable.
  if (relocated) renameSync(moved, original)
  rmSync(scratch, { recursive: true, force: true })
}
JS