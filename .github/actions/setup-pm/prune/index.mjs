import { spawnSync } from 'node:child_process'
import { appendFileSync } from 'node:fs'
import { pathToFileURL } from 'node:url'

// Composite actions cannot register a post step. Called after actions/cache,
// this action's post runs first, pruning the cache just before it is saved.
export function run(env, execute = spawnSync) {
  if (!env.STATE_python) {
    for (const [key, value] of Object.entries({ python: env.INPUT_PYTHON, cache: env.INPUT_CACHE })) {
      if (!value || /[\r\n\0]/.test(value)) throw new Error(`invalid ${key}`)
      appendFileSync(env.GITHUB_STATE, `${key}=${value}\n`, 'utf8')
    }
    return
  }
  const result = execute(env.STATE_python, ['-m', 'pm.build_env', '--prune-cache', '--cache', env.STATE_cache, '--ci'], {
    env,
    stdio: 'inherit',
  })
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(`PM cache prune failed: ${result.status}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  run(process.env)
}
