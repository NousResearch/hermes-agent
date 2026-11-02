// Can a packaged desktop app built at an earlier commit stand for HEAD?
//
// The freshness receipt already proves the bundle's inputs are byte-identical
// (sources, lockfile, build scripts, prepared icons/native trees, renderer env,
// Node/platform/arch). The one input it cannot see is the install stamp a build
// at HEAD would bake, because only a build rewrites build/install-stamp.json.
// This compares that would-be stamp with the baked one: provenance that moves
// with every commit (commit, its date, the dirty flag, the build clock) is
// expected to differ; anything else (branch, payload, update mechanism, tag,
// channel, version strings) changes the shipped app and means "build".
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { pathToFileURL } from 'node:url'
import { isMain } from './frontend-common.mjs'

const commitProvenance = new Set(['commit', 'commitDate', 'dirty', 'builtAt'])

/** Stamp fields a build at HEAD would bake differently from ``baked`` (empty = it may stand). */
export async function stampDrift({ source, baked, env = process.env, platform = process.platform }) {
  const stampModule = join(resolve(source), 'apps/desktop/scripts/write-build-stamp.mjs')
  const { resolveStamp, buildStampPayload } = await import(pathToFileURL(stampModule).href)
  const fresh = buildStampPayload(resolveStamp({ env, repoRoot: resolve(source) }), env, platform, null)
  return [...new Set([...Object.keys(baked), ...Object.keys(fresh)])].sort()
    .filter(key => !commitProvenance.has(key) && JSON.stringify(baked[key]) !== JSON.stringify(fresh[key]))
}

if (isMain(import.meta.url)) {
  const { values } = parseArgs({ options: { source: { type: 'string' }, baked: { type: 'string' } } })
  if (!values.source || !values.baked) throw new Error('--source and --baked are required')
  const baked = JSON.parse(readFileSync(values.baked, 'utf8'))
  console.log(JSON.stringify(await stampDrift({ source: values.source, baked })))
}
