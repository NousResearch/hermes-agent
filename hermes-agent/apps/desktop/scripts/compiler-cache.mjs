import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * The React Compiler babel pass is ~85% of a production renderer build (≈13 s over ~770
 * files), and every source update rebuilt all of it for a one-file change. Its output is a
 * pure function of the module text, its id and the toolchain, so memoize it on disk by that
 * content key: an update that touched three components re-runs babel on three files. A
 * missing or corrupt entry is just a miss.
 *
 * Only production builds use it: the compiler emits HMR cache-reset code when NODE_ENV is
 * development, and a dev server would grow it one entry per save with no build to prune it.
 *
 * @param {any} plugin the babel plugin whose `transform.handler` is memoized
 * @param {{ command: string, cacheRoot: string, base: string, toolchain: string[], env?: Record<string, string | undefined> }} options
 *   `toolchain` lists the files that pin everything the pass runs (the lockfile, the config
 *   holding the preset); `base` makes cache keys independent of the checkout location.
 */
export function withCompilerCache(plugin, { command, cacheRoot, base, toolchain, env = process.env }) {
  if (command !== 'build') return plugin

  // The toolchain files, this module, and NODE_ENV (which selects the compiler's dev mode)
  // are together the whole identity of the pass.
  const identity = crypto.createHash('sha256')
  for (const file of [...toolchain, fileURLToPath(import.meta.url)]) identity.update(fs.readFileSync(file)).update('\0')
  const generation = identity.update(env.NODE_ENV ?? '').digest('hex').slice(0, 16)
  const dir = path.join(cacheRoot, generation)
  const used = new Set()
  const handler = plugin.transform.handler

  plugin.transform.handler = async function (code, id, opts) {
    const key = crypto.createHash('sha256').update(`${path.relative(base, id)}\0${code}`).digest('hex')
    const file = path.join(dir, `${key}.json`)
    used.add(file)
    try {
      const hit = JSON.parse(fs.readFileSync(file, 'utf8'))
      // The stored map drops sourcesContent: it is the module text the key already hashed.
      if (hit.map) hit.map.sourcesContent = [code]
      return hit
    } catch {
      // miss (or a torn entry): compile and store below
    }
    const result = await handler.call(this, code, id, opts)
    if (result) {
      try {
        fs.mkdirSync(dir, { recursive: true })
        const temp = `${file}.${process.pid}.tmp`
        const map = result.map ? { ...result.map, sourcesContent: undefined } : result.map
        fs.writeFileSync(temp, JSON.stringify({ code: result.code, map }))
        fs.renameSync(temp, file)
      } catch {
        // a read-only or full disk only costs the next build this file's compile
      }
    }
    return result
  }

  // Keep exactly what this build used: entries for since-edited files and other toolchains would
  // otherwise accumulate one generation per update. A watch build never prunes (partial graph).
  const closeBundle = plugin.closeBundle
  plugin.closeBundle = async function (...args) {
    if (!this?.meta?.watchMode) {
      try {
        for (const name of fs.readdirSync(cacheRoot)) {
          if (name !== generation) fs.rmSync(path.join(cacheRoot, name), { recursive: true, force: true })
        }
        for (const name of fs.readdirSync(dir)) {
          if (!used.has(path.join(dir, name))) fs.rmSync(path.join(dir, name), { force: true })
        }
      } catch {
        // nothing cached yet, or a read-only tree
      }
    }
    return closeBundle?.apply(this, args)
  }
  return plugin
}
