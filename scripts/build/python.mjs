#!/usr/bin/env node
/** Run build helpers with the prepared Python, never an ambient uv environment. */
import { execFileSync } from 'node:child_process'
import { pathToFileURL } from 'node:url'

/**
 * @param {string[]} args
 * @param {import('node:child_process').ExecFileSyncOptions} [options]
 */
export function runPython(args, options = {}) {
  const env = options.env ?? process.env
  return execFileSync(env.HERMES_PYTHON || 'python', args, { stdio: 'inherit', ...options, env })
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    runPython(process.argv.slice(2))
  } catch (/** @type {unknown} */ error) {
    // execFileSync throws a child_process exception carrying .status; anything
    // else is reported as-is so a missing Python still prints a useful line.
    const detail = error instanceof Error ? error.message : String(error)
    console.error('[python build]', detail)
    const status = typeof (/** @type {any} */ (error).status) === 'number' ? /** @type {any} */ (error).status : 0
    process.exitCode = status !== 0 ? status : 1
  }
}
