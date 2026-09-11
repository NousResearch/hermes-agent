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
  } catch (error) {
    console.error('[python build]', error.message)
    process.exitCode = Number.isInteger(error.status) && error.status !== 0 ? error.status : 1
  }
}
