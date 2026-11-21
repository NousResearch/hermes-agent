#!/usr/bin/env node
/** Run build helpers with the prepared Python, never an ambient uv environment. */
import { execFileSync } from 'node:child_process'
import { existsSync, realpathSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

/**
 * @param {string[]} args
 * @param {import('node:child_process').ExecFileSyncOptions} [options]
 */
export function runPython(args, options = {}) {
  const env = options.env ?? process.env
  if (!env.HERMES_PYTHON) {
    throw new Error('Set HERMES_PYTHON to the prepared build interpreter')
  }
  return execFileSync(env.HERMES_PYTHON, args, { stdio: 'inherit', ...options, env })
}

// Node resolves symlinks in import.meta.url but not in argv[1]; compare canonical paths.
const mainEntry = process.argv[1] ? resolve(process.argv[1]) : ''
if (mainEntry && existsSync(mainEntry) && import.meta.url === pathToFileURL(realpathSync(mainEntry)).href) {
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
