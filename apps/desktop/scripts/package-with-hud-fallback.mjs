// An optional helper can be quarantined after electron-builder's file walk.
// Retry only that precise ENOENT, once, without reintroducing the executable.
import fs from 'node:fs'
import path from 'node:path'
import { spawn } from 'node:child_process'
import { hudModifierBinaryRelativePath, warnWindowsHudUnavailable } from './build-hud-modifier-monitor.mjs'

export async function packageWithHudFallback({ command, args, options, app, platform, arch }) {
  /** @param {boolean} omit @returns {Promise<{status: number | null, signal: NodeJS.Signals | null, output: string}>} */
  const run = omit => new Promise((resolve, reject) => {
    const child = spawn(command, args, { ...options, stdio: ['inherit', 'pipe', 'pipe'],
      // Private parent-to-hook signal for this packaging attempt only.
      env: { ...options.env, HERMES_PACKAGING_OMIT_HUD_HELPER: omit ? '1' : '' } })
    let output = ''
    // electron-builder logs errors to stdout; Node stream errors use stderr.
    // Keep progress live and retain only a bounded diagnostic tail.
    const forward = stream => data => {
      stream.write(data)
      output = (output + data.toString()).slice(-65536)
    }
    child.stdout.on('data', forward(process.stdout))
    child.stderr.on('data', forward(process.stderr))
    child.once('error', reject)
    child.once('close', (status, signal) => resolve({ status, signal, output }))
  })
  const helper = path.join(app, 'dist', hudModifierBinaryRelativePath(platform, arch))
  const result = await run(false)
  const normalize = value => String(value).replace(/\\+/g, '/').toLowerCase()
  const helperOpenFailed = result.output.split(/\r?\n/).some(line => line.includes('ENOENT') &&
    normalize(line).split(normalize(helper)).slice(1).some(tail => /^(?:['"\s]|$)/.test(tail)))
  if (platform !== 'win32' || result.signal || !result.status || fs.existsSync(helper) ||
      !helperOpenFailed) return result
  warnWindowsHudUnavailable(helper, 'removed while packaging; retrying once without the optional helper')
  return run(true)
}
