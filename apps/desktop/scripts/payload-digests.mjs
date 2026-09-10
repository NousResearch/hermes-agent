import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

/** @param {string} payload */
export function rehashPayloadDigests(payload) {
  // Light builds have no payload. Present payloads require valid tool facts.
  if (!fs.existsSync(path.join(payload, 'manifest.json'))) return
  execFileSync('uv', ['run', '--no-project', 'python',
    path.resolve(import.meta.dirname, '../../../scripts/bundles/payload.py'), 'rehash', payload],
  { stdio: 'inherit' })
}
