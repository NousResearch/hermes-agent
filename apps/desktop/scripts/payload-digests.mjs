import { runPython } from '../../../scripts/build/python.mjs'
import fs from 'node:fs'
import path from 'node:path'

/** @param {string} payload */
export function rehashPayloadDigests(payload) {
  // Light builds have no payload. Present payloads require valid tool facts.
  if (!fs.existsSync(path.join(payload, 'manifest.json'))) return
  console.log(`[after-pack] refreshing payload hashes after signing: ${payload}`)
  runPython([
    path.resolve(import.meta.dirname, '../../../scripts/bundles/payload.py'), 'rehash', payload],
  { stdio: 'inherit' })
  console.log('[after-pack] payload hash refresh complete')
}
