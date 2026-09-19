import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

import { detectBundleSwap, readBundleSwapStamp } from './bundle-swap'

const RUNNING = { builtAt: '2026-08-29T04:00:00.000Z', commit: 'a'.repeat(40), source: 'local' }

it('reads only the installed stamp for swap detection, without schema conversion', () => {
  const resources = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-swap-'))

  try {
    expect(readBundleSwapStamp(resources)).toBeNull()

    const file = path.join(resources, 'install-stamp.json')

    fs.writeFileSync(file, JSON.stringify(RUNNING))
    expect(readBundleSwapStamp(resources)).toEqual(RUNNING)

    const replaced = { ...RUNNING, commit: 'b'.repeat(40) }

    fs.writeFileSync(file, JSON.stringify(replaced))
    expect(detectBundleSwap(RUNNING, readBundleSwapStamp(resources))).toBe(true)
  } finally {
    fs.rmSync(resources, { recursive: true, force: true })
  }
})

describe('detectBundleSwap', () => {
  it('reports a swap when the on-disk stamp carries a different commit', () => {
    const onDisk = { ...RUNNING, commit: 'b'.repeat(40) }

    expect(detectBundleSwap(RUNNING, onDisk)).toBe(true)
  })

  it('reports a swap when the same commit was rebuilt (builtAt moved)', () => {
    const onDisk = { ...RUNNING, builtAt: '2026-08-31T23:55:41.149Z' }

    expect(detectBundleSwap(RUNNING, onDisk)).toBe(true)
  })

  // The Windows locked-binary case (#92233): the swap leg failed, so the
  // bundle on disk is still the one we are running. A relaunch would repair
  // nothing and cost the user their window.
  it('is quiet when the on-disk stamp matches the running one', () => {
    expect(detectBundleSwap(RUNNING, { ...RUNNING })).toBe(false)
  })

  it('is quiet without a running stamp (dev runs)', () => {
    expect(detectBundleSwap(null, { ...RUNNING })).toBe(false)
  })

  it('is quiet without an on-disk stamp (unreadable resources)', () => {
    expect(detectBundleSwap(RUNNING, null)).toBe(false)
  })

  it('is quiet on a fallback stamp on either side (non-git build)', () => {
    const fallbackTagged = { ...RUNNING, source: 'fallback' }
    const fallbackCommit = { ...RUNNING, commit: '0'.repeat(40) }

    expect(detectBundleSwap(fallbackTagged, { ...RUNNING, commit: 'b'.repeat(40) })).toBe(false)
    expect(detectBundleSwap(RUNNING, fallbackCommit)).toBe(false)
  })


})
