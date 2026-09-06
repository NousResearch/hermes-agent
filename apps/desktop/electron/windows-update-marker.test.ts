import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { transferUpdateMarkerIfOwnedBy } from './windows-update-marker'

test.runIf(process.platform === 'win32')('repair transfer changes only the exact current claim', async () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-repair-marker-'))
  const marker = path.join(home, '.hermes-update-in-progress')
  const expected = { pid: process.pid, startedAt: Math.floor(Date.now() / 1000) }
  const next = { pid: process.pid, startedAt: expected.startedAt + 1 }
  const body = (claim: typeof expected) => `${claim.pid}\n${claim.startedAt}\n`

  try {
    fs.writeFileSync(marker, body(expected))
    expect(await transferUpdateMarkerIfOwnedBy(home, expected, next)).toBe(true)
    expect(fs.readFileSync(marker, 'utf8')).toBe(body(next))
    expect(await transferUpdateMarkerIfOwnedBy(home, expected, next)).toBe(false)
    expect(fs.readFileSync(marker, 'utf8')).toBe(body(next))
    fs.unlinkSync(marker)
    expect(await transferUpdateMarkerIfOwnedBy(home, expected, next)).toBe(false)
    expect(fs.existsSync(marker)).toBe(false)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
}, 20_000)
