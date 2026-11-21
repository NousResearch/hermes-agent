import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, it } from 'vitest'

import { macosFocusActive } from './do-not-disturb'

it('reads macOS Focus from live assertions and fails open on a missing or corrupt store', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-dnd-'))
  const store = path.join(dir, 'Assertions.json')
  const write = (value: unknown) => fs.writeFileSync(store, typeof value === 'string' ? value : JSON.stringify(value))

  try {
    expect(await macosFocusActive(store)).toBe(false)
    write({ data: [{ storeAssertionRecords: [{ assertionDetails: { assertionDetailsModeIdentifier: 'dnd' } }] }] })
    expect(await macosFocusActive(store)).toBe(true)
    write({ data: [{ storeAssertionRecords: [] }] })
    expect(await macosFocusActive(store)).toBe(false)
    write('{not json')
    expect(await macosFocusActive(store)).toBe(false)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
