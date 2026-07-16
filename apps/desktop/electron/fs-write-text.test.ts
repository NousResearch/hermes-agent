import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it } from 'vitest'

import { sha256Bytes, writeTextFileCas } from './fs-write-text'

const roots: string[] = []

function tempRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-fs-write-text-'))

  roots.push(root)

  return root
}

afterEach(() => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

describe('writeTextFileCas', () => {
  it('atomically replaces the expected revision and returns its next hash', async () => {
    const file = path.join(tempRoot(), 'plan.md')
    fs.writeFileSync(file, 'old')

    const result = await writeTextFileCas(file, 'new', sha256Bytes('old'))

    expect(fs.readFileSync(file, 'utf8')).toBe('new')
    expect(result).toEqual({ contentHash: sha256Bytes('new'), path: file })
  })

  it('rejects stale revisions without modifying the file', async () => {
    const file = path.join(tempRoot(), 'plan.md')
    fs.writeFileSync(file, 'newer')

    await expect(writeTextFileCas(file, 'mine', sha256Bytes('old'))).rejects.toThrow('FILE_CHANGED')
    expect(fs.readFileSync(file, 'utf8')).toBe('newer')
  })

  it('serializes competing saves so only the first expected revision wins', async () => {
    const file = path.join(tempRoot(), 'plan.md')
    fs.writeFileSync(file, 'base')
    const expectedHash = sha256Bytes('base')

    const outcomes = await Promise.allSettled([
      writeTextFileCas(file, 'first', expectedHash),
      writeTextFileCas(file, 'second', expectedHash)
    ])

    expect(outcomes.filter(item => item.status === 'fulfilled')).toHaveLength(1)
    expect(outcomes.filter(item => item.status === 'rejected')).toHaveLength(1)
    expect(['first', 'second']).toContain(fs.readFileSync(file, 'utf8'))
  })
})
