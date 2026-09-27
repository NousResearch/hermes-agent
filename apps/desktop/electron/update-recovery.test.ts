import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, expect, test } from 'vitest'

import { markRecoveryHealthy, prepareRecovery, readRecovery, restoreRecoveryFiles } from './update-recovery'

const dirs: string[] = []
afterEach(() => dirs.splice(0).forEach(dir => fs.rmSync(dir, { recursive: true, force: true })))

async function fixture() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'czesiek-recovery-'))
  dirs.push(dir)
  const home = path.join(dir, 'home')
  const installRoot = path.join(dir, 'app')
  const root = path.join(dir, 'recovery')
  fs.mkdirSync(home)
  fs.mkdirSync(installRoot)
  fs.writeFileSync(path.join(home, 'config.yaml'), 'version: old')
  fs.writeFileSync(path.join(home, 'MEMORY.md'), 'remember this')
  fs.writeFileSync(path.join(installRoot, 'Czesiek.exe'), 'old binary')
  const record = await prepareRecovery({ home, installRoot, root, version: 'test', executable: 'Czesiek.exe' })

  return { dir, home, installRoot, root, record }
}

test('restores app and memory together, preserving the replaced version and clearing stale WAL', async () => {
  const { home, installRoot, root, record, dir } = await fixture()
  fs.writeFileSync(path.join(home, 'config.yaml'), 'version: new')
  fs.writeFileSync(path.join(home, 'state.db-wal'), 'new WAL must not accompany old DB')
  fs.writeFileSync(path.join(installRoot, 'Czesiek.exe'), 'new binary')
  markRecoveryHealthy(root)
  expect(readRecovery(root)?.status).toBe('healthy')
  restoreRecoveryFiles(record, root)
  expect(fs.readFileSync(path.join(home, 'config.yaml'), 'utf8')).toBe('version: old')
  expect(fs.readFileSync(path.join(home, 'MEMORY.md'), 'utf8')).toBe('remember this')
  expect(fs.existsSync(path.join(home, 'state.db-wal'))).toBe(false)
  expect(fs.readFileSync(path.join(installRoot, 'Czesiek.exe'), 'utf8')).toBe('old binary')
  const preserved = fs.readdirSync(dir).find(name => name.startsWith('app.before-restore-'))!
  expect(fs.readFileSync(path.join(dir, preserved, 'Czesiek.exe'), 'utf8')).toBe('new binary')
  expect(readRecovery(root)?.status).toBe('restored')
})

test('a damaged data snapshot aborts restore and rolls back all already changed files', async () => {
  const { home, installRoot, root, record } = await fixture()
  fs.writeFileSync(path.join(home, 'config.yaml'), 'keep current')
  fs.writeFileSync(path.join(installRoot, 'Czesiek.exe'), 'keep binary')
  fs.unlinkSync(path.join(record.snapshot, 'home', 'MEMORY.md'))
  expect(() => restoreRecoveryFiles(record, root)).toThrow('Przywracanie przerwane')
  expect(fs.readFileSync(path.join(home, 'config.yaml'), 'utf8')).toBe('keep current')
  expect(fs.readFileSync(path.join(installRoot, 'Czesiek.exe'), 'utf8')).toBe('keep binary')
})
