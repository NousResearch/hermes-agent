import assert from 'node:assert/strict'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, test } from 'vitest'

import { dirToRemember, nextPickerDefaultPath, readLastPickerDir, writeLastPickerDir } from './picker-state'

afterEach(() => {
  fs.rmSync(STATE_DIR, { recursive: true, force: true })
})

const STATE_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'picker-state-'))
const STATE_PATH = path.join(STATE_DIR, 'picker-state.json')

test('explicit defaultPath always wins over the remembered directory', () => {
  const resolved = nextPickerDefaultPath('/explicit/cwd', '/remembered/dir', () => true)

  assert.equal(resolved, '/explicit/cwd')
})

test('remembered directory applies only when the caller has no opinion', () => {
  const resolved = nextPickerDefaultPath(undefined, '/remembered/dir', () => true)

  assert.equal(resolved, '/remembered/dir')
})

test('a remembered directory that no longer exists is ignored', () => {
  const resolved = nextPickerDefaultPath(undefined, '/gone/dir', () => false)

  assert.equal(resolved, undefined)
})

test('no explicit and no remembered default resolves to undefined (OS default)', () => {
  const resolved = nextPickerDefaultPath(undefined, undefined, () => true)

  assert.equal(resolved, undefined)
})

test('dirToRemember returns the parent of the first picked file', () => {
  assert.equal(dirToRemember(['/home/user/shots/a.png']), path.dirname('/home/user/shots/a.png'))
})

test('dirToRemember tolerates canceled / empty results', () => {
  assert.equal(dirToRemember([]), null)
  assert.equal(dirToRemember(undefined), null)
  assert.equal(dirToRemember(['']), null)
})

test('last-used directory survives a save/load round trip', () => {
  fs.rmSync(STATE_PATH, { force: true })

  assert.equal(readLastPickerDir(STATE_PATH), undefined)

  writeLastPickerDir(STATE_PATH, '/home/user/shots')

  assert.equal(readLastPickerDir(STATE_PATH), '/home/user/shots')
})

test('a corrupt state file reads as "nothing remembered"', () => {
  fs.mkdirSync(STATE_DIR, { recursive: true })
  fs.writeFileSync(STATE_PATH, '{not json')

  assert.equal(readLastPickerDir(STATE_PATH), undefined)
})
