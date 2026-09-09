import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'vitest'

const PRELOAD = path.join(import.meta.dirname, 'fs-open-limit.cjs')

function runNode(body) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'fsopen-sync-'))
  const script = path.join(dir, 'body.mjs')
  try {
    fs.writeFileSync(script, body)
    return spawnSync(process.execPath, ['--require', PRELOAD, script], {
      cwd: dir,
      encoding: 'utf8',
      timeout: 15000,
      env: { ...process.env, NODE_OPTIONS: '', HERMES_FS_OPEN_LIMIT: '1' },
    })
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
}

test('immediate validation errors preserve throws and rejections without consuming slots', () => {
  const result = runNode(`
import assert from 'node:assert/strict'
import fs from 'node:fs'
for (let i = 0; i < 3; i++) {
  assert.throws(() => fs.open(1, 'r', () => {}), { code: 'ERR_INVALID_ARG_TYPE' })
}
await assert.rejects(fs.promises.open(1, 'r'), { code: 'ERR_INVALID_ARG_TYPE' })
const file = await fs.promises.open(new URL(import.meta.url), 'r')
await file.close()
console.log('SLOTS_RELEASED')
`)
  assert.equal(result.status, 0, result.error?.message || result.stderr)
  assert.match(result.stdout, /SLOTS_RELEASED/)
})

test('a deferred validation throw cannot suppress a settled callback or strand queued work', () => {
  const result = runNode(`
import assert from 'node:assert/strict'
import fs from 'node:fs'
const errors = []
const completed = []
process.on('uncaughtException', error => { errors.push(error) })
function open(name) {
  return new Promise((resolve, reject) => {
    fs.open(new URL(import.meta.url), 'r', (error, fd) => {
      if (error) { reject(error); return }
      fs.close(fd, closeError => {
        if (closeError) { reject(closeError); return }
        completed.push(name)
        resolve()
      })
    })
  })
}
const first = open('first')
fs.open(1, 'r', () => { throw new Error('validation must throw, not call back') })
const last = open('last')
await Promise.all([first, last])
assert.equal(errors.length, 1)
assert.equal(errors[0].code, 'ERR_INVALID_ARG_TYPE')
assert.deepEqual(completed.sort(), ['first', 'last'])
console.log('QUEUE_DRAINED')
`)
  assert.equal(result.status, 0, result.error?.message || result.stderr)
  assert.match(result.stdout, /QUEUE_DRAINED/)
})
