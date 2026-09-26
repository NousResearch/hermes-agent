import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import assert from 'node:assert/strict'
import { afterEach, expect, test, vi } from 'vitest'

// Regression for #123216: verifyChunksParse used to spawn one
// `node --check` child per emitted chunk (~960 in a real renderer build,
// seconds each on Windows). It ran silent far past the desktop-update
// hand-off's 600s idle watchdog, so every Windows Desktop update died with
// exit code 124. The whole assets tree must be covered by ONE child process
// regardless of chunk count, with the chunk list on stdin (Windows env blocks
// cap at ~32 KiB), never as per-file argv.
vi.mock('node:child_process', () => ({
  spawnSync: vi.fn(),
}))

const { spawnSync } = await import('node:child_process')
const { checkDistBuilt } = await import('../scripts/assert-dist-built.mjs')

afterEach(() => {
  vi.mocked(spawnSync).mockReset()
})

function makeDist(chunkCount) {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-assert-dist-batch-'))
  const distDir = path.join(tempRoot, 'dist')
  fs.mkdirSync(path.join(distDir, 'assets'), { recursive: true })
  fs.writeFileSync(path.join(distDir, 'index.html'), '<!doctype html>', 'utf8')
  for (let i = 0; i < chunkCount; i++) {
    fs.writeFileSync(path.join(distDir, 'assets', `chunk-${i}-abc123.js`), 'export const a = 1', 'utf8')
  }
  return { tempRoot, distDir }
}

test('checkDistBuilt syntax-checks every chunk in a single node child', () => {
  const { tempRoot, distDir } = makeDist(200)
  vi.mocked(spawnSync).mockImplementation(() => ({ status: 0, stdout: '', stderr: '' }))
  try {
    assert.deepEqual(checkDistBuilt(distDir), { ok: true })
    expect(vi.mocked(spawnSync)).toHaveBeenCalledTimes(1)
    const [cmd, argv, options] = vi.mocked(spawnSync).mock.calls[0]
    assert.equal(cmd, process.execPath)
    // argv carries only the fixed probe: no per-chunk file arguments, the
    // chunk list travels over stdin via options.input instead.
    const chunkArgs = argv.filter(a => typeof a === 'string' && a.includes('chunk-'))
    assert.equal(chunkArgs.length, 0)
    const input = JSON.parse(options.input)
    assert.equal(input.length, 200)
    assert.ok(input.every(f => typeof f === 'string' && f.endsWith('.js')))
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})

test('a probe child that dies without a JSON report still fails the check', () => {
  const { tempRoot, distDir } = makeDist(3)
  vi.mocked(spawnSync).mockImplementation(() => ({
    status: 1,
    stdout: '',
    stderr: '(node:1) ExperimentalWarning: VM Modules is an experimental feature\nError: node probe crashed\n    at foo',
  }))
  try {
    const result = checkDistBuilt(distDir)
    assert.equal(result.ok, false)
    assert.match(result.error, /not valid ES module syntax/)
    assert.match(result.error, /node probe crashed/)
  } finally {
    fs.rmSync(tempRoot, { recursive: true, force: true })
  }
})
