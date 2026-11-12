import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test, vi } from 'vitest'
import { generateIcons } from '../scripts/generate-icons.mjs'

test('the real wrapper builds and checks native icon artifacts in an independent output', () => {
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-icons-'))
  const env = { ...process.env, HERMES_HOME: path.join(out, 'home'),
    HERMES_RUNTIME_DIR: path.join(out, 'tools'), HERMES_PAYLOAD_TAG: 'v1.2.3', HERMES_BUILD_COMMIT: '',
    PYTHONPATH: path.join(out, 'foreign-site'), PYTHONHOME: path.join(out, 'foreign-python') }
  try {
    const args = ['--source', fileURLToPath(new URL('..', import.meta.url)), '--out', out]
    expect(generateIcons(args, { env })).toBe(0)
    const png = fs.readFileSync(path.join(out, 'apps/desktop/assets/icon.png'))
    expect(png.subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    expect([png.readUInt32BE(16), png.readUInt32BE(20)]).toEqual([1024, 1024])
    expect(generateIcons([...args, '--check'], { env })).toBe(0)
    expect(env.PYTHONPATH).toBe(path.join(out, 'foreign-site'))
  } finally {
    fs.rmSync(out, { recursive: true, force: true })
  }
}, 180_000)

test('failed icon processes cannot report a successful build', () => {
  expect(generateIcons([], { run: () => ({ status: 7 }), env: {} })).toBe(7)
  expect(generateIcons([], { run: () => ({ status: null, signal: 'SIGTERM' }), env: {} })).toBe(1)
  const error = vi.spyOn(console, 'error').mockImplementation(() => {})
  try {
    expect(generateIcons([], { run: () => ({ error: new Error('prepared Python missing') }), env: {} })).toBe(1)
    expect(error).toHaveBeenCalledWith(expect.stringContaining('failed to launch'), 'prepared Python missing')
  } finally {
    error.mockRestore()
  }
})
