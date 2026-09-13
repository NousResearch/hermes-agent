import { execFile } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { createRequire } from 'node:module'
import path from 'node:path'
import { promisify } from 'node:util'

import { expect, test } from 'vitest'

const require = createRequire(import.meta.url)
const { wrapDmgbuildExecFile } = require('./dmgbuild-diagnostics.cjs')

test('the resolved dmgbuild uses its paired Python and preserves args, results and live diagnostics', () => {
  const invocations = []
  const output = []
  const child = { stderr: new EventEmitter() }
  const original = (...args) => {
    invocations.push(args)
    return child
  }
  const wrapped = wrapDmgbuildExecFile(original, chunk => output.push(chunk))
  const vendor = path.resolve('tool cache', 'dmgbuild bundle')
  const args = ['-s', 'settings with spaces.json', 'Install Hermes Agent', 'output.dmg']
  const options = { maxBuffer: 1024, env: { KEEP: 'value', PYTHONPATH: 'old' } }
  const callback = () => {}

  expect(wrapped(path.join(vendor, 'dmgbuild'), args, options, callback)).toBe(child)
  const invocation = invocations[0]
  expect(invocation[0]).toBe(path.join(vendor, 'python', 'bin', 'python3'))
  expect(path.basename(invocation[1][0])).toBe('dmgbuild_diagnostics.py')
  expect(invocation[1].slice(1)).toEqual(args)
  expect(invocation[2]).toEqual({ ...options, env: { KEEP: 'value', PYTHONPATH: path.join(vendor, 'python', 'lib') } })
  expect(invocation[3]).toBe(callback)
  expect(options.env.PYTHONPATH).toBe('old')
  child.stderr.emit('data', Buffer.from('[dmg-detach] actual child output'))
  expect(output[0].toString()).toContain('[dmg-detach]')
})

test('promisified callers retain stdout, stderr and the real child handle', async () => {
  const wrapped = wrapDmgbuildExecFile(execFile)
  const pending = promisify(wrapped)(process.execPath, [
    '-e',
    'process.stdout.write("out"); process.stderr.write("err")'
  ])
  expect(pending.child).toBeDefined()
  expect(await pending).toEqual({ stdout: 'out', stderr: 'err' })
  await expect(
    promisify(wrapped)(process.execPath, [
      '-e',
      'process.stdout.write("out"); process.stderr.write("err"); process.exit(23)'
    ])
  ).rejects.toMatchObject({ code: 23, stdout: 'out', stderr: 'err' })
})

test('other executables and explicit custom dmgbuild overrides pass through unchanged', () => {
  const calls = []
  const original = (...args) => calls.push(args)
  const wrapped = wrapDmgbuildExecFile(original)
  const callback = () => {}
  wrapped('/usr/bin/true', callback)
  expect(calls[0]).toEqual(['/usr/bin/true', callback])
  const invocation = [
    path.resolve('custom', 'dmgbuild'),
    ['--help'],
    { env: { CUSTOM_DMGBUILD_PATH: '/custom/dmgbuild' } },
    callback
  ]
  wrapped(...invocation)
  expect(calls[1]).toEqual(invocation)
})
