import { execFileSync } from 'node:child_process'
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, readlinkSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { expect, test } from 'vitest'
import afterPack from './after-pack.mjs'

// This is the Linux afterPack path on a real Linux host, not a fake host flag.
// macOS adds Developer ID signing; its native release lane owns that proof.
test.runIf(process.platform === 'linux')('afterPack uses the provisioned Python to repair actual payload links', async () => {
  const directory = mkdtempSync(path.join(tmpdir(), 'after-pack-toolchain-'))
  const payload = path.join(directory, 'resources', 'agent-payload')
  const store = path.join(payload, 'tools', 'python', 'bin')
  const venv = path.join(payload, 'venv', 'bin')
  const python = execFileSync('python3', ['-c', 'import sys; print(sys.executable)'], { encoding: 'utf8' }).trim()
  const previous = { HERMES_PYTHON: process.env.HERMES_PYTHON, PATH: process.env.PATH, UV_PYTHON: process.env.UV_PYTHON, PYTHONPATH: process.env.PYTHONPATH }
  const observed = path.join(directory, 'interpreter.json')
  const identity = 'import json,os,sys; print(json.dumps(os.path.realpath(sys.executable)))'
  const expected = JSON.parse(execFileSync(python, ['-c', identity], { encoding: 'utf8' }))
  try {
    // Record the interpreter that actually executes the relocation script.
    writeFileSync(path.join(directory, 'sitecustomize.py'), `import json,os,sys\nfrom pathlib import Path\nPath(${JSON.stringify(observed)}).write_text(json.dumps(os.path.realpath(sys.executable)), encoding="utf-8")\n`)
    process.env.PYTHONPATH = directory
    process.env.HERMES_PYTHON = python
    process.env.UV_PYTHON = path.join(directory, 'not-the-build-python')
    writeFileSync(path.join(directory, 'uv'), '#!/bin/sh\nexit 91\n')
    chmodSync(path.join(directory, 'uv'), 0o755)
    process.env.PATH = `${directory}${path.delimiter}${process.env.PATH}`
    mkdirSync(store, { recursive: true })
    mkdirSync(venv, { recursive: true })
    writeFileSync(path.join(payload, 'manifest.json'), '{}')
    writeFileSync(path.join(store, 'python3'), 'payload interpreter link target')
    symlinkSync('/builder/tools/python/bin/python3', path.join(venv, 'python'))
    symlinkSync('python', path.join(venv, 'python3'))
    await afterPack({ electronPlatformName: process.platform, appOutDir: directory })
    expect(JSON.parse(readFileSync(observed, 'utf8'))).toBe(expected)
    expect(readlinkSync(path.join(venv, 'python'))).toBe('../../tools/python/bin/python3')
    expect(readFileSync(path.join(venv, 'python3'), 'utf8')).toBe('payload interpreter link target')
  } finally {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
    rmSync(directory, { recursive: true, force: true })
  }
})
