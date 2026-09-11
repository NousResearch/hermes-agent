import { spawnSync } from 'node:child_process'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'
import { fileURLToPath } from 'node:url'
import { expect, test, vi } from 'vitest'
import { generateIcons } from '../scripts/generate-icons.mjs'

test('icon builds delegate environment preparation to PM using the prepared Python', () => {
  const run = vi.fn(() => ({ status: 0 }))
  const root = path.resolve('icon-build-fixture')
  const env = { PATH: 'tools', HERMES_PYTHON: 'prepared-python', VIRTUAL_ENV: 'runtime-venv', PYTHONPATH: 'payload-libraries', PYTHONHOME: 'payload-python' }
  expect(generateIcons(['--check'], { root, run, env })).toBe(0)
  expect(run).toHaveBeenCalledExactlyOnceWith(env.HERMES_PYTHON, [
    path.join(root, 'scripts', 'build', 'icon_environment.py'), '--source', root, '--out', root, '--check'
  ], { cwd: root, stdio: 'inherit', windowsHide: true, env: { PATH: 'tools', HERMES_PYTHON: env.HERMES_PYTHON, VIRTUAL_ENV: 'runtime-venv' } })
  expect(env.PYTHONPATH).toBe('payload-libraries')
})

test('icon preparation passes explicit source and independent output roots', () => {
  const run = vi.fn(() => ({ status: 0 }))
  const source = path.resolve('source with spaces')
  const out = path.resolve('icon outputs')
  expect(generateIcons(['--source', source, '--out', out], { run, env: {} })).toBe(0)
  const [python, args, options] = run.mock.calls[0]
  expect(python).toBe('python')
  expect(args.slice(-4)).toEqual(['--source', source, '--out', out])
  expect(options.cwd).toBe(source)
})

test('the PM driver owns a temporary group-only environment and preserves generator argv and status', () => {
  const result = spawnSync(process.env.HERMES_PYTHON || 'python', ['-c', `
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from scripts.build import icon_environment

with TemporaryDirectory() as directory:
    source = Path(directory) / 'source with spaces'
    source.mkdir()
    argv = ['--source', str(source), '--out', str(Path(directory) / 'icon outputs'), '--check']
    outputs = []
    def build(**options):
        output = options.pop('out')
        assert output.parent.is_dir() and not output.exists()
        assert output.name == 'venv'
        assert options == dict(source=source, groups=['icon-build'], only_groups=True,
                               explicit=True, cache=source / '.cache/icon-build')
        outputs.append(output)
        return Path(sys.executable)
    def generate(command, **options):
        assert outputs[0].parent.is_dir()
        assert command == [sys.executable, '-I', str(Path(icon_environment.__file__).resolve().parents[1] / 'generate_icons.py'), *argv]
        assert options == dict(cwd=source)
        return subprocess.CompletedProcess(command, 7)
    with patch.object(icon_environment.pm, 'build_environment', side_effect=build) as prepare, patch.object(icon_environment.subprocess, 'run', side_effect=generate) as run:
        assert icon_environment.main(argv) == 7
        prepare.assert_called_once()
        run.assert_called_once()
    assert not outputs[0].parent.exists()
`], { cwd: fileURLToPath(new URL('..', import.meta.url)), encoding: 'utf8' })
  expect(result.error).toBeUndefined()
  expect(result.status, result.stderr).toBe(0)
})

test.each([
  { HERMES_PAYLOAD_TAG: 'v1.2.3', HERMES_BUILD_COMMIT: '' },
  { HERMES_PAYLOAD_TAG: 'v1.2.3-canary.20260911010203', HERMES_BUILD_COMMIT: '' },
  { HERMES_PAYLOAD_TAG: '', HERMES_BUILD_COMMIT: 'abcdef0'.padEnd(40, '1') }
])('build identity reaches the child unchanged with separate source/output roots: %j', (identity) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-icon-env-'))
  try {
    const source = path.join(root, 'separate source')
    const out = path.join(root, 'generated output')
    const driver = path.join(root, 'scripts', 'build')
    fs.mkdirSync(driver, { recursive: true })
    fs.mkdirSync(source)
    // Observe the real subprocess environment at the PM-driver boundary.
    fs.writeFileSync(path.join(driver, 'icon_environment.py'), `
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1])
out.mkdir()
(out / 'child.json').write_text(json.dumps({
    'identity': {key: os.environ.get(key) for key in ['HERMES_PAYLOAD_TAG', 'HERMES_BUILD_COMMIT']},
    'source': sys.argv[sys.argv.index('--source') + 1],
    'cwd': os.getcwd(),
}))
`)
    expect(generateIcons(['--source', source, '--out', out], {
      root, env: { ...process.env, ...identity }
    })).toBe(0)
    expect(JSON.parse(fs.readFileSync(path.join(out, 'child.json'), 'utf8'))).toEqual({
      identity, source, cwd: source
    })
    expect(fs.readdirSync(source)).toEqual([])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

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
