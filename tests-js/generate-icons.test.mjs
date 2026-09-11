import path from 'node:path'
import { expect, test, vi } from 'vitest'
import { generateIcons } from '../scripts/generate-icons.mjs'

test('icon builds use only the locked build group outside every application venv', () => {
  const run = vi.fn(() => ({ status: 0 }))
  const root = path.resolve('icon-build-fixture')
  const env = { PATH: 'tools', UV_PYTHON: 'pm-locked-python', VIRTUAL_ENV: 'runtime-venv', PYTHONPATH: 'payload-libraries', PYTHONHOME: 'payload-python' }
  expect(generateIcons(['--check'], { root, run, env })).toBe(0)
  expect(run).toHaveBeenCalledExactlyOnceWith('uv', [
    'run', '--isolated', '--locked', '--only-group', 'icon-build',
    '--cache-dir', path.join(root, '.cache', 'icon-build'),
    'python', path.join(root, 'scripts', 'generate_icons.py'), '--source', root, '--out', root, '--check'
  ], { cwd: root, stdio: 'inherit', windowsHide: true, env: { PATH: 'tools', UV_PYTHON: env.UV_PYTHON, VIRTUAL_ENV: 'runtime-venv' } })
  expect(env.PYTHONPATH).toBe('payload-libraries')
})

test('icon preparation passes explicit source and independent output roots', () => {
  const run = vi.fn(() => ({ status: 0 }))
  const source = path.resolve('source with spaces')
  const out = path.resolve('icon outputs')
  expect(generateIcons(['--source', source, '--out', out], { run })).toBe(0)
  const [, args, options] = run.mock.calls[0]
  expect(args.slice(-4)).toEqual(['--source', source, '--out', out])
  expect(options.cwd).toBe(source)
})

test('failed icon processes cannot report a successful build', () => {
  expect(generateIcons([], { run: () => ({ status: 7 }), env: {} })).toBe(7)
  expect(generateIcons([], { run: () => ({ status: null, signal: 'SIGTERM' }), env: {} })).toBe(1)
  const error = vi.spyOn(console, 'error').mockImplementation(() => {})
  try {
    expect(generateIcons([], { run: () => ({ error: new Error('uv missing') }), env: {} })).toBe(1)
    expect(error).toHaveBeenCalledWith(expect.stringContaining('failed to launch'), 'uv missing')
  } finally {
    error.mockRestore()
  }
})
