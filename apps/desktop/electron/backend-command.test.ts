import assert from 'node:assert/strict'

import { test } from 'vitest'

import { dashboardFallbackArgs, serveBackendArgs, sourceDeclaresServe } from './backend-command'

test('serveBackendArgs pins a profile when provided', () => {
  assert.deepEqual(serveBackendArgs('worker'), ['--profile', 'worker', 'serve', '--host', '127.0.0.1', '--port', '0'])
})

test('dashboardFallbackArgs preserves a --profile flag ahead of serve', () => {
  const serve = ['-m', 'hermes_cli.main', '--profile', 'worker', 'serve', '--host', '127.0.0.1', '--port', '0']
  assert.deepEqual(dashboardFallbackArgs(serve), [
    '-m',
    'hermes_cli.main',
    '--profile',
    'worker',
    'dashboard',
    '--no-open',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('dashboardFallbackArgs skips a profile named "serve" and rewrites the subcommand', () => {
  // The profile VALUE 'serve' is not the subcommand: rewriting at index 1
  // yields `--profile dashboard --no-open serve`, which still runs `serve`.
  const serve = ['-m', 'hermes_cli.main', '--profile', 'serve', 'serve', '--host', '127.0.0.1', '--port', '0']
  assert.deepEqual(dashboardFallbackArgs(serve), [
    '-m',
    'hermes_cli.main',
    '--profile',
    'serve',
    'dashboard',
    '--no-open',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('dashboardFallbackArgs skips a -p profile value named "serve"', () => {
  const serve = ['-p', 'serve', 'serve', '--host', '127.0.0.1', '--port', '0']
  assert.deepEqual(dashboardFallbackArgs(serve), [
    '-p',
    'serve',
    'dashboard',
    '--no-open',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('dashboardFallbackArgs rewrites through a --profile=serve self-contained flag', () => {
  const serve = ['--profile=serve', 'serve', '--host', '127.0.0.1', '--port', '0']
  assert.deepEqual(dashboardFallbackArgs(serve), [
    '--profile=serve',
    'dashboard',
    '--no-open',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('dashboardFallbackArgs leaves a serve-looking profile value alone when there is no subcommand', () => {
  const args = ['--profile', 'serve']
  assert.deepEqual(dashboardFallbackArgs(args), args)
})

test('dashboardFallbackArgs is a no-op (copy) when there is no serve token', () => {
  const args = ['-m', 'hermes_cli.main', 'dashboard', '--no-open']
  const out = dashboardFallbackArgs(args)
  assert.deepEqual(out, args)
})

test('sourceDeclaresServe detects the serve subparser registration', () => {
  assert.equal(sourceDeclaresServe('subparsers.add_parser("serve", help="...")'), true)
  assert.equal(sourceDeclaresServe("subparsers.add_parser('serve')"), true)
  assert.equal(sourceDeclaresServe('subparsers.add_parser(\n        "serve",\n)'), true)
})

test('sourceDeclaresServe does not false-positive on the substring "server"', () => {
  const oldSource = `
    dashboard_parser = subparsers.add_parser("dashboard", help="Start the web UI dashboard")
    from hermes_cli.web_server import start_server  # web server
  `

  assert.equal(sourceDeclaresServe(oldSource), false)
})

test('serveBackendArgs drops a profile value that is not a valid profile id', () => {
  // The roster/SSH bridge can hand us unvalidated values (a numeric id, an empty
  // string, a display label): a non-slug must never reach the backend spawn argv,
  // where it would bootstrap a phantom profile directory (#88842).
  const base = ['serve', '--host', '127.0.0.1', '--port', '0']
  assert.deepEqual(serveBackendArgs(0 as unknown as string), base)
  assert.deepEqual(serveBackendArgs(''), base)
  assert.deepEqual(serveBackendArgs('   '), base)
  assert.deepEqual(serveBackendArgs('Not A Slug!'), base)
})

test('serveBackendArgs keeps a valid profile id pinned, normalized like the CLI', () => {
  assert.deepEqual(serveBackendArgs('worker'), ['--profile', 'worker', 'serve', '--host', '127.0.0.1', '--port', '0'])
  assert.deepEqual(serveBackendArgs('a-1_b'), ['--profile', 'a-1_b', 'serve', '--host', '127.0.0.1', '--port', '0'])
  assert.deepEqual(serveBackendArgs('  Worker  '), ['--profile', 'worker', 'serve', '--host', '127.0.0.1', '--port', '0'])
  assert.deepEqual(serveBackendArgs('default'), ['--profile', 'default', 'serve', '--host', '127.0.0.1', '--port', '0'])
})
