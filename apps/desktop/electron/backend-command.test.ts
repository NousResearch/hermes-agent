import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  dashboardFallbackArgs,
  helpDeclaresIsolated,
  serveBackendArgs,
  sourceDeclaresIsolated,
  sourceDeclaresServe,
  withoutIsolated
} from './backend-command'

const BS = String.fromCharCode(92)
const NEWLINE = String.fromCharCode(10)

test('serveBackendArgs builds a headless serve invocation', () => {
  assert.deepEqual(serveBackendArgs(), ['serve', '--host', '127.0.0.1', '--port', '0'])
})

test('serveBackendArgs pins a profile when provided', () => {
  assert.deepEqual(serveBackendArgs('worker'), [
    '--profile',
    'worker',
    'serve',
    '--isolated',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('dashboardFallbackArgs rewrites serve -> dashboard --no-open, keeping the -m prefix', () => {
  const serve = ['-m', 'hermes_cli.main', 'serve', '--host', '127.0.0.1', '--port', '0']
  assert.deepEqual(dashboardFallbackArgs(serve), [
    '-m',
    'hermes_cli.main',
    'dashboard',
    '--no-open',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ])
})

test('dashboardFallbackArgs preserves a --profile flag ahead of serve', () => {
  const serve = [
    '-m',
    'hermes_cli.main',
    '--profile',
    'worker',
    'serve',
    '--isolated',
    '--host',
    '127.0.0.1',
    '--port',
    '0'
  ]

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

test('dashboardFallbackArgs is a no-op (copy) when there is no serve token', () => {
  const args = ['-m', 'hermes_cli.main', 'dashboard', '--no-open']
  const out = dashboardFallbackArgs(args)
  assert.deepEqual(out, args)
  assert.notEqual(out, args, 'should return a copy, not the same reference')
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

test('dashboardFallbackArgs keeps a second --isolated the user supplied', () => {
  const serve = ['--profile', 'worker', 'serve', '--isolated', '--isolated', '--port', '0']

  assert.deepEqual(dashboardFallbackArgs(serve), [
    '--profile',
    'worker',
    'dashboard',
    '--no-open',
    '--isolated',
    '--port',
    '0'
  ])
})

test('withoutIsolated drops only the first occurrence and copies', () => {
  const args = ['serve', '--isolated', '--host', '127.0.0.1', '--isolated']
  const out = withoutIsolated(args)

  assert.deepEqual(out, ['serve', '--host', '127.0.0.1', '--isolated'])
  assert.notEqual(out, args, 'should return a copy, not the same reference')
})

test('withoutIsolated is a no-op (copy) when the flag is absent', () => {
  const args = ['serve', '--host', '127.0.0.1']
  const out = withoutIsolated(args)

  assert.deepEqual(out, args)
  assert.notEqual(out, args, 'should return a copy, not the same reference')
})

test('sourceDeclaresIsolated detects the --isolated flag registration', () => {
  assert.equal(sourceDeclaresIsolated('parser.add_argument(' + BS + 'n        "--isolated",' + BS + 'n)'), true)
  assert.equal(sourceDeclaresIsolated("parser.add_argument('--isolated', action='store_true')"), true)
})

test('sourceDeclaresIsolated is false for a serve-capable runtime that predates the flag', () => {
  const preIsolatedSource = `
    serve_parser = subparsers.add_parser("serve", help="Run the headless gateway")
    serve_parser.add_argument("--port", type=int, default=9119)
    serve_parser.add_argument("--host", default="127.0.0.1")
  `

  assert.equal(sourceDeclaresServe(preIsolatedSource), true)
  assert.equal(sourceDeclaresIsolated(preIsolatedSource), false)
})

test('helpDeclaresIsolated reads the flag out of a --help listing', () => {
  const help = [
    'usage: hermes serve [-h] [--port PORT] [--host HOST] [--isolated]',
    '',
    'options:',
    '  --isolated            Run a dedicated server scoped to that profile'
  ].join(NEWLINE)

  assert.equal(helpDeclaresIsolated(help), true)
})

test('helpDeclaresIsolated is false for help text without the flag', () => {
  const help = ['usage: hermes serve [-h] [--port PORT] [--host HOST]', '', 'options:', '  --port PORT'].join(
    NEWLINE
  )

  assert.equal(helpDeclaresIsolated(help), false)
  assert.equal(helpDeclaresIsolated(''), false)
  assert.equal(helpDeclaresIsolated(null), false)
})
