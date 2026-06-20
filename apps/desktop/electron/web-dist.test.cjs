const assert = require('node:assert/strict')
const path = require('node:path')
const test = require('node:test')

const { resolveWebDist } = require('./web-dist.cjs')

test('resolveWebDist prefers the built dashboard web_dist over the desktop renderer dist', () => {
  const root = '/repo/hermes-agent'
  const appRoot = path.join(root, 'apps/desktop')
  const existing = new Set([path.join(root, 'hermes_cli/web_dist'), path.join(appRoot, 'dist')])

  assert.equal(
    resolveWebDist({
      activeHermesRoot: '/Users/test/.hermes/hermes-agent',
      appRoot,
      directoryExists: value => existing.has(value),
      env: {},
      isPackaged: false,
      sourceRepoRoot: root,
      unpackedPathFor: value => path.join('/unpacked', value)
    }),
    path.join(root, 'hermes_cli/web_dist')
  )
})

test('resolveWebDist still honors an explicit HERMES_DESKTOP_WEB_DIST override', () => {
  const override = '/tmp/custom-web-dist'

  assert.equal(
    resolveWebDist({
      activeHermesRoot: '/Users/test/.hermes/hermes-agent',
      appRoot: '/repo/hermes-agent/apps/desktop',
      directoryExists: value => value === override,
      env: { HERMES_DESKTOP_WEB_DIST: override },
      isPackaged: false,
      sourceRepoRoot: '/repo/hermes-agent',
      unpackedPathFor: value => path.join('/unpacked', value)
    }),
    override
  )
})

