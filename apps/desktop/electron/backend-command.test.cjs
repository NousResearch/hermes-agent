'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  dashboardFallbackArgs,
  localBackendDisabledError,
  serveBackendArgs,
  sourceDeclaresServe
} = require('./backend-command.cjs')

test('local backend command construction is disabled', () => {
  assert.throws(() => serveBackendArgs(), /remote frontend/)
  assert.throws(() => dashboardFallbackArgs(['serve']), /remote frontend/)
})

test('disabled helper exposes a clear error message', () => {
  const error = localBackendDisabledError('Test feature')
  assert.match(error.message, /Test feature is disabled/)
  assert.match(error.message, /configure a gateway URL/)
})

test('sourceDeclaresServe does not advertise local serve support', () => {
  assert.equal(sourceDeclaresServe('subparsers.add_parser("serve")'), false)
})
