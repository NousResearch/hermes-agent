import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isPenSchemaAction } from './mcp'

test('schema fetches the live list; get_app_state is an editor tool', () => {
  assert.equal(isPenSchemaAction('schema'), true)
  assert.equal(isPenSchemaAction('get-mcp-schema'), true)
  assert.equal(isPenSchemaAction('get_app_state'), false)
  assert.equal(isPenSchemaAction('execute'), false)
})
