import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isPenSchemaAction, penToolNames, unknownPenToolError } from './mcp'

test('schema fetches the live list; get_app_state is an editor tool', () => {
  assert.equal(isPenSchemaAction('schema'), true)
  assert.equal(isPenSchemaAction('get-mcp-schema'), true)
  assert.equal(isPenSchemaAction('get_app_state'), false)
  assert.equal(isPenSchemaAction('execute'), false)
})

const schema = { tools: [{ name: 'execute' }, { name: 'read_skill' }, { name: '' }, null, { nope: 1 }] }

test('penToolNames reads the editor list and nothing else', () => {
  assert.deepEqual(penToolNames(schema), ['execute', 'read_skill'])
  assert.deepEqual(penToolNames({ result: [] }), [])
  assert.deepEqual(penToolNames(null), [])
})

test('a made-up tool is refused with the real list and the way in; a real one passes', () => {
  const tools = penToolNames(schema)
  const message = unknownPenToolError('create_frame', tools)

  assert.ok(message)
  assert.match(message, /'create_frame' is not an editor tool/)
  assert.match(message, /execute, read_skill/)
  assert.match(message, /execute\(\{ input:/)
  assert.equal(unknownPenToolError('execute', tools), null)
  // No list yet: nothing to check against, let the editor answer.
  assert.equal(unknownPenToolError('create_frame', []), null)
})
