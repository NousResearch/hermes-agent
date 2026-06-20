'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const source = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8').replace(/\r\n/g, '\n')

test('workflow copilot webview partition is explicitly allowed by the attach guard', () => {
  assert.match(source, /const ALLOWED_WEBVIEW_PARTITIONS = new Set\(\[/)
  assert.match(source, /'persist:hermes-workflow-chat'/)
  assert.match(source, /function isAllowedWebviewAttach\(params\)/)
  assert.match(source, /if \(!ALLOWED_WEBVIEW_PARTITIONS\.has\(partition\)\) \{\n\s+return false\n\s+\}/)
})
