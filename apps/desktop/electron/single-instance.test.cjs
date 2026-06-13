const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

test('main process enforces a single app instance on Windows launches', () => {
  const mainSource = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')

  assert.match(mainSource, /requestSingleInstanceLock\(/)
  assert.match(mainSource, /second-instance/)
  assert.match(mainSource, /focusWindow\(mainWindow\)/)
})
