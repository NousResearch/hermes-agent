const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

const mainSource = fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8')

test('macOS Dock icon is not overridden at runtime', () => {
  assert.equal(
    mainSource.includes('app.dock?.setIcon'),
    false,
    'macOS should use the packaged bundle icon instead of a runtime Dock override'
  )
})

test('BrowserWindow still receives an icon for non-macOS platforms', () => {
  assert.match(mainSource, /const icon = getAppIconPath\(\)/)
  assert.match(mainSource, /new BrowserWindow\(\{[\s\S]*\n\s+icon,\n/)
})
