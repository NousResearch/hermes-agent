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

test('macOS BrowserWindow icon is left to the bundle icon', () => {
  assert.match(mainSource, /const icon = IS_MAC \? undefined : getAppIconPath\(\)/)
})
