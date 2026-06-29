const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')

function readMain() {
  return fs.readFileSync(path.join(__dirname, 'main.cjs'), 'utf8').replace(/\r\n/g, '\n')
}

test('second-instance launches show an existing hidden main window', () => {
  const source = readMain()
  const lockHandler = source.match(/app\.on\('second-instance',[\s\S]*?\n\s+\}\)\n\}/)
  assert.ok(lockHandler, 'missing second-instance handler')
  assert.match(
    lockHandler[0],
    /else if \(mainWindow\) \{\n\s+focusWindow\(mainWindow\)\n\s+\}/,
    'second-instance handler must use focusWindow(mainWindow) so hidden windows are shown before focus'
  )
  assert.match(
    source,
    /function focusWindow\(win\) \{[\s\S]*?if \(!win\.isVisible\(\)\) win\.show\(\)[\s\S]*?win\.focus\(\)/,
    'focusWindow must show hidden windows before focusing them'
  )
})
