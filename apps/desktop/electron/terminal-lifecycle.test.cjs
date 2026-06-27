'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ROOT = path.resolve(__dirname, '..')

function readDesktopSource(...parts) {
  return fs.readFileSync(path.join(ROOT, 'src', ...parts), 'utf8').replace(/\r\n/g, '\n')
}

test('embedded terminal cwd changes do not restart the PTY session', () => {
  const source = readDesktopSource('app', 'right-sidebar', 'terminal', 'use-terminal-session.ts')

  assert.match(source, /const launchCwdRef = useRef\(cwd\)/)
  assert.match(source, /\.start\(\{ cols: term\.cols, cwd: launchCwdRef\.current, rows: term\.rows \}\)/)
  assert.doesNotMatch(source, /\}, \[addSelectionToChat, cwd\]\)/)
})
