'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const ELECTRON_DIR = __dirname

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

// Regression guard for #50491: closing the window (Cmd+W) without quitting
// destroys the BrowserWindow but leaves `mainWindow` set. The 'second-instance'
// handler (re-focus on a second launch / dock click / hermes:// deep link) used
// to focus mainWindow after only a truthiness check, so it touched a destroyed
// native object and crashed the still-running process with an uncaught
// 'TypeError: Object has been destroyed'. The focus branch must additionally
// check !mainWindow.isDestroyed() before calling isMinimized()/restore()/focus().
test("second-instance focus branch guards against a destroyed mainWindow", () => {
  const source = readElectronFile('main.cjs')

  const handlerIndex = source.indexOf("app.on('second-instance'")
  assert.notEqual(handlerIndex, -1, "missing 'second-instance' handler")

  const handler = source.slice(handlerIndex, handlerIndex + 400)

  // The branch that focuses an existing window must confirm the window is not
  // destroyed before touching it, matching the sibling 'activate' handler.
  assert.match(
    handler,
    /else if \(mainWindow && !mainWindow\.isDestroyed\(\)\) \{/,
    "second-instance focus branch must guard with !mainWindow.isDestroyed()"
  )

  // A bare truthiness-only guard immediately followed by isMinimized() is the
  // exact crash shape from #50491 and must not regress.
  assert.doesNotMatch(
    handler,
    /else if \(mainWindow\) \{\s*if \(mainWindow\.isMinimized\(\)\)/,
    "second-instance focus branch must not call isMinimized() behind an unguarded mainWindow check"
  )
})
