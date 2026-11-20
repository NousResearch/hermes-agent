const fs = require('node:fs')

// Observe at launch: a renderer can close before the native process and its
// stdio pipes. Playwright's driver exit cleanup tree-kills until that close.
function observeProcessClose(child) {
  let closed = false
  const completion = new Promise(resolve => child.once('close', () => {
    closed = true
    resolve()
  }))
  // Windows descendants can inherit pipe handles and postpone 'close' after
  // the launch process exits. Release our handles, never kill descendants.
  const releasePipes = () => {
    for (const stream of child.stdio) stream?.destroy()
  }
  child.once('exit', releasePipes)
  if (child.exitCode !== null || child.signalCode !== null) releasePipes()
  return async function waitForClose(timeoutMs = 120_000) {
    if (closed) return
    let timer
    try {
      await Promise.race([
        completion,
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('Electron process did not close after update hand-off')), timeoutMs)
        }),
      ])
    } finally {
      clearTimeout(timer)
    }
  }
}

// The main process reports updater failures on its own stdout/stderr (the
// Windows App Installer checker's stderr goes nowhere else). Synchronous
// appends survive the driver's process.exit right after the app closes.
function recordProcessOutput(child, file) {
  for (const name of ['stdout', 'stderr']) {
    child[name]?.on('data', chunk => fs.appendFileSync(file, `[${name}] ${chunk}`))
  }
}

module.exports = { observeProcessClose, recordProcessOutput }
