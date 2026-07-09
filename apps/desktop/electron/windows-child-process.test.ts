import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import test from 'node:test'
import { fileURLToPath } from 'node:url'

const ELECTRON_DIR = path.dirname(fileURLToPath(import.meta.url))

// TODO FIXME these tests all grep source code for specific things. This is an antipattern.
// Tests should NEVER read src, only assert behavior.

function readElectronFile(name) {
  return fs.readFileSync(path.join(ELECTRON_DIR, name), 'utf8').replace(/\r\n/g, '\n')
}

function requireHiddenChildOptions(source, needle) {
  const match = needle instanceof RegExp ? needle.exec(source) : null
  const index = needle instanceof RegExp ? (match?.index ?? -1) : source.indexOf(needle)
  assert.notEqual(index, -1, `missing call site: ${needle}`)
  const snippet = source.slice(index, index + 700)
  assert.match(
    snippet,
    /hiddenWindowsChildOptions\(/,
    `expected ${needle} to wrap child-process options with hiddenWindowsChildOptions`
  )
}

test('desktop background child processes opt into hidden Windows consoles', () => {
  const source = readElectronFile('main.ts')

  assert.match(source, /function hiddenWindowsChildOptions\(options: any = \{\}\)/)

  requireHiddenChildOptions(source, "execFileSync(\n          'reg'")
  requireHiddenChildOptions(source, /execFileSync\(\s*pyExe/)
  requireHiddenChildOptions(source, /spawn\(\s*resolveGitBinary\(\)/)
  requireHiddenChildOptions(source, "execFileSync('taskkill'")
  requireHiddenChildOptions(source, /spawn\(\s*command,\s*args/)
  requireHiddenChildOptions(source, "spawn('curl'")
  requireHiddenChildOptions(source, /spawn\(\s*backend\.command,\s*backend\.args/)
  requireHiddenChildOptions(source, /hermesProcess = spawn\(\s*backend\.command,\s*backend\.args/)
  requireHiddenChildOptions(source, /spawn\(\s*py,\s*\['-m', 'hermes_cli\.main', 'uninstall', '--gui-summary'\]/)

  assert.match(source, /function unwrapWindowsVenvHermesCommand\(command, backendArgs\)/)
  assert.match(source, /function getVenvSitePackagesEntries\(venvRoot\)/)
  assert.match(source, /getWindowsVenvPythonPathEntries\(venvRoot/)
  assert.match(source, /args: \['-m', 'hermes_cli\.main', \.\.\.backendArgs\]/)
})

test('desktop backend launches console python so child consoles are inherited, not pythonw', () => {
  const source = readElectronFile('main.ts')

  // The flash fix is structural: the backend runs as a console-subsystem
  // python.exe under hiddenWindowsChildOptions() (-> CREATE_NO_WINDOW), so it
  // owns ONE windowless console that every descendant spawn inherits. Launching
  // it as GUI-subsystem pythonw.exe is what made each child allocate (and flash)
  // its own console, so the backend command must never be pythonw.
  assert.doesNotMatch(source, /pythonw\.exe'\)/, 'backend must not be launched via pythonw.exe')
  assert.doesNotMatch(
    source,
    /function getNoConsoleVenvPython\b/,
    'pythonw-conversion helper should be gone; console python is launched directly'
  )
  assert.doesNotMatch(
    source,
    /function applyWindowsNoConsoleSpawnHints\b/,
    'pythonw spawn-hint rewriter should be gone'
  )

  // Console python restores stdout, so the port is announced on the normal
  // HERMES_DASHBOARD_READY stdout line — no ready-file side channel is set.
  assert.doesNotMatch(source, /readyFile: true/, 'no backend should opt into the pythonw ready-file path')

  // Both desktop backend launches must still go through hiddenWindowsChildOptions
  // so the single backend console is created windowless.
  requireHiddenChildOptions(source, /spawn\(\s*backend\.command,\s*backend\.args/)
  requireHiddenChildOptions(source, /hermesProcess = spawn\(\s*backend\.command,\s*backend\.args/)
})

test('desktop backend teardown tree-kills Windows backend descendants', () => {
  const source = readElectronFile('main.ts')

  const helperIndex = source.indexOf('function stopBackendChild(child)')
  assert.notEqual(helperIndex, -1, 'missing backend teardown helper')
  const helperSnippet = source.slice(helperIndex, helperIndex + 500)
  assert.match(helperSnippet, /IS_WINDOWS && Number\.isInteger\(child\.pid\)/)
  assert.match(helperSnippet, /forceKillProcessTree\(child\.pid\)/)
  assert.match(helperSnippet, /child\.kill\('SIGTERM'\)/)

  const resetIndex = source.indexOf('function resetHermesConnection()')
  assert.notEqual(resetIndex, -1, 'missing resetHermesConnection')
  const resetSnippet = source.slice(resetIndex, resetIndex + 300)
  assert.match(resetSnippet, /stopBackendChild\(hermesProcess\)/)
  assert.doesNotMatch(resetSnippet, /hermesProcess\.kill\('SIGTERM'\)/)

  const quitIndex = source.indexOf("app.on('before-quit'")
  assert.notEqual(quitIndex, -1, 'missing before-quit handler')
  const quitSnippet = source.slice(quitIndex, quitIndex + 900)
  assert.match(quitSnippet, /stopBackendChild\(hermesProcess\)/)
  assert.doesNotMatch(quitSnippet, /hermesProcess\.kill\('SIGTERM'\)/)
})

test('intentional or interactive desktop child processes stay documented', () => {
  const source = readElectronFile('main.ts')

  assert.match(source, /windowsHide: false/)
  assert.match(source, /handOffWindowsBootstrapRecovery/)
  assert.match(source, /'--repair', '--branch'/)
  assert.match(source, /'--update', '--branch'/)
  assert.match(source, /nodePty\.spawn\(command, args/)
  assert.match(source, /spawn\('cmd\.exe', \['\/c', 'start'/)
})

test('bootstrap PowerShell runner hides Windows console children', () => {
  const source = readElectronFile('bootstrap-runner.ts')

  assert.match(source, /function hiddenWindowsChildOptions\(options = \{\}\)/)
  requireHiddenChildOptions(source, /spawn\(\s*ps,\s*fullArgs/)
})
