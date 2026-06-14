'use strict'

const childProcess = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const BROKER_APP_NAME = 'HermesMacBroker.app'
const BROKER_EXECUTABLE_NAME = 'HermesMacBroker'

function brokerExecutableForAppBundle(appBundlePath) {
  if (!appBundlePath || typeof appBundlePath !== 'string') return null
  return path.join(
    appBundlePath,
    'Contents',
    'Library',
    'LoginItems',
    BROKER_APP_NAME,
    'Contents',
    'MacOS',
    BROKER_EXECUTABLE_NAME
  )
}

function brokerExecutableFromProcess({ platform = process.platform, resourcesPath = process.resourcesPath, execPath = process.execPath } = {}) {
  if (platform !== 'darwin') return null
  // Packaged Electron process executable normally lives at:
  // /Applications/Hermes.app/Contents/MacOS/Hermes
  if (execPath && execPath.includes('.app/Contents/MacOS/')) {
    const appBundle = execPath.slice(0, execPath.indexOf('.app/Contents/MacOS/') + '.app'.length)
    return brokerExecutableForAppBundle(appBundle)
  }
  // Fallback for tests or unusual packaged layouts where resourcesPath is:
  // /Applications/Hermes.app/Contents/Resources
  if (resourcesPath && resourcesPath.includes('.app/Contents/Resources')) {
    const appBundle = resourcesPath.slice(0, resourcesPath.indexOf('.app/Contents/Resources') + '.app'.length)
    return brokerExecutableForAppBundle(appBundle)
  }
  return null
}

function brokerAvailable(executable, fsImpl = fs) {
  return Boolean(executable && fsImpl.existsSync(executable))
}

function parseBrokerJson(stdout) {
  try {
    return JSON.parse(stdout)
  } catch (error) {
    return { ok: false, error: `invalid broker JSON: ${error.message}`, raw: String(stdout || '').slice(0, 500) }
  }
}

function runBrokerCommand(executable, args, options = {}) {
  if (!brokerAvailable(executable, options.fsImpl || fs)) {
    return { ok: false, error: 'macOS permission broker is not installed', executable }
  }
  try {
    const stdout = (options.execFileSync || childProcess.execFileSync)(executable, args, {
      encoding: 'utf8',
      timeout: options.timeoutMs || 5_000,
      stdio: ['ignore', 'pipe', 'pipe']
    })
    return parseBrokerJson(stdout)
  } catch (error) {
    return {
      ok: false,
      error: error.message,
      executable,
      stderr: String(error.stderr || '').slice(0, 500)
    }
  }
}

function brokerStatus(executable, options = {}) {
  return runBrokerCommand(executable, ['--status-json'], options)
}

function openBrokerSettings(executable, pane, options = {}) {
  return runBrokerCommand(executable, ['--open-settings', pane], options)
}

module.exports = {
  BROKER_APP_NAME,
  BROKER_EXECUTABLE_NAME,
  brokerAvailable,
  brokerExecutableForAppBundle,
  brokerExecutableFromProcess,
  brokerStatus,
  openBrokerSettings,
  parseBrokerJson,
  runBrokerCommand
}
