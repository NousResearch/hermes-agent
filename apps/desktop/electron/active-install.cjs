const fs = require('node:fs')
const path = require('node:path')

function fileExists(target) {
  try {
    return fs.statSync(target).isFile()
  } catch {
    return false
  }
}

function directoryExists(target) {
  try {
    return fs.statSync(target).isDirectory()
  } catch {
    return false
  }
}

function getVenvPython(venvRoot, platform = process.platform) {
  return platform === 'win32'
    ? path.join(venvRoot, 'Scripts', 'python.exe')
    : path.join(venvRoot, 'bin', 'python')
}

function isHermesSourceRoot(root) {
  return directoryExists(root) && fileExists(path.join(root, 'hermes_cli', 'main.py'))
}

function hasUsableActiveInstall(activeRoot, venvRoot, options = {}) {
  const platform = options.platform || process.platform
  return isHermesSourceRoot(activeRoot) && fileExists(getVenvPython(venvRoot, platform))
}

module.exports = {
  fileExists,
  directoryExists,
  getVenvPython,
  isHermesSourceRoot,
  hasUsableActiveInstall
}
