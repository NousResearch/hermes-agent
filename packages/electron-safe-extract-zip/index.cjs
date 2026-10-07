'use strict'

const fs = require('node:fs')
const path = require('node:path')
const { spawn } = require('node:child_process')

const STDERR_LIMIT = 16 * 1024
const PYTHON_EXTRACTOR = path.join(__dirname, 'extract.py')

function resolveExtractorCommand(platform = process.platform, env = process.env, exists = fs.existsSync) {
  if (platform === 'win32') {
    const systemRoot = env.SystemRoot || env.SYSTEMROOT
    if (systemRoot && path.win32.isAbsolute(systemRoot)) {
      const command = path.win32.join(systemRoot, 'System32', 'tar.exe')
      if (exists(command)) {
        return { command, args: (archive, dir) => ['-xf', archive, '-C', dir] }
      }
    }
  } else if (platform === 'darwin') {
    const command = '/usr/bin/tar'
    if (exists(command)) {
      return { command, args: (archive, dir) => ['-xf', archive, '-C', dir] }
    }
  } else {
    for (const command of ['/usr/bin/unzip', '/bin/unzip']) {
      if (exists(command)) {
        return { command, args: (archive, dir) => ['-q', archive, '-d', dir] }
      }
    }
    for (const command of ['/usr/bin/python3', '/usr/local/bin/python3']) {
      if (exists(command)) {
        return {
          command,
          args: (archive, dir) => [PYTHON_EXTRACTOR, archive, dir]
        }
      }
    }
  }

  throw new Error(`trusted ZIP extractor is unavailable for ${platform}`)
}

async function extract(archive, options) {
  const dir = options && options.dir
  if (!path.isAbsolute(archive)) {
    throw new TypeError('archive path must be absolute')
  }
  if (typeof dir !== 'string' || !path.isAbsolute(dir)) {
    throw new TypeError('output directory must be absolute')
  }

  await fs.promises.mkdir(dir, { recursive: true })
  const extractor = resolveExtractorCommand()
  const args = extractor.args(archive, dir)

  await new Promise((resolve, reject) => {
    const child = spawn(extractor.command, args, {
      shell: false,
      windowsHide: true,
      stdio: ['ignore', 'ignore', 'pipe']
    })
    let stderr = ''
    child.stderr.setEncoding('utf8')
    child.stderr.on('data', (chunk) => {
      if (stderr.length < STDERR_LIMIT) stderr += chunk.slice(0, STDERR_LIMIT - stderr.length)
    })
    child.once('error', reject)
    child.once('close', (code, signal) => {
      if (code === 0) {
        resolve()
        return
      }
      const reason = signal ? `signal ${signal}` : `exit code ${code}`
      reject(new Error(`Electron ZIP extraction failed (${reason})${stderr ? `: ${stderr.trim()}` : ''}`))
    })
  })
}

module.exports = { extract, resolveExtractorCommand }
