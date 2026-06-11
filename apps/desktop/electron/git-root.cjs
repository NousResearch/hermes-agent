'use strict'

const fs = require('node:fs')
const path = require('node:path')
const { resolveRequestedPathForIpc } = require('./hardening.cjs')

function findGitRoot(start, fsImpl = fs) {
  let dir = start

  for (let i = 0; i < 50; i += 1) {
    try {
      if (fsImpl.existsSync(path.join(dir, '.git'))) {
        return dir
      }
    } catch {
      return null
    }

    const parent = path.dirname(dir)

    if (parent === dir) {
      return null
    }

    dir = parent
  }

  return null
}

async function gitRootForIpc(startPath, options = {}) {
  const fsImpl = options.fs || fs
  let resolved

  try {
    resolved = resolveRequestedPathForIpc(startPath, { purpose: 'Git root' })
  } catch {
    return null
  }

  try {
    const stat = await fsImpl.promises.stat(resolved)
    const start = stat.isDirectory() ? resolved : path.dirname(resolved)

    return findGitRoot(start, fsImpl)
  } catch {
    return findGitRoot(resolved, fsImpl)
  }
}

module.exports = {
  findGitRoot,
  gitRootForIpc
}
