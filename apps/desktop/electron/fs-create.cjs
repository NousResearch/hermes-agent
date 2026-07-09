'use strict'

const fs = require('node:fs')
const path = require('node:path')

const { resolveDirectoryForIpc, resolveRequestedPathForIpc } = require('./hardening.cjs')

const CREATE_TEXT_MAX_CHARS = 1_000_000

function validateCreatePath(rawPath, purpose) {
  const raw = String(rawPath || '').trim()

  if (!raw) {
    throw new Error(`${purpose} failed: path is required.`)
  }

  const name = path.basename(raw)

  if (!name || name === '.' || name === '..') {
    throw new Error(`${purpose} failed: invalid name.`)
  }

  return resolveRequestedPathForIpc(raw, { purpose })
}

async function assertParentDirectory(resolvedPath, fsImpl, purpose) {
  await resolveDirectoryForIpc(path.dirname(resolvedPath), { fs: fsImpl, purpose: `${purpose} parent` })
}

async function createTextFileForIpc(filePath, content = '', options = {}) {
  const fsImpl = options.fs || fs
  const purpose = 'Create file'
  const resolved = validateCreatePath(filePath, purpose)
  const text = String(content ?? '')

  if (text.length > CREATE_TEXT_MAX_CHARS) {
    throw new Error('Create file failed: content too large.')
  }

  await assertParentDirectory(resolved, fsImpl, purpose)
  await fsImpl.promises.writeFile(resolved, text, { encoding: 'utf8', flag: 'wx' })

  return { path: resolved }
}

async function createFolderForIpc(folderPath, options = {}) {
  const fsImpl = options.fs || fs
  const purpose = 'Create folder'
  const resolved = validateCreatePath(folderPath, purpose)

  await assertParentDirectory(resolved, fsImpl, purpose)
  await fsImpl.promises.mkdir(resolved)

  return { path: resolved }
}

module.exports = {
  CREATE_TEXT_MAX_CHARS,
  createFolderForIpc,
  createTextFileForIpc
}
