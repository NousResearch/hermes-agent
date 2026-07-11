'use strict'

// hc-517 — "New blank project" support for the desktop project picker. The
// renderer picks a PARENT directory (via the standard open dialog) and a NAME;
// this creates exactly `<parent>/<name>` and returns the absolute path to bind
// as the new session's cwd. Name is validated to a single, traversal-free path
// segment and an existing entry is never clobbered.

const fs = require('node:fs')
const path = require('node:path')
const { resolveRequestedPathForIpc } = require('./hardening.cjs')

// Traversal-safety guard: a project name may never contain a path separator,
// so it can only ever create a child of the chosen parent. Null bytes are
// already blocked upstream by resolveRequestedPathForIpc, and the OS rejects
// any other invalid name at mkdir with a clear error. Spaces, hyphens, dots and
// unicode in a name are all fine.
const PATH_SEPARATOR = /[/\\]/

function nameError(message) {
  const error = new Error(message)
  error.code = 'invalid-name'
  return error
}

/** Validate a project folder name — a single segment, no separators, no
 *  traversal. Returns the trimmed name. */
function validateProjectName(name) {
  const raw = typeof name === 'string' ? name.trim() : ''

  if (!raw) {
    throw nameError('Project name is required.')
  }

  if (raw === '.' || raw === '..') {
    throw nameError('Project name is invalid.')
  }

  if (PATH_SEPARATOR.test(raw)) {
    throw nameError('Project name cannot contain a path separator.')
  }

  return raw
}

function failure(error, fallbackCode) {
  return {
    ok: false,
    path: null,
    error: error && error.message ? error.message : 'Could not create the project folder.',
    code: (error && error.code) || fallbackCode
  }
}

/** Create `<parentDir>/<name>` and return its absolute path. Never overwrites
 *  an existing entry; the parent must already be a real directory. */
async function createProjectDirForIpc(parentDir, name, options = {}) {
  const fsImpl = options.fs || fs

  let validName
  try {
    validName = validateProjectName(name)
  } catch (error) {
    return failure(error, 'invalid-name')
  }

  let parent
  try {
    parent = resolveRequestedPathForIpc(parentDir, { purpose: 'Create project' })
  } catch (error) {
    return failure(error, 'invalid-path')
  }

  try {
    const stat = await fsImpl.promises.stat(parent)

    if (!stat.isDirectory()) {
      return failure(new Error('The selected parent path is not a folder.'), 'ENOTDIR')
    }
  } catch {
    return failure(new Error('The selected parent folder does not exist.'), 'ENOENT')
  }

  const target = path.join(parent, validName)

  // Refuse to reuse an existing file/folder — creating "blank" must never adopt
  // someone else's directory (or silently succeed onto a stale one).
  let exists = false
  try {
    await fsImpl.promises.stat(target)
    exists = true
  } catch {
    exists = false
  }

  if (exists) {
    return failure(new Error('A file or folder with that name already exists here.'), 'EEXIST')
  }

  try {
    await fsImpl.promises.mkdir(target, { recursive: false })
  } catch (error) {
    return failure(error, 'mkdir-error')
  }

  return { ok: true, path: target, error: null, code: null }
}

module.exports = {
  createProjectDirForIpc,
  validateProjectName
}
