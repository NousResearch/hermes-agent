const path = require('node:path')

const STATUS_LABELS = {
  A: 'added',
  C: 'copied',
  D: 'deleted',
  M: 'modified',
  R: 'renamed',
  T: 'type-changed',
  U: 'conflicted'
}

function statusLabel(code) {
  return STATUS_LABELS[code] || ''
}

function parseBranchLine(line, branch) {
  if (line.startsWith('# branch.head ')) {
    const head = line.slice('# branch.head '.length)
    branch.name = head === '(detached)' ? '' : head
    branch.detached = head === '(detached)'
  } else if (line.startsWith('# branch.oid ')) {
    branch.oid = line.slice('# branch.oid '.length)
  } else if (line.startsWith('# branch.upstream ')) {
    branch.upstream = line.slice('# branch.upstream '.length)
  } else if (line.startsWith('# branch.ab ')) {
    const match = line.match(/^# branch\.ab \+(\d+) -(\d+)$/)

    if (match) {
      branch.ahead = Number.parseInt(match[1], 10)
      branch.behind = Number.parseInt(match[2], 10)
    }
  }
}

function entryFromRecord(record, root, originalPath) {
  const kind = record[0]
  const parts = record.split(' ')
  let xy = ''
  let relativePath = ''

  if (kind === '1') {
    xy = parts[1] || '..'
    relativePath = parts.slice(8).join(' ')
  } else if (kind === '2') {
    xy = parts[1] || '..'
    relativePath = parts.slice(9).join(' ')
  } else if (kind === 'u') {
    xy = parts[1] || 'UU'
    relativePath = parts.slice(10).join(' ')
  } else if (kind === '?') {
    xy = '??'
    relativePath = record.slice(2)
  } else {
    return null
  }

  const indexCode = xy[0] || '.'
  const worktreeCode = xy[1] || '.'
  const conflicted = kind === 'u' || indexCode === 'U' || worktreeCode === 'U'

  return {
    absolutePath: path.resolve(root, relativePath),
    conflicted,
    index: indexCode === '.' || indexCode === '?' ? '' : statusLabel(indexCode),
    indexCode: indexCode === '.' ? '' : indexCode,
    originalPath: originalPath || undefined,
    path: relativePath,
    untracked: kind === '?',
    worktree: worktreeCode === '.' || worktreeCode === '?' ? '' : statusLabel(worktreeCode),
    worktreeCode: worktreeCode === '.' ? '' : worktreeCode
  }
}

function parseGitStatusPorcelainV2(output, root) {
  const records = String(output || '').split('\0')
  const branch = { ahead: 0, behind: 0, detached: false, name: '', oid: '', upstream: '' }
  const entries = []

  for (let index = 0; index < records.length; index += 1) {
    const record = records[index]

    if (!record) {
      continue
    }

    if (record.startsWith('# ')) {
      parseBranchLine(record, branch)

      continue
    }

    const renamed = record[0] === '2'
    const originalPath = renamed ? records[index + 1] || '' : ''
    const entry = entryFromRecord(record, root, originalPath)

    if (renamed) {
      index += 1
    }

    if (entry) {
      entries.push(entry)
    }
  }

  return { branch, entries, root }
}

module.exports = { parseGitStatusPorcelainV2 }
