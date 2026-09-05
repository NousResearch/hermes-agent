import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  COMMIT_LOG_LIMIT,
  parseCommitLog,
  readCommitLogRange,
  readOfficialSshCommitLog
} from './update-commit-log'

const SEP = '\x1f'
const REC = '\x1e'

const record = (sha, summary, author, at) => `${sha}${SEP}${summary}${SEP}${author}${SEP}${at}${REC}`

test('parseCommitLog parses records and converts the unix timestamp to millis', () => {
  const stdout = record('abc123', 'fix: thing', 'Ada', '1700000000') + record('def456', 'feat: other', 'Grace', '1700000060')

  assert.deepEqual(parseCommitLog(stdout), [
    { sha: 'abc123', summary: 'fix: thing', author: 'Ada', at: 1700000000000 },
    { sha: 'def456', summary: 'feat: other', author: 'Grace', at: 1700000060000 }
  ])
})

test('parseCommitLog returns [] for empty output', () => {
  assert.deepEqual(parseCommitLog(''), [])
})

test('parseCommitLog tolerates a summary containing spaces (unit-separated fields)', () => {
  const stdout = record('sha', 'fix: keep spaces and punctuation, intact', 'A B', '1700000000')

  assert.deepEqual(parseCommitLog(stdout), [
    { sha: 'sha', summary: 'fix: keep spaces and punctuation, intact', author: 'A B', at: 1700000000000 }
  ])
})

// FAIL-BEFORE: the official-SSH branch in checkUpdates() hardcoded `commits: []`,
// so an SSH-origin install always showed the "Improvements and fixes" fallback
// (#69081). This asserts the new path fetches over the anonymous HTTPS URL — never
// `git fetch origin` (which would trigger a passkey touch) — and returns the log.
test('readOfficialSshCommitLog fetches over HTTPS (not origin) then reads HEAD..<sha>', async () => {
  const calls: string[][] = []

  const run = async (args: string[]) => {
    calls.push(args)

    if (args[0] === 'fetch') {
      return { code: 0, stdout: '', stderr: '' }
    }

    return { code: 0, stdout: record('sha1', 'fix: a', 'Ada', '1700000000'), stderr: '' }
  }

  const commits = await readOfficialSshCommitLog(
    run,
    'https://github.com/NousResearch/hermes-agent.git',
    'main',
    'targetsha'
  )

  // First git call is a fetch against the HTTPS URL, not the `origin` SSH remote.
  assert.deepEqual(calls[0], ['fetch', '--quiet', 'https://github.com/NousResearch/hermes-agent.git', 'main'])
  assert.equal(calls.some(c => c.includes('origin')), false)
  // Second git call reads the log up to the ls-remote target SHA.
  assert.deepEqual(calls[1], ['log', 'HEAD..targetsha', `--pretty=format:%H${SEP}%s${SEP}%an${SEP}%at${REC}`, '-n', String(COMMIT_LOG_LIMIT)])
  assert.deepEqual(commits, [{ sha: 'sha1', summary: 'fix: a', author: 'Ada', at: 1700000000000 }])
})

test('readOfficialSshCommitLog returns [] (and skips the log) when the HTTPS fetch fails', async () => {
  const calls: string[][] = []

  const run = async (args: string[]) => {
    calls.push(args)

    return { code: 1, stdout: '', stderr: 'network down' }
  }

  const commits = await readOfficialSshCommitLog(run, 'https://example/repo.git', 'main', 'targetsha')

  assert.deepEqual(commits, [])
  assert.equal(calls.length, 1) // fetch attempted; no `git log` after a failed fetch
  assert.equal(calls[0][0], 'fetch')
})

test('readCommitLogRange issues the log for the given range and caps at COMMIT_LOG_LIMIT', async () => {
  const calls: string[][] = []

  const run = async (args: string[]) => {
    calls.push(args)

    return { code: 0, stdout: record('s', 'msg', 'Author', '1700000000'), stderr: '' }
  }

  const commits = await readCommitLogRange(run, 'HEAD..origin/main')

  assert.deepEqual(calls[0], ['log', 'HEAD..origin/main', `--pretty=format:%H${SEP}%s${SEP}%an${SEP}%at${REC}`, '-n', String(COMMIT_LOG_LIMIT)])
  assert.deepEqual(commits, [{ sha: 's', summary: 'msg', author: 'Author', at: 1700000000000 }])
})
