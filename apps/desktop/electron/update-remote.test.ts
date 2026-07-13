/**
 * Tests for electron/update-remote.ts — the remote-detection helpers that
 * keep passive update checks off the SSH origin for official installs.
 *
 * Run with: node --test electron/update-remote.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: a public install can carry
 * origin=git@github.com:NousResearch/hermes-agent.git. A background
 * `git fetch origin` then authenticates over SSH and, with a FIDO2/passkey
 * key, triggers an unexplained hardware-touch prompt. isOfficialSshRemote
 * must reliably recognize the official SSH remote (in every URL form,
 * case-insensitively) so the caller can swap in the anonymous HTTPS path —
 * while NOT misclassifying forks, other hosts, or the HTTPS remote (which
 * never prompts and should keep the normal fetch path).
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import {
  canonicalGitHubRemote,
  isOfficialRemote,
  isOfficialSshRemote,
  isSshRemote,
  OFFICIAL_REPO_CANONICAL,
  OFFICIAL_REPO_HTTPS_URL,
  selectUpdateRemote
} from './update-remote'

test('canonicalGitHubRemote normalizes SSH and HTTPS forms to the same value', () => {
  assert.equal(canonicalGitHubRemote('git@github.com:NousResearch/hermes-agent.git'), OFFICIAL_REPO_CANONICAL)
  assert.equal(canonicalGitHubRemote('git@github.com:NousResearch/hermes-agent'), OFFICIAL_REPO_CANONICAL)
  assert.equal(canonicalGitHubRemote('ssh://git@github.com/NousResearch/hermes-agent.git'), OFFICIAL_REPO_CANONICAL)
  assert.equal(canonicalGitHubRemote('https://github.com/NousResearch/hermes-agent.git'), OFFICIAL_REPO_CANONICAL)
  // Case-insensitive: an uppercased owner still canonicalizes to the same repo.
  assert.equal(canonicalGitHubRemote('git@github.com:nousresearch/hermes-agent.git'), OFFICIAL_REPO_CANONICAL)
  // Trailing slashes are stripped.
  assert.equal(canonicalGitHubRemote('https://github.com/NousResearch/hermes-agent/'), OFFICIAL_REPO_CANONICAL)
})

test('canonicalGitHubRemote is empty for falsy input', () => {
  assert.equal(canonicalGitHubRemote(''), '')
  assert.equal(canonicalGitHubRemote(null), '')
  assert.equal(canonicalGitHubRemote(undefined), '')
})

test('isSshRemote detects scp-like and ssh:// forms only', () => {
  assert.equal(isSshRemote('git@github.com:NousResearch/hermes-agent.git'), true)
  assert.equal(isSshRemote('ssh://git@github.com/NousResearch/hermes-agent.git'), true)
  assert.equal(isSshRemote('https://github.com/NousResearch/hermes-agent.git'), false)
  assert.equal(isSshRemote(''), false)
  assert.equal(isSshRemote(null), false)
})

test('isOfficialSshRemote is true only for the official repo over SSH', () => {
  assert.equal(isOfficialSshRemote('git@github.com:NousResearch/hermes-agent.git'), true)
  assert.equal(isOfficialSshRemote('git@github.com:NousResearch/hermes-agent'), true)
  assert.equal(isOfficialSshRemote('ssh://git@github.com/NousResearch/hermes-agent.git'), true)
  // Case-insensitive owner/repo match.
  assert.equal(isOfficialSshRemote('git@github.com:nousresearch/hermes-agent.git'), true)
})

test('isOfficialSshRemote does NOT match forks, other hosts, or HTTPS', () => {
  // A fork over SSH belongs to the user — fetching it is their own remote,
  // not the official upstream, so the SSH-avoidance swap must not apply.
  assert.equal(isOfficialSshRemote('git@github.com:someuser/hermes-agent.git'), false)
  // Same repo name on a different host is not the official repo.
  assert.equal(isOfficialSshRemote('git@gitlab.com:NousResearch/hermes-agent.git'), false)
  // HTTPS to the official repo never prompts for SSH/FIDO2, so it keeps the
  // normal fetch path — must not be flagged as an official SSH remote.
  assert.equal(isOfficialSshRemote('https://github.com/NousResearch/hermes-agent.git'), false)
  assert.equal(isOfficialSshRemote(''), false)
  assert.equal(isOfficialSshRemote(null), false)
})

test('OFFICIAL_REPO_HTTPS_URL canonicalizes to OFFICIAL_REPO_CANONICAL', () => {
  // Invariant: the URL we substitute in must be the same repo we detect.
  assert.equal(canonicalGitHubRemote(OFFICIAL_REPO_HTTPS_URL), OFFICIAL_REPO_CANONICAL)
})


test('isOfficialRemote matches official HTTPS and SSH forms but not forks', () => {
  assert.equal(isOfficialRemote('https://github.com/NousResearch/hermes-agent.git'), true)
  assert.equal(isOfficialRemote('git@github.com:NousResearch/hermes-agent.git'), true)
  assert.equal(isOfficialRemote('https://github.com/matantsevs/hermes-agent.git'), false)
})

test('selectUpdateRemote prefers official upstream when origin is a fork', () => {
  assert.deepEqual(
    selectUpdateRemote({
      originUrl: 'https://github.com/matantsevs/hermes-agent.git',
      upstreamUrl: 'https://github.com/NousResearch/hermes-agent.git'
    }),
    { kind: 'upstream', remote: 'upstream', ref: null }
  )
})

test('selectUpdateRemote keeps official origin and falls back to origin without upstream', () => {
  assert.deepEqual(
    selectUpdateRemote({
      originUrl: 'https://github.com/NousResearch/hermes-agent.git',
      upstreamUrl: ''
    }),
    { kind: 'origin', remote: 'origin', ref: null }
  )
  assert.deepEqual(
    selectUpdateRemote({
      originUrl: 'https://github.com/matantsevs/hermes-agent.git',
      upstreamUrl: ''
    }),
    { kind: 'origin', remote: 'origin', ref: null }
  )
})

test('selectUpdateRemote uses anonymous HTTPS for official SSH remotes', () => {
  assert.deepEqual(
    selectUpdateRemote({
      originUrl: 'git@github.com:matantsevs/hermes-agent.git',
      upstreamUrl: 'git@github.com:NousResearch/hermes-agent.git'
    }),
    { kind: 'official-ssh', remote: OFFICIAL_REPO_HTTPS_URL, ref: 'FETCH_HEAD' }
  )
})
