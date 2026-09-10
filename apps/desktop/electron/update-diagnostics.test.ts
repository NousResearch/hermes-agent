import assert from 'node:assert/strict'

import { test } from 'vitest'

import { classifyUpdateCheckFailure, describeUpdateCheckFailure, proxyHint } from './update-diagnostics'

// Real stderr captured on Windows from the desktop check's own git invocation
// (`git -c windows.appendAtomically=false fetch --quiet origin main`).
const LOCAL_PATH_REMOTE_STDERR = [
  "fatal: 'D:/definitely-not-a-repo-105855.git' does not appear to be a git repository",
  'fatal: Could not read from remote repository.',
  '',
  'Please make sure you have the correct access rights'
].join('\n')

const DNS_STDERR =
  "fatal: unable to access 'https://nonexistent-host.invalid/foo.git/': Could not resolve host: nonexistent-host.invalid"

const LOCK_STDERR = "fatal: Unable to create 'C:/x/.git/shallow.lock': File exists."

const TLS_STDERR =
  "fatal: unable to access 'https://github.com/NousResearch/hermes-agent.git/': SSL certificate problem: unable to get local issuer certificate"

const IMPLICIT_PROMPT_STDERR = "fatal: could not read Username for 'https://github.com': terminal prompts disabled"

test('a non-network failure is never described as an unreachable server', () => {
  const message = describeUpdateCheckFailure({
    remote: 'D:/definitely-not-a-repo-105855.git',
    branch: 'main',
    code: 128,
    stderr: LOCAL_PATH_REMOTE_STDERR,
    env: {}
  })

  assert.doesNotMatch(message, /unreachable|reach the update server/i)
  assert.match(message, /not a git repository/i)
})

test('a locked git metadata failure is classified as a lock, not as the network', () => {
  const cause = classifyUpdateCheckFailure(LOCK_STDERR)

  assert.match(cause, /lock/i)
  assert.doesNotMatch(cause, /network|unreachable/i)
})

test('a TLS certificate failure is classified as a certificate problem', () => {
  const cause = classifyUpdateCheckFailure(TLS_STDERR)

  assert.match(cause, /certificate/i)
  assert.doesNotMatch(cause, /unreachable/i)
})

test('a disabled credential prompt is classified as credentials, not as the network', () => {
  const cause = classifyUpdateCheckFailure(IMPLICIT_PROMPT_STDERR)

  assert.match(cause, /credential|login|auth/i)
  assert.doesNotMatch(cause, /unreachable/i)
})

test('a genuine name-resolution failure is classified as a network failure', () => {
  const cause = classifyUpdateCheckFailure(DNS_STDERR)

  assert.match(cause, /DNS|network|connection/i)
})

test('the diagnostic names the URL, the exit code and the proxy state', () => {
  const message = describeUpdateCheckFailure({
    remote: 'https://github.com/NousResearch/hermes-agent.git',
    branch: 'main',
    code: 128,
    stderr: DNS_STDERR,
    env: { HTTPS_PROXY: 'http://user:secret@corp-proxy:8080' }
  })

  assert.match(message, /https:\/\/github\.com\/NousResearch\/hermes-agent\.git/)
  assert.match(message, /exit 128/)
  assert.match(message, /proxy/i)
  // Which proxy variables are set, never their (credential-bearing) values.
  assert.match(message, /HTTPS_PROXY/)
  assert.doesNotMatch(message, /secret/)
})

test('the diagnostic survives a git failure that printed no stderr at all', () => {
  const message = describeUpdateCheckFailure({
    remote: 'origin',
    branch: 'main',
    code: 128,
    stderr: '',
    env: {}
  })

  assert.match(message, /exit 128/)
  assert.match(message, /origin/)
  assert.match(message, /no stderr/i)
})

test('proxyHint reports none set when the environment carries no proxy', () => {
  assert.match(proxyHint({}), /none set/i)
})

test('proxyHint reports the variable names, never their values', () => {
  const hint = proxyHint({ HTTPS_PROXY: 'http://user:secret@corp-proxy:8080', NO_PROXY: 'localhost' })

  assert.match(hint, /HTTPS_PROXY/)
  assert.match(hint, /NO_PROXY/)
  assert.doesNotMatch(hint, /secret|corp-proxy/)
})
