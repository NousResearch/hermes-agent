import assert from 'node:assert/strict'

import { test } from 'vitest'

import { describeGitExitFailure } from './select-runnable-binary'

// The macOS Xcode-license trap: Xcode auto-updates to a new major, the
// /usr/bin/git shim refuses every invocation until the new licence is accepted,
// and it does so by EXITING 69 with the notice on stderr. The binary spawns
// fine, so describeGitSpawnFailure (spawn-level only) returns null and the
// caller falls through to the update-server/network wording — sending the user
// to debug their connection while git is what is broken.
test.each([
  {
    name: 'the Xcode licence notice is reported as a local git problem, not a network one',
    stderr:
      'You have not agreed to the Xcode license agreements. Please run ' +
      "'sudo xcodebuild -license' from within a Terminal window to review and agree to the Xcode and Apple SDKs license.",
    expected: /Git on this computer cannot run/
  },
  {
    // Same class, different trigger: the tools are missing//not selected rather
    // than unlicensed. Also a local executable problem reported via exit code.
    name: 'a missing/unselected developer directory is a local git problem',
    stderr: 'xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun',
    expected: /Git on this computer cannot run/
  }
])('$name', ({ stderr, expected }) => {
  const described = describeGitExitFailure(stderr)

  assert.match(described ?? '', expected)
  // The actionable remedy must survive into the message — the user cannot act
  // on "git failed" alone, and this is precisely the text the generic network
  // copy was hiding.
  assert.match(described ?? '', /xcodebuild -license|xcode-select/)
})

test('a genuine network or repository failure keeps its own wording', () => {
  // These are the failures the update-server copy is actually for; misrouting
  // them into the local-toolchain branch would be the same bug mirrored.
  for (const stderr of [
    'fatal: unable to access https://github.com/NousResearch/hermes-agent/: Could not resolve host: github.com',
    'fatal: not a git repository (or any of the parent directories): .git',
    'ssh: connect to host github.com port 22: Operation timed out',
    ''
  ]) {
    assert.equal(describeGitExitFailure(stderr), null)
  }
})
