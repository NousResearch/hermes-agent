import assert from 'node:assert/strict'
import path from 'node:path'

import { describe, test } from 'vitest'

import { type GitCandidateFs, selectWindowsGitBinary, windowsGitCandidates } from './git-binary-candidates'

const LAD = path.join('C:', 'Users', 'tester', 'AppData', 'Local')

function fakeFs(dirs: Record<string, string[]>, files: string[]): GitCandidateFs {
  return {
    existsSync: candidate => files.includes(candidate),
    readdirSync: dir => [...(dirs[dir] ?? [])]
  }
}

const env = {
  localAppData: LAD,
  programFiles: path.join('C:', 'Program Files'),
  programFilesX86: path.join('C:', 'Program Files (x86)')
}

describe('selectWindowsGitBinary', () => {
  test('skips an existing but unlaunchable candidate for a later one that runs', () => {
    // The reported host: the PM-pinned git exists but its ACLs were replaced by
    // archive metadata on extraction, so the interactive user's token cannot
    // spawn it (ERROR_ACCESS_DENIED) while the system Git further down the list
    // works. Selecting by existence alone commits to the broken copy and the
    // update check answers head-unavailable.
    const managed = path.join(LAD, 'hermes', 'tools', 'git-2.53.0+3-win32-x64', 'cmd', 'git.exe')
    const system = path.join(env.programFiles, 'Git', 'cmd', 'git.exe')
    const fs = fakeFs({ [path.join(LAD, 'hermes', 'tools')]: ['git-2.53.0+3-win32-x64'] }, [managed, system])

    const selected = selectWindowsGitBinary({
      env,
      fs,
      binaryRuns: candidate => candidate === system
    })

    assert.equal(selected, system)
  })

  test('keeps the first candidate when it exists and runs', () => {
    const managed = path.join(LAD, 'hermes', 'tools', 'git-2.53.0+3-win32-x64', 'cmd', 'git.exe')
    const system = path.join(env.programFiles, 'Git', 'cmd', 'git.exe')
    const fs = fakeFs({ [path.join(LAD, 'hermes', 'tools')]: ['git-2.53.0+3-win32-x64'] }, [managed, system])

    const selected = selectWindowsGitBinary({ env, fs, binaryRuns: () => true })

    assert.equal(selected, managed)
  })

  test('falls back to the first existing candidate when none can be probed', () => {
    // The probe itself can fail where spawning is blocked (locked-down execution
    // policy, an antivirus interposing on spawn). Behaviour must not regress
    // there: an existing candidate still wins over no candidate at all.
    const managed = path.join(LAD, 'hermes', 'tools', 'git-2.53.0+3-win32-x64', 'cmd', 'git.exe')
    const system = path.join(env.programFiles, 'Git', 'cmd', 'git.exe')
    const fs = fakeFs({ [path.join(LAD, 'hermes', 'tools')]: ['git-2.53.0+3-win32-x64'] }, [managed, system])

    const selected = selectWindowsGitBinary({ env, fs, binaryRuns: () => false })

    assert.equal(selected, managed)
  })

  test('returns null when no candidate exists', () => {
    const fs = fakeFs({}, [])

    assert.equal(selectWindowsGitBinary({ env, fs, binaryRuns: () => true }), null)
  })

  test('a candidate that runs but was not found on disk is never selected', () => {
    // The probe must not smuggle in a path the candidate list never produced:
    // a missing candidate is not spawned, and a path outside the list is not
    // this resolver's to choose.
    const fs = fakeFs({}, [])

    assert.equal(selectWindowsGitBinary({ env, fs, binaryRuns: () => true }), null)
  })
})

describe('windowsGitCandidates', () => {
  test('the managed git leads the list and the system git trails it', () => {
    const managed = path.join(LAD, 'hermes', 'tools', 'git-2.53.0+3-win32-x64', 'cmd', 'git.exe')
    const system = path.join(env.programFiles, 'Git', 'cmd', 'git.exe')
    const fs = fakeFs({ [path.join(LAD, 'hermes', 'tools')]: ['git-2.53.0+3-win32-x64'] }, [managed, system])

    const candidates = windowsGitCandidates(env, fs)

    assert.equal(candidates[0], managed)
    assert.ok(candidates.includes(system))
  })
})
