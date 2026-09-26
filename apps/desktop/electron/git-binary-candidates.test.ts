import assert from 'node:assert/strict'
import path from 'node:path'

import { describe, test } from 'vitest'

import { type GitCandidateFs, ugitGitBinaries, windowsGitCandidates } from './git-binary-candidates'

const LAD = path.join('C:', 'Users', 'suceru', 'AppData', 'Local')

function fakeFs(dirs: Record<string, string[]>, files: string[]): GitCandidateFs {
  return {
    existsSync: candidate => files.includes(candidate),
    readdirSync: dir => {
      const entries = dirs[dir]

      if (!entries) {
        throw Object.assign(new Error(`ENOENT: no such file or directory, scandir '${dir}'`), { code: 'ENOENT' })
      }

      return [...entries]
    }
  }
}

const ugitGitExe = (version: string): string =>
  path.join(LAD, 'UGit', `app-${version}`, 'resources', 'app', 'git', 'cmd', 'git.exe')

describe('ugitGitBinaries (#61494)', () => {
  test('enumerates the UGit app-* glob and returns each bundled git.exe, newest first', () => {
    const fs = fakeFs(
      { [path.join(LAD, 'UGit')]: ['app-5.50.1', 'app-5.51.0', 'Update.exe', 'tools'] },
      [ugitGitExe('5.50.1'), ugitGitExe('5.51.0')]
    )

    assert.deepEqual(ugitGitBinaries(LAD, fs), [ugitGitExe('5.51.0'), ugitGitExe('5.50.1')])
  })

  test('sorts app dirs by version, not lexically (app-10.0.0 beats app-9.0.0)', () => {
    const fs = fakeFs(
      { [path.join(LAD, 'UGit')]: ['app-9.0.0', 'app-10.0.0'] },
      [ugitGitExe('9.0.0'), ugitGitExe('10.0.0')]
    )

    assert.deepEqual(ugitGitBinaries(LAD, fs), [ugitGitExe('10.0.0'), ugitGitExe('9.0.0')])
  })

  test('returns [] when UGit is not installed (a missing dir must not throw)', () => {
    assert.deepEqual(ugitGitBinaries(LAD, fakeFs({}, [])), [])
  })

  test('skips app-* dirs whose bundled git.exe is missing', () => {
    const fs = fakeFs({ [path.join(LAD, 'UGit')]: ['app-5.50.1', 'app-6.0.0'] }, [])

    assert.deepEqual(ugitGitBinaries(LAD, fs), [])
  })
})

describe('windowsGitCandidates (#61494)', () => {
  const env = {
    localAppData: LAD,
    programFiles: path.join('C:', 'Program Files'),
    programFilesX86: path.join('C:', 'Program Files (x86)')
  }

  test('includes the UGit candidate and the resolver selection finds it when nothing earlier exists', () => {
    // The reported machine (#61494): no hermes portable git, no Program Files
    // Git, nothing on PATH — only the UGit-bundled copy exists. An Electron
    // process launched from Explorer inherits the login-time environment
    // block, which lacks the PATH entry the UGit installer added later, so
    // the fixed candidate list must find it on disk.
    const ugitGit = ugitGitExe('5.50.1')
    const fs = fakeFs({ [path.join(LAD, 'UGit')]: ['app-5.50.1'] }, [ugitGit])

    const candidates = windowsGitCandidates(env, fs)

    assert.ok(candidates.includes(ugitGit))
    // Ordered after the hermes-bundled portable git (preferred), before the
    // system-wide defaults.
    assert.ok(
      candidates.indexOf(ugitGit) > candidates.indexOf(path.join(env.localAppData, 'hermes', 'git', 'cmd', 'git.exe'))
    )
    assert.ok(candidates.indexOf(ugitGit) < candidates.indexOf(path.join(env.programFiles, 'Git', 'cmd', 'git.exe')))

    // resolveGitBinary's selection rule: the first existing candidate wins.
    assert.equal(candidates.find(fs.existsSync), ugitGit)
  })

  test('still prefers the hermes-bundled portable git over UGit', () => {
    const ugitGit = ugitGitExe('5.50.1')
    const portable = path.join(env.localAppData, 'hermes', 'git', 'cmd', 'git.exe')
    const fs = fakeFs({ [path.join(LAD, 'UGit')]: ['app-5.50.1'] }, [portable, ugitGit])

    const candidates = windowsGitCandidates(env, fs)

    assert.equal(candidates.find(fs.existsSync), portable)
  })

  test('keeps the historical candidates when UGit is absent', () => {
    const candidates = windowsGitCandidates(env, fakeFs({}, []))

    assert.deepEqual(candidates, [
      path.join(env.localAppData, 'hermes', 'git', 'cmd', 'git.exe'),
      path.join(env.localAppData, 'hermes', 'git', 'bin', 'git.exe'),
      path.join(env.programFiles, 'Git', 'cmd', 'git.exe'),
      path.join(env.programFilesX86, 'Git', 'cmd', 'git.exe'),
      path.join(env.localAppData, 'Programs', 'Git', 'cmd', 'git.exe')
    ])
  })
})
