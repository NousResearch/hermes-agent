// #103288: one resolver picks the ssh client for every desktop spawn. Platform,
// env and filesystem are passed as data; nothing touches process.platform.
import { describe, expect, it } from 'vitest'

import { gitForWindowsSshCandidates, resolveSshBinary, system32OpenSsh, type WindowsSshEnv } from './ssh-binary'

const ENV: WindowsSshEnv = {
  systemRoot: 'C:\\Windows',
  localAppData: 'C:\\Users\\me\\AppData\\Local',
  programFiles: 'C:\\Program Files',
  programFilesX86: 'C:\\Program Files (x86)'
}

const SYSTEM32_SSH = 'C:\\Windows\\System32\\OpenSSH\\ssh.exe'
const PROGRAM_FILES_GIT_SSH = 'C:\\Program Files\\Git\\usr\\bin\\ssh.exe'
const PORTABLE_GIT_SSH = 'C:\\Users\\me\\AppData\\Local\\hermes\\git\\usr\\bin\\ssh.exe'

// Compare on win32 separators whatever the host OS joins with.
const norm = (p: string) => p.replace(/\//g, '\\')

function fakeFs(files: string[], dirs: Record<string, string[]> = {}) {
  const present = new Set(files.map(norm))
  const probed: string[] = []

  return {
    probed,
    fs: {
      existsSync: (candidate: string) => {
        probed.push(norm(candidate))

        return present.has(norm(candidate))
      },
      readdirSync: (dir: string) => {
        const entries = dirs[norm(dir)]

        if (!entries) {
          throw Object.assign(new Error(`ENOENT: ${dir}`), { code: 'ENOENT' })
        }

        return entries
      }
    }
  }
}

describe('resolveSshBinary', () => {
  it('keeps bare ssh on non-Windows platforms, ignores the override, and never probes the filesystem', () => {
    for (const platform of ['darwin', 'linux', 'freebsd']) {
      const { fs, probed } = fakeFs([SYSTEM32_SSH])

      expect(resolveSshBinary({ platform, override: '/opt/ssh', env: ENV, fs })).toBe('ssh')
      expect(probed).toEqual([])
    }
  })

  it('returns an explicit desktop.ssh_path override verbatim, even when it does not exist', () => {
    const { fs, probed } = fakeFs([SYSTEM32_SSH])

    expect(resolveSshBinary({ platform: 'win32', override: '  D:\\tools\\ssh.exe  ', env: ENV, fs })).toBe(
      'D:\\tools\\ssh.exe'
    )
    expect(probed).toEqual([])
  })

  it('prefers the in-box System32 OpenSSH when no override is set', () => {
    const { fs } = fakeFs([SYSTEM32_SSH, PROGRAM_FILES_GIT_SSH])

    expect(norm(resolveSshBinary({ platform: 'win32', override: '', env: ENV, fs }))).toBe(SYSTEM32_SSH)
  })

  it('honours a non-default SystemRoot', () => {
    const { fs } = fakeFs(['D:\\Win\\System32\\OpenSSH\\ssh.exe'])

    expect(norm(resolveSshBinary({ platform: 'win32', env: { ...ENV, systemRoot: 'D:\\Win' }, fs }))).toBe(
      'D:\\Win\\System32\\OpenSSH\\ssh.exe'
    )
  })

  it("falls back to Git for Windows' ssh.exe when System32 OpenSSH is missing", () => {
    const { fs } = fakeFs([PROGRAM_FILES_GIT_SSH])

    expect(norm(resolveSshBinary({ platform: 'win32', env: ENV, fs }))).toBe(PROGRAM_FILES_GIT_SSH)
  })

  it('prefers the Hermes PortableGit ssh over a system Git install, matching resolveGitBinary order', () => {
    const { fs } = fakeFs([PORTABLE_GIT_SSH, PROGRAM_FILES_GIT_SSH])

    expect(norm(resolveSshBinary({ platform: 'win32', env: ENV, fs }))).toBe(PORTABLE_GIT_SSH)
  })

  it('falls back to bare ssh (PATH lookup) when no candidate exists', () => {
    const { fs } = fakeFs([])

    expect(resolveSshBinary({ platform: 'win32', env: ENV, fs })).toBe('ssh')
  })
})

describe('gitForWindowsSshCandidates', () => {
  it('maps every resolveGitBinary install root to usr\\bin\\ssh.exe, deduped and in order', () => {
    const ugitRoot = 'C:\\Users\\me\\AppData\\Local\\UGit'
    const ugitGit = `${ugitRoot}\\app-5.50.1\\resources\\app\\git\\cmd\\git.exe`
    const { fs } = fakeFs([ugitGit], { [ugitRoot]: ['app-5.50.1'] })

    expect(gitForWindowsSshCandidates(ENV, fs).map(norm)).toEqual([
      PORTABLE_GIT_SSH,
      `${ugitRoot}\\app-5.50.1\\resources\\app\\git\\usr\\bin\\ssh.exe`,
      PROGRAM_FILES_GIT_SSH,
      'C:\\Program Files (x86)\\Git\\usr\\bin\\ssh.exe',
      'C:\\Users\\me\\AppData\\Local\\Programs\\Git\\usr\\bin\\ssh.exe'
    ])
  })

  it('skips per-user candidates when LOCALAPPDATA is unset', () => {
    const { fs } = fakeFs([])

    expect(gitForWindowsSshCandidates({ ...ENV, localAppData: '' }, fs).map(norm)).toEqual([
      PROGRAM_FILES_GIT_SSH,
      'C:\\Program Files (x86)\\Git\\usr\\bin\\ssh.exe'
    ])
  })
})

describe('system32OpenSsh', () => {
  it('matches the path the desktop hard-coded before #103288', () => {
    expect(system32OpenSsh('C:\\Windows')).toBe(SYSTEM32_SSH)
    expect(system32OpenSsh('')).toBe(SYSTEM32_SSH)
  })
})
