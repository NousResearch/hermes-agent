// Which ssh client the desktop spawns (#103288).
//
// Windows used to hard-code %SystemRoot%\System32\OpenSSH\ssh.exe for the
// `ssh -G` probes and the SSH terminal, while SshConnection spawned a bare
// `ssh` off PATH. When the in-box OpenSSH component is missing or broken
// (component-store corruption after an update makes every native ssh.exe exit
// 255 with no output), no config could point the desktop at a working client
// such as Git for Windows' bundled `usr\bin\ssh.exe`.
//
// Every ssh spawn now resolves through resolveSshBinary. It is pure: platform,
// env slice and filesystem probe are injected so vitest can drive each branch.
// Non-Windows always gets bare `ssh`, exactly as before.

import path from 'node:path'

import { type GitCandidateFs, windowsGitCandidates } from './git-binary-candidates'

/** Windows env slice the resolver reads (injectable for tests). */
export interface WindowsSshEnv {
  localAppData: string
  programFiles: string
  programFilesX86: string
  systemRoot: string
}

export interface SshBinaryInputs {
  platform: string
  /** `desktop.ssh_path` from config.yaml; empty when unset. Windows only. */
  override?: string
  env: WindowsSshEnv
  fs: GitCandidateFs
}

/** The in-box Windows OpenSSH client under a Windows root. */
export function system32OpenSsh(systemRoot: string): string {
  return path.win32.join(systemRoot || 'C:\\Windows', 'System32', 'OpenSSH', 'ssh.exe')
}

/**
 * Git-for-Windows ssh.exe candidates, derived from the same install list
 * resolveGitBinary uses (Hermes PortableGit, UGit, Program Files, per-user).
 * Git ships its MSYS OpenSSH at `<git root>\usr\bin\ssh.exe`, and every git
 * candidate is `<git root>\{cmd,bin}\git.exe`.
 */
export function gitForWindowsSshCandidates(env: WindowsSshEnv, fs: GitCandidateFs): string[] {
  const roots = windowsGitCandidates(env, fs).map(gitExe => path.win32.dirname(path.win32.dirname(gitExe)))

  return [...new Set(roots)].map(root => path.win32.join(root, 'usr', 'bin', 'ssh.exe'))
}

/**
 * The ssh executable to spawn.
 *
 * Windows, in order: an explicit `desktop.ssh_path` (returned as-is so a typo
 * fails loudly with its own path instead of silently using another client),
 * then the in-box System32 OpenSSH, then Git for Windows' bundled ssh.exe,
 * then bare `ssh` for PATH lookup. Every other platform: bare `ssh`.
 */
export function resolveSshBinary({ platform, override, env, fs }: SshBinaryInputs): string {
  if (platform !== 'win32') {
    return 'ssh'
  }

  const explicit = String(override ?? '').trim()

  if (explicit) {
    return explicit
  }

  const inbox = system32OpenSsh(env.systemRoot)

  if (fs.existsSync(inbox)) {
    return inbox
  }

  return gitForWindowsSshCandidates(env, fs).find(candidate => fs.existsSync(candidate)) || 'ssh'
}
