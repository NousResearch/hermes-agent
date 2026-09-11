/**
 * windows-powershell-path.ts
 *
 * The one place that resolves Windows PowerShell.
 *
 * Never spawn `powershell` by bare name: PATH is attacker-influenced (and
 * during an update the venv's own `Scripts` directory is prepended to it and
 * rewritten mid-flight), so a bare name can resolve to something other than
 * the inbox interpreter. Four modules had grown their own private copy of
 * this join; the update-termination path now shares this one.
 */

import path from 'node:path'

/** `%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe`, always absolute. */
export function windowsPowerShellExecutable(systemRoot = process.env.SystemRoot || 'C:\\Windows'): string {
  return path.join(systemRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
}

/** `%SystemRoot%\System32`. */
export function windowsSystem32Dir(systemRoot = process.env.SystemRoot || 'C:\\Windows'): string {
  return path.join(systemRoot, 'System32')
}

/** `%SystemRoot%\System32\<name>`, for the other inbox tools we shell out to. */
export function windowsSystem32Executable(
  name: string,
  systemRoot = process.env.SystemRoot || 'C:\\Windows'
): string {
  return path.join(windowsSystem32Dir(systemRoot), name)
}
