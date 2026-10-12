// The published launcher owns PM activation. A Python guessed beside bin/hermes.exe does not.
export interface WindowsRuntime {
  os?: string
  arch?: string
  hermesHome: string
  hermesPath: string
  python: string
  launcherPath?: string
  runtimeCommand?: string[]
}

export function windowsRuntimeProgramArgv(runtime: WindowsRuntime, program: string): string[] {
  if (!runtime.runtimeCommand) {return [runtime.python, '-c', program]}
  const command = [...runtime.runtimeCommand]
  const codeIndex = command.indexOf('-c')
  const entry = /runpy\.run_module\((['"])hermes_cli\.main\1,\s*run_name=(['"])__main__\2,\s*alter_sys=True\)\s*$/

  if (codeIndex >= 1 && codeIndex === command.length - 2 && entry.test(command[codeIndex + 1])) {
    // PowerShell 5.1 strips embedded double quotes when forming native argv.
    // ASCII hex carries the program losslessly without changing the launcher bootstrap.
    const encodedProgram = Buffer.from(program, 'utf8').toString('hex')
    command[codeIndex + 1] = command[codeIndex + 1].replace(entry, () => `exec(bytes.fromhex('${encodedProgram}').decode('utf-8'))`)

    return command
  }

  // An unrecognized launcher is a capability gap, not permission to drop its bootstrap.
  const error: any = new Error('The Windows launcher does not expose a supported boot-selecting runtime command; refresh its launcher.')
  error.kind = 'update-required'
  throw error
}

export function windowsHelperArgv(runtime: WindowsRuntime, operation: string, args: string[] = []): string[] {
  if (!runtime.runtimeCommand) {return [runtime.python, '-m', 'hermes_cli.windows_ssh_runtime', operation, ...args]}

  return [...windowsRuntimeProgramArgv(runtime,
    "import runpy; runpy.run_module('hermes_cli.windows_ssh_runtime', run_name='__main__', alter_sys=True)"), operation, ...args]
}

// Same ancestor/leaf guard as the existing platform and marker probes.
export const WINDOWS_NO_REPARSE_PS = [
  'function Assert-NoReparse([string]$candidate,[bool]$allowMissing=$false){',
  'if([string]::IsNullOrWhiteSpace($candidate)){throw "Empty Windows path"}',
  '$current=[IO.Path]::GetFullPath($candidate);$first=$true',
  'while($true){',
  'try{$item=Get-Item -LiteralPath $current -Force -ErrorAction Stop}catch [Management.Automation.ItemNotFoundException]{if(-not $allowMissing -and $first){throw "Path was not found: $candidate"};$parent=[IO.Path]::GetDirectoryName($current);if(-not $parent -or $parent -eq $current){break};$current=$parent;$first=$false;continue}',
  'if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Path contains a link or reparse point: $current"}',
  '$parent=[IO.Path]::GetDirectoryName($current);if(-not $parent -or $parent -eq $current){break};$current=$parent;$first=$false',
  '}',
  '}'
].join('\r\n')
