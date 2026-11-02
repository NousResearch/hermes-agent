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

// Helpers enter through the launcher's own module dispatch, which owns PM activation;
// every caller then needs only the launcher path it already carries.
export function windowsHelperArgv(runtime: WindowsRuntime, operation: string, args: string[] = []): string[] {
  return [runtime.hermesPath, '--run-module', 'hermes_cli.windows_ssh_runtime', operation, ...args]
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
