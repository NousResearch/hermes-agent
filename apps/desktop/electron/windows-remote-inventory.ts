import { parseRemoteProfileListing } from './connection-registry'
import { assertWindowsRemoteInstallUpdateClear, encodedPowerShell, powerShellStdinCommand, psLiteral, stripPowerShellNoise } from './windows-remote-lifecycle'
import { WINDOWS_NO_REPARSE_PS, type WindowsRuntime, windowsRuntimeProgramArgv } from './windows-runtime-command'

// Use the canonical identity/tombstone predicate without list_profile_names' OSError suppression.
// No configs, credentials or sessions are opened, and inventory never creates a home or identity.
export const WINDOWS_PROFILE_LIST_PY = [
  'import json, sys',
  'sys.dont_write_bytecode = True',
  'from hermes_cli.profiles import _iter_named_profile_dirs',
  "print(json.dumps({'profiles': ['default', *[p.name for p in _iter_named_profile_dirs()]]}, ensure_ascii=False))"
].join('\n')

const ID_NOFOLLOW_PS = `Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class HermesInventoryNoFollow {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  private static extern SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr security, uint creation, uint flags, IntPtr template);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  private static extern uint GetFinalPathNameByHandle(SafeFileHandle handle, StringBuilder path, uint size, uint flags);
  public static FileStream OpenRead(string name) {
    var handle=CreateFile(name, 0x80000000, 0x00000007, IntPtr.Zero, 3, 0x00200000, IntPtr.Zero);
    if(handle.IsInvalid) Marshal.ThrowExceptionForHR(Marshal.GetHRForLastWin32Error());
    try {
      var path=new StringBuilder(32768);
      var size=GetFinalPathNameByHandle(handle,path,32768,0);
      var actual=path.ToString();
      if(actual.StartsWith(@"\\\\?\\")) actual=actual.Substring(4);
      if(size==0 || size>=32768 || !String.Equals(actual,Path.GetFullPath(name),StringComparison.OrdinalIgnoreCase)) throw new IOException("Inventory path escaped its expected location");
      return new FileStream(handle, FileAccess.Read);
    } catch { handle.Dispose(); throw; }
  }
}
'@
`

export function windowsInventoryScript(runtime: WindowsRuntime, operation: 'install-id' | 'profiles'): string {
  const prefix = [
    '$ProgressPreference="SilentlyContinue"',
    '$ErrorActionPreference="Stop"',
    '[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)',
    WINDOWS_NO_REPARSE_PS,
    `$hermesHome=${psLiteral(runtime.hermesHome)}`,
    'if($hermesHome -notmatch "^[a-zA-Z]:[\\\\/]"){throw "Unsafe Windows Hermes home"}',
    'Assert-NoReparse $hermesHome $false',
    '$installRoot=$hermesHome;$parent=Split-Path -Parent $hermesHome',
    'if((Split-Path -Leaf $parent) -ieq "profiles"){$installRoot=Split-Path -Parent $parent}',
    'Assert-NoReparse $installRoot $false'
  ]

  if (operation === 'profiles') {
    const argv = windowsRuntimeProgramArgv(runtime, WINDOWS_PROFILE_LIST_PY)
    prefix.push(
      '$profilesDir=Join-Path $installRoot "profiles"',
      'Assert-NoReparse $profilesDir $true',
      // Only metadata: reject links before the canonical predicate can follow any directory.
      'if(Test-Path -LiteralPath $profilesDir -PathType Container){foreach($entry in Get-ChildItem -LiteralPath $profilesDir -Force){Assert-NoReparse $entry.FullName $false;if($entry.PSIsContainer){foreach($child in Get-ChildItem -LiteralPath $entry.FullName -Force){Assert-NoReparse $child.FullName $false}}}}',
      '$env:HERMES_HOME=$hermesHome',
      `& ${argv.map(psLiteral).join(' ')}`,
      'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}'
    )
  } else {
    prefix.push(
      ID_NOFOLLOW_PS,
      '$file=Join-Path $installRoot "install_id";$id=$null',
      'Assert-NoReparse $file $true',
      'if(Test-Path -LiteralPath $file -PathType Leaf){$stream=$null;$memory=$null;try{$stream=[HermesInventoryNoFollow]::OpenRead($file);Assert-NoReparse $file $false;$memory=New-Object IO.MemoryStream;$buffer=New-Object byte[] 256;$count=$stream.Read($buffer,0,256);if($count -lt 256){$id=[Text.Encoding]::UTF8.GetString($buffer,0,$count).Trim([char]0xFEFF).Trim().ToLowerInvariant()}}finally{if($memory){$memory.Dispose()};if($stream){$stream.Dispose()}}}',
      '[ordered]@{id=$id}|ConvertTo-Json -Compress'
    )
  }

  return prefix.join('\r\n')
}

async function readInventory(ssh: any, runtime: WindowsRuntime, operation: 'install-id' | 'profiles'): Promise<any> {
  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome, runtime.python)

  const output = await ssh.exec(powerShellStdinCommand(), {
    stdinData: `${encodedPowerShell(windowsInventoryScript(runtime, operation))}\r\n`
  })

  const parsed = JSON.parse(stripPowerShellNoise(output).pop() || 'null')

  if (!parsed || typeof parsed !== 'object' || parsed.error) {throw new Error('Windows inventory did not return valid JSON.')}

  return parsed
}

export async function listWindowsRemoteProfiles(ssh: any, runtime: WindowsRuntime): Promise<string[]> {
  const result = await readInventory(ssh, runtime, 'profiles')

  if (!Array.isArray(result.profiles) || !result.profiles.every((name: unknown) => typeof name === 'string') || !result.profiles.includes('default')) {
    throw new Error('Windows inventory did not return canonical profile names.')
  }

  return parseRemoteProfileListing(result.profiles.join('\n'))
}

export async function readWindowsRemoteInstallId(ssh: any, runtime: WindowsRuntime): Promise<string | undefined> {
  const result = await readInventory(ssh, runtime, 'install-id')

  if (!Object.hasOwn(result, 'id')) {throw new Error('Windows inventory did not report its install identity.')}
  const id = typeof result.id === 'string' ? result.id.trim().toLowerCase() : ''

  return /^[0-9a-f]{32}$/.test(id) ? id : undefined
}
