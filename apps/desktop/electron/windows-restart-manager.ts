/**
 * Windows Restart Manager lock-owner discovery for update force-release.
 *
 * Prefer RmGetList for exact locked resources over command-line heuristics.
 * Implemented via PowerShell + P/Invoke from Microsoft RM docs (not SuperF4).
 *
 * Resources are handed to PowerShell through a UTF-8 list file, never inline:
 * the venv mutation set can hold hundreds of paths and a `-Command` string is
 * capped by the Windows command line. Each line is `<flag>\t<path>`:
 *
 * - `D` (definite): a locked file with a single hard link. Whoever RM names
 *   holds *our* link and is a holder. Small lists are queried one file per RM
 *   session so each holder is attributed to its exact resource; large lists
 *   use one batched session and are attributed to the first file.
 * - `A` (ambiguous): a locked file that uv also hard-linked into other venvs
 *   or its cache. RM identifies holders by file, not by link, so a foreign
 *   venv mapping the same wheel through its own link is listed too. Each RM
 *   holder of an ambiguous batch is therefore kept only when its module list
 *   contains a path under the attribution root, and that mapped path becomes
 *   its resource. Holders that only map another link cannot break our unlink
 *   and are dropped -- but only when the module list was actually read. A
 *   holder whose modules cannot be enumerated is reported UNATTRIBUTED (it
 *   still blocks the update) and never carries an ownership proof.
 */

import { execFile } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { windowsPowerShellExecutable } from './windows-powershell-path'
import { psLiteral } from './windows-remote-lifecycle'
import type { ForceReleaseHolder } from './windows-update-force-release'

const execFileAsync = promisify(execFile)

/**
 * Distinct from `RunPowerShell` in windows-process-terminate.ts, which also
 * carries an abort signal, an absolute deadline and a hard-boundary dependency
 * bag. This is the plain query runner; the two are not interchangeable.
 */
export type RunRestartManagerPowerShell = (
  script: string,
  timeoutMs?: number
) => Promise<{ stdout: string; stderr: string; code: number }>

/** Above this many definite resources RM runs one batched session instead of one per file. */
export const RESTART_MANAGER_PER_FILE_LIMIT = 12

/**
 * RmGetList resolves a friendly application name per holder, which costs
 * roughly half a second for each console process. A handful of holders plus
 * PowerShell start-up fits comfortably; the old 3.5 s cap returned an empty
 * holder set on every real install.
 */
export const RESTART_MANAGER_DEFAULT_TIMEOUT_MS = 12_000

async function runRestartManagerPowerShell(
  script: string,
  timeoutMs = 4_000
): Promise<{ stdout: string; stderr: string; code: number }> {
  const budget = Math.max(1, Math.trunc(timeoutMs))

  try {
    const { stdout, stderr } = await execFileAsync(
      windowsPowerShellExecutable(),
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
      { encoding: 'utf8', timeout: budget, windowsHide: true, maxBuffer: 2 * 1024 * 1024 }
    )

    return { stdout: String(stdout ?? ''), stderr: String(stderr ?? ''), code: 0 }
  } catch (error: any) {
    return {
      stdout: String(error?.stdout ?? ''),
      stderr: String(error?.stderr ?? error?.message ?? ''),
      code: typeof error?.code === 'number' ? error.code : 1
    }
  }
}

/**
 * Emitted PowerShell must split RM rows on a literal pipe.
 * Prefer String.Split over -split regex so TS template escaping cannot
 * accidentally emit a character-class / alternation pattern. Module-private:
 * the contract is proved by running the generated script through real
 * PowerShell (windows-restart-manager.windows-live.test.ts), not by restating
 * this literal in an assertion.
 */
const RESTART_MANAGER_ROW_SPLIT_EXPRESSION = "$part.Split([char]'|', 4)"

/**
 * P/Invoke surface for Restart Manager, compiled by `Add-Type -TypeDefinition`
 * on every run.
 *
 * There is deliberately NO cached assembly. The previous cache wrote
 * `%TEMP%\hermes-restart-manager\HermesRm-<sha1-of-source>.dll` and loaded it
 * with `Add-Type -Path` whenever it existed: a deterministic, same-user
 * writable path holding executable code that was never verified. Recording a
 * hash beside it does not close that hole -- whoever can replace the DLL can
 * replace the hash file -- and a per-process temp directory is exactly what
 * the compiler already does with a random name. Compilation costs ~0.5-0.9 s
 * (measured, Windows PowerShell 5.1), which is affordable now that attribution
 * runs once per release-gate run instead of once per 300 ms poll.
 */
export const RESTART_MANAGER_NATIVE_SOURCE = `
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class HermesRm {
  public const int CCH_RM_SESSION_KEY = 32;
  public const int CCH_RM_MAX_APP_NAME = 255;
  public const int CCH_RM_MAX_SVC_NAME = 63;
  public enum RM_APP_TYPE { RmUnknownApp=0, RmMainWindow=1, RmOtherWindow=2, RmService=3, RmExplorer=4, RmConsole=5, RmCritical=1000 }
  [StructLayout(LayoutKind.Sequential)]
  public struct FILETIME { public uint dwLowDateTime; public uint dwHighDateTime; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  public struct RM_UNIQUE_PROCESS { public int dwProcessId; public FILETIME ProcessStartTime; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  public struct RM_PROCESS_INFO {
    public RM_UNIQUE_PROCESS Process;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=CCH_RM_MAX_APP_NAME+1)] public string strAppName;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=CCH_RM_MAX_SVC_NAME+1)] public string strServiceShortName;
    public RM_APP_TYPE ApplicationType;
    public uint AppStatus;
    public uint TSSessionId;
    [MarshalAs(UnmanagedType.Bool)] public bool bRestartable;
  }
  // strSessionKey is an OUT buffer of CCH_RM_SESSION_KEY+1 WCHARs per RM docs.
  [DllImport("rstrtmgr.dll", CharSet=CharSet.Unicode)] public static extern int RmStartSession(out uint pSessionHandle, int dwSessionFlags, StringBuilder strSessionKey);
  [DllImport("rstrtmgr.dll")] public static extern int RmEndSession(uint pSessionHandle);
  [DllImport("rstrtmgr.dll", CharSet=CharSet.Unicode)] public static extern int RmRegisterResources(uint pSessionHandle, uint nFiles, string[] rgsFilenames, uint nApplications, IntPtr rgApplications, uint nServices, string[] rgsServiceNames);
  [DllImport("rstrtmgr.dll")] public static extern int RmGetList(uint dwSessionHandle, out uint pnProcInfoNeeded, ref uint pnProcInfo, [In,Out] RM_PROCESS_INFO[] rgAffectedApps, ref uint lpdwRebootReasons);
  // Attribution compares CANONICAL paths. A module list reports the path the
  // loader used; the attribution root arrives from the desktop's own
  // resolution. Junctions, symlinks, 8.3 short names and mixed case make two
  // spellings of the same file compare unequal, which silently drops a real
  // holder. Same resolution the terminate script uses: GetFinalPathNameByHandle.
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  private static extern IntPtr CreateFileW(string path, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  private static extern uint GetFinalPathNameByHandleW(IntPtr handle, StringBuilder buffer, uint length, uint flags);
  [DllImport("kernel32.dll", SetLastError=true)]
  private static extern bool CloseHandle(IntPtr handle);
  private const char PathSep = '\\\\';
  private static string StripExtendedPrefix(string value) {
    if (string.IsNullOrEmpty(value)) return "";
    if (value.Length <= 4 || value[0] != PathSep || value[1] != PathSep || value[2] != '?' || value[3] != PathSep) return value;
    string rest = value.Substring(4);
    bool unc = rest.Length > 4 && rest[3] == PathSep
      && (rest[0] == 'U' || rest[0] == 'u') && (rest[1] == 'N' || rest[1] == 'n') && (rest[2] == 'C' || rest[2] == 'c');
    return unc ? new string(PathSep, 2) + rest.Substring(4) : rest;
  }
  public static string FinalPath(string target) {
    if (string.IsNullOrEmpty(target)) return "";
    string full;
    try { full = System.IO.Path.GetFullPath(target); } catch { full = target; }
    IntPtr handle = CreateFileW(full, 0, 7, IntPtr.Zero, 3, 0x02000000, IntPtr.Zero);
    if (handle == new IntPtr(-1)) return StripExtendedPrefix(full);
    try {
      StringBuilder buffer = new StringBuilder(4096);
      uint written = GetFinalPathNameByHandleW(handle, buffer, 4095, 0);
      if (written == 0 || written > 4095) return StripExtendedPrefix(full);
      return StripExtendedPrefix(buffer.ToString());
    } finally { CloseHandle(handle); }
  }
  public static bool IsSameOrUnderRoot(string child, string root) {
    if (string.IsNullOrEmpty(child) || string.IsNullOrEmpty(root)) return false;
    string c = child.TrimEnd(PathSep);
    string r = root.TrimEnd(PathSep);
    if (c.Length == 0 || r.Length == 0) return false;
    if (string.Equals(c, r, StringComparison.OrdinalIgnoreCase)) return true;
    return c.StartsWith(r + PathSep, StringComparison.OrdinalIgnoreCase);
  }
  // One RM session over every file in the batch. Rows carry the batch label
  // as their resource because RM does not report per-file ownership.
  public static string Query(string[] files, string label) {
    if (files == null || files.Length == 0) return "";
    uint handle;
    StringBuilder key = new StringBuilder(CCH_RM_SESSION_KEY + 1);
    int rc = RmStartSession(out handle, 0, key);
    if (rc != 0) return "";
    try {
      rc = RmRegisterResources(handle, (uint)files.Length, files, 0, IntPtr.Zero, 0, null);
      if (rc != 0) return "";
      uint needed = 0, count = 0, reboot = 0;
      rc = RmGetList(handle, out needed, ref count, null, ref reboot);
      if (rc == 234) { // ERROR_MORE_DATA
        count = needed;
        RM_PROCESS_INFO[] arr = new RM_PROCESS_INFO[count];
        rc = RmGetList(handle, out needed, ref count, arr, ref reboot);
        if (rc != 0) return "";
        var parts = new System.Collections.Generic.List<string>();
        string safeLabel = (label ?? "").Replace("|","/");
        for (int i=0;i<count;i++) {
          var p = arr[i];
          long fileTime = ((long)p.Process.ProcessStartTime.dwHighDateTime << 32) | (uint)p.Process.ProcessStartTime.dwLowDateTime;
          // FILETIME is 100ns since 1601; convert to unix seconds
          long unix = (fileTime - 116444736000000000L) / 10000000L;
          string name = (p.strAppName ?? "").Replace("|","/");
          parts.Add(p.Process.dwProcessId.ToString() + "|" + unix.ToString() + "|" + name + "|" + safeLabel);
        }
        return string.Join(";", parts);
      }
      return "";
    } finally { RmEndSession(handle); }
  }
}
`.trim()

/**
 * Attribute one ambiguous (hard-link-shared) RM holder to a module it maps
 * under the attribution root.
 *
 * Three outcomes, and the difference between the last two is the whole point:
 *   mapped      -> it maps OUR link; the canonical module path is the proof.
 *   foreign     -> its module list was read in full and contains nothing under
 *                  the root; another venv's link, cannot block our unlink.
 *   unattributed-> the question could not be answered (the process is gone,
 *                  its create time no longer matches the generation RM named,
 *                  or Process.Modules threw: access denied, bitness mismatch,
 *                  a module list changing under enumeration). It is still
 *                  reported as a holder -- the update must not proceed -- but
 *                  it carries no ownership proof, so the terminate boundary
 *                  will not accept it as an authorized target on that basis.
 *
 * Exported because the generated script embeds it verbatim and the live suite
 * runs it against synthetic rows under real PowerShell.
 */
export const RESTART_MANAGER_ATTRIBUTION_FUNCTION = `
function Resolve-HolderAttribution([int]$holderPid, [double]$expectedCreatedAt, [string]$rootClaim) {
  if ([string]::IsNullOrWhiteSpace($rootClaim)) { return [pscustomobject]@{ status = 'unattributed'; path = '' } }
  # Canonicalize the root with the SAME resolution applied to each module
  # below. The desktop passes the install path as it resolved it; a junction, a
  # symlinked install, an 8.3 short name or a different case all spell the same
  # directory, and the raw prefix compare this replaced made a real holder of
  # our own venv look like somebody else's link.
  $root = [HermesRm]::FinalPath($rootClaim)
  if ([string]::IsNullOrWhiteSpace($root)) { return [pscustomobject]@{ status = 'unattributed'; path = '' } }
  $proc = $null
  try { $proc = [System.Diagnostics.Process]::GetProcessById($holderPid) } catch { $proc = $null }
  if ($null -eq $proc) { return [pscustomobject]@{ status = 'unattributed'; path = '' } }
  $liveCreated = $null
  try { $liveCreated = [DateTimeOffset]::new($proc.StartTime.ToUniversalTime()).ToUnixTimeSeconds() } catch { $liveCreated = $null }
  if ($null -eq $liveCreated) { return [pscustomobject]@{ status = 'unattributed'; path = '' } }
  # PID reuse: RM named one generation, GetProcessById returns whoever owns the
  # PID now. Never read a stranger's module list as this holder's evidence.
  if ([Math]::Abs([double]$liveCreated - $expectedCreatedAt) -gt 1.5) { return [pscustomobject]@{ status = 'unattributed'; path = '' } }
  $modules = $null
  try { $modules = @($proc.Modules) } catch { $modules = $null }
  if ($null -eq $modules) { return [pscustomobject]@{ status = 'unattributed'; path = '' } }
  foreach ($module in $modules) {
    $moduleFinal = ''
    try { $moduleFinal = [HermesRm]::FinalPath([string]$module.FileName) } catch { $moduleFinal = '' }
    if ([HermesRm]::IsSameOrUnderRoot($moduleFinal, $root)) {
      return [pscustomobject]@{ status = 'mapped'; path = $moduleFinal }
    }
  }
  return [pscustomobject]@{ status = 'foreign'; path = '' }
}
`.trim()

export function buildRestartManagerScript(
  resourceListPath: string,
  {
    perFileLimit = RESTART_MANAGER_PER_FILE_LIMIT,
    attributionRoot = ''
  }: { perFileLimit?: number; attributionRoot?: string } = {}
): string {
  const attribution = attributionRoot ? path.resolve(attributionRoot) : ''

  return `
$ErrorActionPreference = 'Stop'
$definite = @()
$ambiguous = @()
foreach ($line in [System.IO.File]::ReadAllLines(${psLiteral(resourceListPath)}, [System.Text.Encoding]::UTF8)) {
  if (-not $line) { continue }
  $tab = $line.IndexOf([char]9)
  if ($tab -lt 1) { continue }
  $flag = $line.Substring(0, $tab)
  $target = $line.Substring($tab + 1)
  if (-not $target) { continue }
  if ($flag -eq 'A') { $ambiguous += $target } else { $definite += $target }
}
if (($definite.Count + $ambiguous.Count) -eq 0) { Write-Output '[]'; exit 0 }
$attributionRootClaim = ${psLiteral(attribution)}
$rmSource = @"
${RESTART_MANAGER_NATIVE_SOURCE}
"@
# No cached assembly: see RESTART_MANAGER_NATIVE_SOURCE. Add-Type compiles into
# the compiler's own randomly named temp output, which no other process can
# predict; a deterministic %TEMP%\HermesRm-<hash>.dll was loadable code that
# nothing verified.
Add-Type -TypeDefinition $rmSource
${RESTART_MANAGER_ATTRIBUTION_FUNCTION}
function Convert-RmRows([string]$raw, [string]$label) {
  $rows = @()
  if (-not $raw) { return $rows }
  foreach ($part in ($raw -split ';')) {
    if (-not $part) { continue }
    $bits = ${RESTART_MANAGER_ROW_SPLIT_EXPRESSION}
    if ($bits.Count -lt 2) { continue }
    $pidVal = 0; $created = 0.0
    if (-not [int]::TryParse($bits[0], [ref]$pidVal)) { continue }
    if (-not [double]::TryParse($bits[1], [ref]$created)) { continue }
    if ($pidVal -le 0 -or $created -le 0) { continue }
    $name = if ($bits.Count -ge 3) { $bits[2] } else { 'unknown' }
    $res = if ($bits.Count -ge 4 -and $bits[3]) { $bits[3] } else { $label }
    $rows += [pscustomobject]@{ pid = $pidVal; createdAt = $created; name = $name; resource = $res; attribution = 'proven' }
  }
  return $rows
}
$items = @()
$batches = @()
if ($definite.Count -gt 0 -and $definite.Count -le ${Math.max(1, Math.trunc(perFileLimit))}) {
  foreach ($resource in $definite) { $batches += ,@([string[]]@($resource)) }
} elseif ($definite.Count -gt 0) {
  $batches += ,@([string[]]$definite)
}
foreach ($batch in $batches) {
  try { $raw = [HermesRm]::Query([string[]]$batch, [string]$batch[0]) } catch { continue }
  foreach ($row in @(Convert-RmRows $raw ([string]$batch[0]))) { $items += $row }
}
if ($ambiguous.Count -gt 0 -and $attributionRootClaim) {
  try { $raw = [HermesRm]::Query([string[]]$ambiguous, [string]$ambiguous[0]) } catch { $raw = '' }
  foreach ($row in @(Convert-RmRows $raw ([string]$ambiguous[0]))) {
    $resolved = Resolve-HolderAttribution ([int]$row.pid) ([double]$row.createdAt) $attributionRootClaim
    if ($resolved.status -eq 'foreign') { continue }
    if ($resolved.status -eq 'mapped') {
      $row.resource = $resolved.path
    } else {
      $row.attribution = 'unattributed'
    }
    $items += $row
  }
}
$items | ConvertTo-Json -Compress -Depth 3
`.trim()
}

export function parseRestartManagerOutput(
  stdout: string,
  resources: readonly string[]
): ForceReleaseHolder[] {
  const text = String(stdout || '').trim()

  if (!text || text === '[]') {return []}

  let parsed: any

  try {
    parsed = JSON.parse(text)
  } catch {
    return []
  }

  const rows = Array.isArray(parsed) ? parsed : [parsed]
  const fallbackResource = resources[0]
  const holders: ForceReleaseHolder[] = []
  const seen = new Set<string>()

  for (const row of rows) {
    const pid = Number(row?.pid)
    const createdAt = Number(row?.createdAt)
    const name = typeof row?.name === 'string' && row.name ? row.name : 'unknown'

    const resource =
      typeof row?.resource === 'string' && row.resource
        ? row.resource
        : fallbackResource

    // An unattributed ambiguous holder blocks the update but proves nothing
    // about ownership of a specific link. `resource` is the ownership claim
    // the terminate boundary re-proves, so it must stay empty; the path is
    // still carried in `resources` for the refusal message and the log.
    const unattributed = row?.attribution === 'unattributed'

    if (!Number.isInteger(pid) || pid <= 0) {continue}

    if (!Number.isFinite(createdAt) || createdAt <= 0) {continue}

    // A batched session reports each holder once; per-file sessions can
    // report the same holder for several files. Keep the first attribution.
    const key = `${pid}:${createdAt}`

    if (seen.has(key)) {continue}
    seen.add(key)

    holders.push({
      pid,
      createdAt,
      name,
      cmdline: name,
      source: 'restart-manager',
      ...(unattributed ? { resources: resource ? [resource] : [] } : { resource }),
      role: 'other'
    })
  }

  return holders
}

export function writeRestartManagerResourceList(
  definite: readonly string[],
  shared: readonly string[] = [],
  dir = os.tmpdir()
): string {
  const target = path.join(dir, `hermes-rm-resources-${process.pid}-${crypto.randomBytes(6).toString('hex')}.txt`)

  const lines = [
    ...definite.filter(Boolean).map(resource => `D\t${resource}`),
    ...shared.filter(Boolean).map(resource => `A\t${resource}`)
  ]

  fs.writeFileSync(target, `${lines.join('\n')}\n`, { encoding: 'utf8', mode: 0o600 })

  return target
}

export async function listRestartManagerHoldersForResources(
  resources: readonly string[],
  {
    platform = process.platform,
    run = runRestartManagerPowerShell,
    timeoutMs = RESTART_MANAGER_DEFAULT_TIMEOUT_MS,
    listDir,
    shared = [],
    attributionRoot
  }: {
    platform?: NodeJS.Platform
    run?: RunRestartManagerPowerShell
    timeoutMs?: number
    listDir?: string
    /** Locked files that other hard links share; holders are kept only when they map a path under `attributionRoot`. */
    shared?: readonly string[]
    attributionRoot?: string
  } = {}
): Promise<ForceReleaseHolder[]> {
  const definite = resources.filter(Boolean)
  const ambiguous = attributionRoot ? shared.filter(Boolean) : []

  if (platform !== 'win32' || definite.length + ambiguous.length === 0) {
    return []
  }

  const budget = Math.max(1, Math.trunc(timeoutMs))
  let listPath: string | null = null

  try {
    listPath = writeRestartManagerResourceList(definite, ambiguous, listDir)
  } catch {
    return []
  }

  try {
    const script = buildRestartManagerScript(listPath, {
      ...(attributionRoot ? { attributionRoot } : {})
    })

    const result = await run(script, budget)

    return parseRestartManagerOutput(result.stdout, [...definite, ...ambiguous])
  } finally {
    try {
      fs.rmSync(listPath, { force: true })
    } catch {
      void 0
    }
  }
}
