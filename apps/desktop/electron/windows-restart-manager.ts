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
 *   and are dropped.
 */

import { execFile } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

/** One process Restart Manager (or module attribution) proved to hold an install file. */
export type RestartManagerHolder = {
  pid: number
  createdAt: number
  name: string
  cmdline: string
  source: 'restart-manager'
  resource: string
  role: 'other'
}

const execFileAsync = promisify(execFile)

export type RunPowerShell = (
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

function powershellExecutable(): string {
  const windowsRoot = process.env.SystemRoot || 'C:\\Windows'

  return path.join(windowsRoot, 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
}

async function defaultRunPowerShell(
  script: string,
  timeoutMs = 4_000
): Promise<{ stdout: string; stderr: string; code: number }> {
  const budget = Math.max(1, Math.trunc(timeoutMs))

  try {
    const { stdout, stderr } = await execFileAsync(
      powershellExecutable(),
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

function escapePsSingleQuoted(value: string): string {
  return `'${String(value).replace(/'/g, "''")}'`
}

/**
 * Emitted PowerShell must split RM rows on a literal pipe.
 * Prefer String.Split over -split regex so TS template escaping cannot
 * accidentally emit a character-class / alternation pattern.
 */
export const RESTART_MANAGER_ROW_SPLIT_EXPRESSION = "$part.Split([char]'|', 4)"

/**
 * P/Invoke surface for Restart Manager. Compiled once per source revision into
 * a cached assembly so later PowerShell processes load it in milliseconds
 * instead of paying `Add-Type -TypeDefinition` compilation.
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

export function restartManagerNativeSourceHash(): string {
  return crypto.createHash('sha1').update(RESTART_MANAGER_NATIVE_SOURCE).digest('hex').slice(0, 16)
}

export function defaultRestartManagerCacheDir(): string {
  return path.join(os.tmpdir(), 'hermes-restart-manager')
}

function withTrailingSeparator(target: string): string {
  return target.endsWith('\\') || target.endsWith('/') ? target : `${target}${path.sep}`
}

export function buildRestartManagerScript(
  resourceListPath: string,
  {
    perFileLimit = RESTART_MANAGER_PER_FILE_LIMIT,
    cacheDir = defaultRestartManagerCacheDir(),
    attributionRoot = ''
  }: { perFileLimit?: number; cacheDir?: string; attributionRoot?: string } = {}
): string {
  const assemblyPath = path.join(cacheDir, `HermesRm-${restartManagerNativeSourceHash()}.dll`)
  const attribution = attributionRoot ? withTrailingSeparator(path.resolve(attributionRoot)) : ''

  return `
$ErrorActionPreference = 'Stop'
$definite = @()
$ambiguous = @()
foreach ($line in [System.IO.File]::ReadAllLines(${escapePsSingleQuoted(resourceListPath)}, [System.Text.Encoding]::UTF8)) {
  if (-not $line) { continue }
  $tab = $line.IndexOf([char]9)
  if ($tab -lt 1) { continue }
  $flag = $line.Substring(0, $tab)
  $target = $line.Substring($tab + 1)
  if (-not $target) { continue }
  if ($flag -eq 'A') { $ambiguous += $target } else { $definite += $target }
}
if (($definite.Count + $ambiguous.Count) -eq 0) { Write-Output '[]'; exit 0 }
$attributionRoot = ${escapePsSingleQuoted(attribution)}
$rmSource = @"
${RESTART_MANAGER_NATIVE_SOURCE}
"@
$rmAssembly = ${escapePsSingleQuoted(assemblyPath)}
$rmLoaded = $false
if (Test-Path -LiteralPath $rmAssembly) {
  try { Add-Type -Path $rmAssembly -ErrorAction Stop; $rmLoaded = $true } catch { $rmLoaded = $false }
}
if (-not $rmLoaded) {
  $compiled = $false
  try {
    $rmDir = Split-Path -Parent $rmAssembly
    if (-not (Test-Path -LiteralPath $rmDir)) { New-Item -ItemType Directory -Path $rmDir -Force | Out-Null }
    $rmTemp = "$rmAssembly.$([System.Diagnostics.Process]::GetCurrentProcess().Id).tmp"
    Add-Type -TypeDefinition $rmSource -OutputAssembly $rmTemp -ErrorAction Stop
    try { Move-Item -LiteralPath $rmTemp -Destination $rmAssembly -Force -ErrorAction Stop } catch { Remove-Item -LiteralPath $rmTemp -Force -ErrorAction SilentlyContinue }
    Add-Type -Path $rmAssembly -ErrorAction Stop
    $compiled = $true
  } catch { $compiled = $false }
  if (-not $compiled) { Add-Type -TypeDefinition $rmSource }
}
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
    $rows += [pscustomobject]@{ pid = $pidVal; createdAt = $created; name = $name; resource = $res }
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
if ($ambiguous.Count -gt 0 -and $attributionRoot) {
  try { $raw = [HermesRm]::Query([string[]]$ambiguous, [string]$ambiguous[0]) } catch { $raw = '' }
  foreach ($row in @(Convert-RmRows $raw ([string]$ambiguous[0]))) {
    $mapped = $null
    try {
      $proc = [System.Diagnostics.Process]::GetProcessById([int]$row.pid)
      foreach ($module in $proc.Modules) {
        if ($module.FileName.StartsWith($attributionRoot, [StringComparison]::OrdinalIgnoreCase)) { $mapped = $module.FileName; break }
      }
    } catch { $mapped = $null }
    if (-not $mapped) { continue }
    $row.resource = $mapped
    $items += $row
  }
}
$items | ConvertTo-Json -Compress -Depth 3
`.trim()
}

export function parseRestartManagerOutput(
  stdout: string,
  resources: readonly string[]
): RestartManagerHolder[] {
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
  const holders: RestartManagerHolder[] = []
  const seen = new Set<string>()

  for (const row of rows) {
    const pid = Number(row?.pid)
    const createdAt = Number(row?.createdAt)
    const name = typeof row?.name === 'string' && row.name ? row.name : 'unknown'

    const resource =
      typeof row?.resource === 'string' && row.resource
        ? row.resource
        : fallbackResource

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
      resource,
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
    run = defaultRunPowerShell,
    timeoutMs = RESTART_MANAGER_DEFAULT_TIMEOUT_MS,
    listDir,
    cacheDir,
    shared = [],
    attributionRoot
  }: {
    platform?: NodeJS.Platform
    run?: RunPowerShell
    timeoutMs?: number
    listDir?: string
    cacheDir?: string
    /** Locked files that other hard links share; holders are kept only when they map a path under `attributionRoot`. */
    shared?: readonly string[]
    attributionRoot?: string
  } = {}
): Promise<RestartManagerHolder[]> {
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
      ...(cacheDir ? { cacheDir } : {}),
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
