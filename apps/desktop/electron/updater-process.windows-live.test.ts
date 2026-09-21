import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { buildSync } from 'esbuild'
import { test } from 'vitest'

const run = promisify(execFile)

// Harmless stand-in for windows.ps1: no update, installer, or Hermes runtime.
// It records actual parameter binding and the native interactive-console state.
const fixture = String.raw`
param([string]$InstallRoot, [string]$Branch, [int]$DesktopPid, [string]$RelaunchExe, [switch]$NoGateway)
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class UpdaterConsoleProbe {
  [DllImport("kernel32.dll")] public static extern IntPtr GetConsoleWindow();
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hWnd);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
  public static extern IntPtr CreateFile(string n, uint a, uint s, IntPtr p, uint d, uint f, IntPtr t);
  [DllImport("kernel32.dll")] public static extern bool GetConsoleMode(IntPtr h, out uint mode);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
}
'@
$window = [UpdaterConsoleProbe]::GetConsoleWindow()
$inputHandle = [UpdaterConsoleProbe]::CreateFile('CONIN$', [uint32]2147483648, 3, [IntPtr]::Zero, 3, 0, [IntPtr]::Zero)
$mode = [uint32]0
$interactive = [UpdaterConsoleProbe]::GetConsoleMode($inputHandle, [ref]$mode)
[void][UpdaterConsoleProbe]::CloseHandle($inputHandle)
$deadline = [DateTime]::UtcNow.AddSeconds(10)
while ((Get-Process -Id $env:HERMES_UPDATER_TEST_PARENT -ErrorAction SilentlyContinue) -and [DateTime]::UtcNow -lt $deadline) {
  Start-Sleep -Milliseconds 50
}
$result = @{
  InstallRoot = $InstallRoot; Branch = $Branch; DesktopPid = $DesktopPid; RelaunchExe = $RelaunchExe
  NoGateway = $NoGateway.IsPresent
  Console = ($window -ne [IntPtr]::Zero); Visible = [UpdaterConsoleProbe]::IsWindowVisible($window)
  Interactive = $interactive; ParentGone = !(Get-Process -Id $env:HERMES_UPDATER_TEST_PARENT -ErrorAction SilentlyContinue)
}
[IO.File]::WriteAllText(($env:HERMES_UPDATER_TEST_RESULT + '.tmp'), ($result | ConvertTo-Json -Compress))
[IO.File]::Move(($env:HERMES_UPDATER_TEST_RESULT + '.tmp'), $env:HERMES_UPDATER_TEST_RESULT)
`

test.skipIf(process.platform !== 'win32').each([
  { launchMode: 'control', noGateway: false },
  { launchMode: 'control', noGateway: true },
  { launchMode: 'adversarial', noGateway: false },
  { launchMode: 'adversarial', noGateway: true }
])(
  'Windows updater $launchMode (NoGateway=$noGateway) preserves data, stays hidden, and survives its parent exiting',
  async ({ launchMode, noGateway }) => {
    const temporary = mkdtempSync(path.join(os.tmpdir(), 'hermes-updater-test-'))

    try {
      const root = path.join(
        temporary,
        launchMode === 'adversarial'
          ? "space & (paren) %HERMES_UPDATER_TEST_EXPAND% !bang! 'quote' `tick` عربي"
          : 'ordinary'
      )

      const scriptDir = path.join(root, 'scripts', 'desktop-update')
      mkdirSync(scriptDir, { recursive: true })
      writeFileSync(path.join(scriptDir, 'windows.ps1'), fixture, 'utf8')

      const branch =
        launchMode === 'adversarial'
          ? '-feature/"quoted"%HERMES_UPDATER_TEST_EXPAND%&|!bang!($value);`tick`‘quote’'
          : 'main'

      const relaunchExe = path.join(root, 'Hermes.exe')
      const resultFile = path.join(temporary, 'result.json')
      const installerResult = path.join(temporary, 'installer.json')
      // The stand-in executable is a temporary copy of Node, not the real
      // installer. Exercise an adversarial executable path without admin rights.
      const installer = path.join(root, 'hermes-setup.exe')
      copyFileSync(process.execPath, installer)
      const installerArgs = ['--update', '--branch', branch, '--install-root', root]

      const recordInstallerArgs = `
        const fs = require('node:fs');
        const output = process.env.HERMES_UPDATER_TEST_EXE_RESULT;
        fs.writeFileSync(output + '.tmp', JSON.stringify(process.argv.slice(1)));
        fs.renameSync(output + '.tmp', output);
      `

      const launcher = path.join(temporary, 'parent.cjs')
      buildSync({
        stdin: {
          contents: `
            import { resolveUpdateScriptHandoff, wrapHandoffForDetachedConsole, spawnUpdaterProcess } from './updater-process';
            const handoff = resolveUpdateScriptHandoff(${JSON.stringify(root)});
            const extra = ${JSON.stringify(['-InstallRoot', root, '-Branch', branch, '-DesktopPid', '42', '-RelaunchExe', relaunchExe, ...(noGateway ? ['-NoGateway'] : [])])};
            const options = { detached: true, stdio: 'ignore', env: { ...process.env, HERMES_UPDATER_TEST_PARENT: String(process.pid) } };
            const wrapped = wrapHandoffForDetachedConsole(handoff, extra);
            spawnUpdaterProcess(wrapped.command, wrapped.args, { ...options, detached: wrapped.detached });
            spawnUpdaterProcess(${JSON.stringify(installer)}, ${JSON.stringify(['-e', recordInstallerArgs, '--', ...installerArgs])}, options);
          `,
          resolveDir: fileURLToPath(new URL('.', import.meta.url)),
          loader: 'ts'
        },
        bundle: true,
        platform: 'node',
        format: 'cjs',
        outfile: launcher
      })
      await run(process.execPath, [launcher], {
        windowsHide: true,
        timeout: 15_000,
        env: {
          ...process.env,
          HERMES_UPDATER_TEST_RESULT: resultFile,
          HERMES_UPDATER_TEST_EXE_RESULT: installerResult,
          HERMES_UPDATER_TEST_EXPAND: 'EXPANDED'
        }
      })
      // Cold Windows PowerShell startup plus Add-Type can exceed 15s while
      // other CI/build workers are active; keep the native probe bounded.
      const deadline = Date.now() + 30_000

      while ((!existsSync(resultFile) || !existsSync(installerResult)) && Date.now() < deadline) {
        await new Promise(resolve => setTimeout(resolve, 100))
      }

      assert.ok(existsSync(resultFile), 'the detached PowerShell script must execute')
      const result = JSON.parse(readFileSync(resultFile, 'utf8'))
      assert.deepEqual(
        {
          InstallRoot: result.InstallRoot,
          Branch: result.Branch,
          DesktopPid: result.DesktopPid,
          RelaunchExe: result.RelaunchExe,
          NoGateway: result.NoGateway,
          Visible: result.Visible,
          ParentGone: result.ParentGone
        },
        {
          InstallRoot: root,
          Branch: branch,
          DesktopPid: 42,
          RelaunchExe: relaunchExe,
          NoGateway: noGateway,
          Visible: false,
          ParentGone: true
        }
      )
      assert.deepEqual(JSON.parse(readFileSync(installerResult, 'utf8')), installerArgs)
    } finally {
      rmSync(temporary, { recursive: true, force: true })
    }
  },
  60_000
)
