import { execFile } from 'node:child_process'
import { stat } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { encodePowerShellCommand } from './wsl-clipboard-image'

export interface ClipboardFilePath {
  path: string
  isDirectory: boolean
}

export interface ClipboardFilePathsResult {
  status: 'files' | 'empty' | 'unsupported' | 'failed'
  files: ClipboardFilePath[]
}

interface ClipboardFilesDependencies {
  platform?: string
  exec?: (command: string, args: string[], options: { encoding: 'utf8'; windowsHide: boolean; timeout: number }) => Promise<{ stdout: string }>
  clipboard?: { readBuffer: (format: string) => Buffer }
  isDirectory?: (path: string) => Promise<boolean>
  candidates?: string[]
}

// PowerShell STA call to Clipboard.GetFileDropList() — the real OS file-list
// clipboard format (CF_HDROP). Produces JSON [{Path, IsDirectory}].
const PS_SCRIPT = [
  "$ErrorActionPreference = 'Stop'",
  '[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding',
  'Add-Type -AssemblyName System.Windows.Forms',
  '$paths = [System.Windows.Forms.Clipboard]::GetFileDropList()',
  'if ($paths.Count -eq 0) { exit 0 }',
  '$files = @($paths | ForEach-Object { [pscustomobject]@{ Path = $_; IsDirectory = [System.IO.Directory]::Exists($_) } })',
  'ConvertTo-Json -InputObject $files -Compress'
].join('\n')

// AppleScript enum of the Finder file-list pasteboard type. Returns POSIX paths
// newline-separated.
const APPLE_SCRIPT = [
  'set clipboardFiles to the clipboard as ?class furl?',
  'set output to ""',
  'repeat with clipboardFile in clipboardFiles as list',
  'set output to output & POSIX path of clipboardFile & linefeed',
  'end repeat',
  'return output'
].join('\n')

export async function readClipboardFilePaths({
  platform = process.platform,
  exec = promisify(execFile),
  clipboard,
  isDirectory = async path => (await stat(path)).isDirectory(),
  candidates = ['powershell.exe', join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')]
}: ClipboardFilesDependencies = {}): Promise<ClipboardFilePathsResult> {
  const options = { encoding: 'utf8' as const, windowsHide: true, timeout: 8000 }
  const result = (files: ClipboardFilePath[]): ClipboardFilePathsResult => ({ status: files.length ? 'files' : 'empty', files })

  if (platform === 'win32') {
    for (const candidate of candidates) {
      try {
        const { stdout } = await exec(candidate, ['-NoProfile', '-NonInteractive', '-STA', '-EncodedCommand', encodePowerShellCommand(PS_SCRIPT)], options)

        if (!stdout.trim()) {
          return result([])
        }

        const parsed: unknown = JSON.parse(stdout)
        const entries = Array.isArray(parsed) ? parsed : [parsed]

        if (!entries.every(entry => entry && typeof entry.Path === 'string' && entry.Path.length > 0 && typeof entry.IsDirectory === 'boolean')) {
          return { status: 'failed', files: [] }
        }

        return result(entries.map(entry => ({ path: entry.Path, isDirectory: entry.IsDirectory })))
      } catch (error) {
        // Only a missing executable warrants trying another installation.
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
          return { status: 'failed', files: [] }
        }
      }
    }

    return { status: 'unsupported', files: [] }
  }

  if (platform !== 'darwin' && platform !== 'linux') {
    return { status: 'unsupported', files: [] }
  }

  try {
    let paths: string[]

    if (platform === 'darwin') {
      const { stdout } = await exec('osascript', ['-e', APPLE_SCRIPT], options)
      paths = stdout.split(/\r?\n/).filter(path => path.startsWith('/'))
    } else {
      const nativeClipboard = clipboard ?? (await import('electron')).clipboard
      paths = nativeClipboard.readBuffer('text/uri-list').toString('utf8').split(/\r?\n/)
        .filter(uri => uri.startsWith('file://')).flatMap(uri => {
          // fileURLToPath on Windows rejects POSIX-shaped paths even when the
          // string is well-formed; a single bad URI must not abort the batch.
          try {
            return [fileURLToPath(uri)]
          } catch {
            return []
          }
        })
    }

    return result(await Promise.all(paths.map(async path => ({ path, isDirectory: await isDirectory(path) }))))
  } catch {
    return { status: platform === 'darwin' ? 'unsupported' : 'failed', files: [] }
  }
}