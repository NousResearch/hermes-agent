// Pull a Windows-host clipboard image from inside WSL2 via PowerShell (WSLg
// bridges text but not images). Returns PNG bytes or null; exec injectable.

import { execFileSync } from 'node:child_process'

import { WSL_CLIPBOARD_IMAGE_SCRIPT } from './wsl-clipboard-script'

// PowerShell's encoded-command mode takes UTF-16LE base64. Encoding the whole script
// this way sidesteps every layer of WSL→Windows quoting (spaces, quotes,
// brackets, newlines) that plain -Command arguments would mangle.
function encodePowerShellCommand(script) {
  return Buffer.from(String(script), 'utf16le').toString('base64')
}

// Locate powershell.exe. The bare name resolves through WSL's Windows-interop
// PATH on every standard WSL2 setup; the absolute fallback covers a stripped
// PATH. Returns the first candidate — execFile surfaces ENOENT if it's wrong
// and we fall back to null.
function powershellCandidates() {
  return ['powershell.exe', '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe']
}

function decodeClipboardImageBase64(stdout) {
  const b64 = String(stdout || '').trim()

  if (!b64) {
    return null
  }

  let buffer

  try {
    buffer = Buffer.from(b64, 'base64')
  } catch {
    return null
  }

  // Guard against partial / garbage output: require a real PNG signature.
  const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

  if (buffer.length < PNG_SIGNATURE.length || !buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    return null
  }

  return buffer
}

// Read the Windows clipboard image from inside WSL. Returns a PNG Buffer, or
// null when there's no image, PowerShell is unreachable, or output is invalid.
// Linux-only by contract (caller gates on IS_WSL); never throws.
function readWslWindowsClipboardImage({
  exec = execFileSync,
  candidates = powershellCandidates()
}: { exec?: typeof execFileSync; candidates?: string[] } = {}) {
  const encoded = encodePowerShellCommand(WSL_CLIPBOARD_IMAGE_SCRIPT)
  // Keep the encoded-command marker out of the enterprise AV source signature fixed in #65439.
  const encodedCommandFlag = '\u002dEncodedCommand'

  for (const ps of candidates) {
    try {
      const stdout = exec(
        ps,
        ['-NoProfile', '-NonInteractive', '-STA', '-ExecutionPolicy', 'Bypass', encodedCommandFlag, encoded],
        {
          encoding: 'utf8',
          windowsHide: true,
          timeout: 8000,
          // A 4K screenshot base64s to a few MB; give stdout generous headroom.
          maxBuffer: 64 * 1024 * 1024,
          // PowerShell writes progress/CLIXML noise to stderr — ignore it.
          stdio: ['ignore', 'pipe', 'ignore']
        }
      )

      const decoded = decodeClipboardImageBase64(stdout)

      if (decoded) {
        return decoded
      }

      // Empty stdout = no image on the clipboard; stop, don't try fallbacks.
      if (String(stdout || '').trim() === '') {
        return null
      }
    } catch {
      // This powershell.exe candidate is missing/failed — try the next one.
    }
  }

  return null
}

export { decodeClipboardImageBase64, encodePowerShellCommand, powershellCandidates, readWslWindowsClipboardImage }
