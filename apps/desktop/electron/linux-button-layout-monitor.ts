import { execFile, spawn } from 'node:child_process'

import { type LinuxWindowButtons, parseLinuxButtonLayout } from './linux-button-layout'

const GSETTINGS = 'gsettings'
const SCHEMA = 'org.gnome.desktop.wm.preferences'
const KEY = 'button-layout'
// gsettings is local and should answer immediately; don't hold startup on a
// wedged D-Bus session.
const READ_TIMEOUT_MS = 1500

function monitorValue(line: string): string {
  // `gsettings monitor` prints "<key>: <value>" (some versions use "=");
  // accept a raw value too so the parser sees the same shape as `get`.
  for (const separator of [':', '=']) {
    const marker = `${KEY}${separator}`
    const at = line.indexOf(marker)

    if (at >= 0) {
      return line.slice(at + marker.length).trim()
    }
  }

  return line
}

export type LinuxButtonLayoutListener = (layout: LinuxWindowButtons) => void

/** Read the current GNOME button layout. Failures leave the caller's default in place. */
export function readLinuxButtonLayout(onLayout: LinuxButtonLayoutListener): void {
  execFile(GSETTINGS, ['get', SCHEMA, KEY], { timeout: READ_TIMEOUT_MS, windowsHide: true }, (error, stdout) => {
    if (error) {
      return
    }

    onLayout(parseLinuxButtonLayout(stdout))
  })
}

/**
 * Watch GNOME's button layout. Returns a stop function; a missing or failing
 * gsettings process leaves the startup value in place.
 */
export function watchLinuxButtonLayout(onLayout: LinuxButtonLayoutListener): () => void {
  let child

  try {
    child = spawn(GSETTINGS, ['monitor', SCHEMA, KEY], {
      stdio: ['ignore', 'pipe', 'ignore'],
      windowsHide: true
    })
  } catch {
    return () => undefined
  }

  let pending = ''

  child.stdout?.setEncoding('utf8')
  child.stdout?.on('data', (chunk: string) => {
    pending += chunk

    for (let newline = pending.indexOf('\n'); newline >= 0; newline = pending.indexOf('\n')) {
      const line = pending.slice(0, newline)
      pending = pending.slice(newline + 1)
      onLayout(parseLinuxButtonLayout(monitorValue(line)))
    }
  })
  child.on('error', () => undefined)

  return () => {
    child.kill()
  }
}
