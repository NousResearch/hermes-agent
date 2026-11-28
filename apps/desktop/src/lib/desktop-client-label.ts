import { isBrowserHostedDesktop } from './platform'

const KEY = 'hermes.desktop.clientLabel.v1'
export const LABEL_MAX_LENGTH = 48

// No machine-derived default. Re-read on send so another window can revoke sharing.
export function validDesktopClientLabel(label: string): boolean {
  return label.length <= LABEL_MAX_LENGTH && /^[A-Za-z0-9]/.test(label) && !/[^A-Za-z0-9 _-]/.test(label)
}

export function readDesktopClientLabel(): string {
  if (isBrowserHostedDesktop()) { return '' }
  try {
    const value = localStorage.getItem(KEY) ?? ''
    return validDesktopClientLabel(value) ? value : ''
  } catch { return '' }
}

export function saveDesktopClientLabel(value: string): void {
  if (!value) { localStorage.removeItem(KEY) }
  else if (validDesktopClientLabel(value)) { localStorage.setItem(KEY, value) }
}
