/**
 * state.ts — live connection target + auth status.
 *
 * The active gateway target (base URL + auth mode) lives here as both a plain
 * singleton (for non-React bridge code like http.ts) and nanostores atoms (for
 * React screens). Persisted to @capacitor/preferences so relaunch restores the
 * last gateway without re-entering the URL.
 */

import { Preferences } from '@capacitor/preferences'
import { atom } from 'nanostores'

import type { AuthMode } from './connection-config'

export interface GatewayTarget {
  baseUrl: string
  authMode: AuthMode
  /** Auth provider name (e.g. "basic") when password-gated; null for token mode. */
  provider: string | null
}

export type AuthStatus =
  | 'unknown' // haven't probed yet
  | 'probing' // GET /api/status in flight
  | 'needs-login' // gated gateway, no live session
  | 'authed' // have a session cookie / token
  | 'error'

export const $target = atom<GatewayTarget | null>(null)
export const $authStatus = atom<AuthStatus>('unknown')
/** Bumped whenever a request 401s in a way that demands re-login. Screens watch
 *  this to bounce back to the login form. */
export const $reauthNonce = atom(0)

let _target: GatewayTarget | null = null

export function currentTarget(): GatewayTarget | null {
  return _target
}

const KEY = 'hermes.target'

export async function loadTarget(): Promise<GatewayTarget | null> {
  const { value } = await Preferences.get({ key: KEY })
  if (value) {
    try {
      _target = JSON.parse(value) as GatewayTarget
      $target.set(_target)
    } catch {
      _target = null
    }
  }
  return _target
}

export async function setTarget(t: GatewayTarget | null): Promise<void> {
  _target = t
  $target.set(t)
  if (t) await Preferences.set({ key: KEY, value: JSON.stringify(t) })
  else await Preferences.remove({ key: KEY })
}

export function setAuthStatus(s: AuthStatus): void {
  $authStatus.set(s)
}

export function requireReauth(): void {
  $authStatus.set('needs-login')
  $reauthNonce.set($reauthNonce.get() + 1)
}
