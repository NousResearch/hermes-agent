// Main-process copy the redesigned shell rewords (Gateway → Machines). Classic keeps
// `main`'s text verbatim; the redesign gets its own. The renderer's catalog does this for its
// strings (src/i18n/redesign); these messages are built in main, so they follow the same
// shell choice from the machine's shared UI preferences (window-chrome.ts), adopted where
// main reads and re-applies it. Messages resolve at use time, so a toggle needs no restart.

import { portalShellPreferred } from './window-chrome'

let portalShell = portalShellPreferred(null)

/** Adopt the shell choice from the shared UI preferences; returns whether it is the redesign. */
export function adoptShellPreference(values: Readonly<Record<string, string>> | null | undefined): boolean {
  portalShell = portalShellPreferred(values)

  return portalShell
}

const pick = (classic: string, redesign: string) => (portalShell ? redesign : classic)

export const shellMessages = {
  remoteSessionExpired: () =>
    pick(
      'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.',
      'Your remote gateway session has expired. Open Settings → Machines and click "Sign in" again.'
    ),
  remoteUnsignedOauth: () =>
    'Remote Hermes gateway uses OAuth, but you are not signed in. ' +
    pick(
      'Open Settings → Gateway and click "Sign in", or switch back to Local.',
      'Open Settings → Machines and click "Sign in", or switch back to Local.'
    ),
  switchToLocalMode: () =>
    pick('or switch to Local mode in Settings → Gateway. ', 'or switch to Local mode in Settings → Machines. '),
  ticketSessionRejected: () =>
    'Reached the gateway over HTTP, but the OAuth session was rejected while minting a WebSocket ticket. ' +
    pick('Open Settings → Gateway and sign in again.', 'Open Settings → Machines and sign in again.'),
  remoteMachineLabel: () => pick('Remote gateway', 'Remote machine'),
  plainTextStoragePrompt: () =>
    pick(
      'confirm the plain-text storage option when prompted in Settings → Gateway, ',
      'confirm the plain-text storage option when prompted in Settings → Machines, '
    ),
  cloudNotSignedIn: () =>
    pick(
      'You are not signed in to Hermes Cloud. Open Settings → Gateway, choose Hermes Cloud, and sign in.',
      'You are not signed in to Hermes Cloud. Open Settings → Machines, choose Hermes Cloud, and sign in.'
    ),
  cloudSessionExpired: () =>
    pick(
      'Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.',
      'Your Hermes Cloud session has expired. Open Settings → Machines and sign in again.'
    ),
  remoteTokenMissing: () =>
    pick(
      'Remote Hermes gateway is selected, but no session token is saved. ' +
        'Open Settings → Gateway and save a token, or switch back to Local.',
      'A remote Hermes machine is selected, but no session token is saved. ' +
        'Open Settings → Machines and save a token, or switch back to Local.'
    ),
  nativeSignInTimedOut: () =>
    'Native sign-in timed out. The browser window may not have completed ' +
    pick('sign-in; open Settings → Gateway and try again.', 'sign-in; open Settings → Machines and try again.'),
  oauthWindowTitle: () => pick('Sign in to Hermes gateway', 'Sign in to Hermes')
}
