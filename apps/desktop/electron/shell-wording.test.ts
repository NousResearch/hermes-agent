import { afterEach, describe, expect, test } from 'vitest'

import { makeNousCloudBackendDownError, makeReauthRequiredError, makeUnsignedOauthError } from './backend-health'
import { migrateV1ToRegistry } from './connection-registry'
import { encryptDesktopSecret } from './hardening'
import { oauthTicketFailureAuthMessage } from './native-auth-decisions'
import { adoptShellPreference, shellMessages } from './shell-wording'
import { SHELL_CHOICE_KEY } from './window-chrome'

const CLASSIC = { [SHELL_CHOICE_KEY]: 'false' }
const REDESIGN = { [SHELL_CHOICE_KEY]: 'true' }

const noKeyring = { isEncryptionAvailable: () => false }

// DESIGN.md › Terminology: the classic nouns the redesign renames (Sessions → Chats, Artifacts →
// Library, Profiles → Agents, Cron / scheduled jobs → Routines, Gateways → Machines).
const CLASSIC_NOUNS = /\b(artifacts?|cron|gateways?|profiles?|scheduled jobs?)\b/i
const SESSION = /\bsessions?\b/i

/** Messages whose "session" is a sign-in session (or its token), not a conversation. */
const SIGN_IN_SESSION = new Set([
  'remoteSessionExpired',
  'ticketSessionRejected',
  'cloudSessionExpired',
  'remoteTokenMissing'
])

/** Messages that keep a classic noun on purpose, with why. */
const KEEPS: Record<string, string> = {
  // The renderer recognises a reauth failure by this phrase (boot-failure-reauth.ts) and the health
  // contract asserts it (backend-health.test.ts); it changes with both.
  remoteSessionExpired: '"remote gateway session has expired" is matched as a reauth signal'
}

function cloudDownMessage(): string {
  const error = Object.assign(new Error('upstream unavailable'), { statusCode: 503 })

  return makeNousCloudBackendDownError('https://ares-3009.agents.nousresearch.com', error)!.message
}

function plainTextRefusal(): string {
  try {
    encryptDesktopSecret('token', noKeyring)
  } catch (error) {
    return (error as Error).message
  }

  throw new Error('expected the keyring-less save to be refused')
}

afterEach(() => {
  adoptShellPreference(null)
})

test('the redesign wording is on until the user rolls the shell back, and follows a later change', () => {
  expect(adoptShellPreference(null)).toBe(true)
  expect(shellMessages.oauthWindowTitle()).toBe('Sign in to Hermes')

  expect(adoptShellPreference(CLASSIC)).toBe(false)
  expect(shellMessages.oauthWindowTitle()).toBe('Sign in to Hermes gateway')

  expect(adoptShellPreference(REDESIGN)).toBe(true)
  expect(shellMessages.oauthWindowTitle()).toBe('Sign in to Hermes')
})

describe('classic keeps the main-branch wording', () => {
  test('in the auth errors main builds', () => {
    adoptShellPreference(CLASSIC)

    const expired = 'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.'

    const unsigned =
      'Remote Hermes gateway uses OAuth, but you are not signed in. ' +
      'Open Settings → Gateway and click "Sign in", or switch back to Local.'

    expect(makeReauthRequiredError().message).toBe(expired)
    expect(makeUnsignedOauthError().message).toBe(unsigned)
    expect(oauthTicketFailureAuthMessage(true)).toBe(expired)
    expect(oauthTicketFailureAuthMessage(false)).toBe(unsigned)
    expect(cloudDownMessage()).toContain('or switch to Local mode in Settings → Gateway. ')
    expect(plainTextRefusal()).toContain('confirm the plain-text storage option when prompted in Settings → Gateway, ')
  })

  test('in every other main-process message', () => {
    adoptShellPreference(CLASSIC)

    expect(Object.fromEntries(Object.entries(shellMessages).map(([key, message]) => [key, message()]))).toEqual({
      cloudNotSignedIn:
        'You are not signed in to Hermes Cloud. Open Settings → Gateway, choose Hermes Cloud, and sign in.',
      cloudSessionExpired: 'Your Hermes Cloud session has expired. Open Settings → Gateway and sign in again.',
      nativeSignInTimedOut:
        'Native sign-in timed out. The browser window may not have completed sign-in; open Settings → Gateway and try again.',
      oauthWindowTitle: 'Sign in to Hermes gateway',
      plainTextStoragePrompt: 'confirm the plain-text storage option when prompted in Settings → Gateway, ',
      remoteSessionExpired:
        'Your remote gateway session has expired. Open Settings → Gateway and click "Sign in" again.',
      remoteTokenMissing:
        'Remote Hermes gateway is selected, but no session token is saved. ' +
        'Open Settings → Gateway and save a token, or switch back to Local.',
      remoteUnsignedOauth:
        'Remote Hermes gateway uses OAuth, but you are not signed in. ' +
        'Open Settings → Gateway and click "Sign in", or switch back to Local.',
      switchToLocalMode: 'or switch to Local mode in Settings → Gateway. ',
      ticketSessionRejected:
        'Reached the gateway over HTTP, but the OAuth session was rejected while minting a WebSocket ticket. ' +
        'Open Settings → Gateway and sign in again.'
    })
  })
})

describe('the redesign points at Settings → Machines', () => {
  test('in the auth errors main builds', () => {
    adoptShellPreference(REDESIGN)

    expect(makeReauthRequiredError().message).toBe(
      'Your remote gateway session has expired. Open Settings → Machines and click "Sign in" again.'
    )
    expect(oauthTicketFailureAuthMessage(false)).toBe(
      'The remote Hermes machine uses OAuth, but you are not signed in. ' +
        'Open Settings → Machines and click "Sign in", or switch back to Local.'
    )
    expect(cloudDownMessage()).toContain('or switch to Local mode in Settings → Machines. ')
    expect(plainTextRefusal()).toContain('confirm the plain-text storage option when prompted in Settings → Machines, ')
  })

  test('in every other main-process message', () => {
    adoptShellPreference(REDESIGN)

    expect(shellMessages.remoteTokenMissing()).toBe(
      'A remote Hermes machine is selected, but no session token is saved. ' +
        'Open Settings → Machines and save a token, or switch back to Local.'
    )
    expect(shellMessages.oauthWindowTitle()).toBe('Sign in to Hermes')
    expect(shellMessages.cloudNotSignedIn()).toContain('Open Settings → Machines, choose Hermes Cloud')
  })

  test('with no classic noun in any message, beyond a sign-in session', () => {
    adoptShellPreference(REDESIGN)

    const offenders = Object.entries(shellMessages)
      .map(([key, message]) => [key, message()] as const)
      .filter(([key]) => !(key in KEEPS))
      .filter(([key, text]) => CLASSIC_NOUNS.test(text) || (!SIGN_IN_SESSION.has(key) && SESSION.test(text)))

    expect(offenders).toEqual([])
  })

  test('whose keeps are real messages that still carry what they keep', () => {
    adoptShellPreference(REDESIGN)

    const messages: Record<string, () => string> = shellMessages

    for (const key of SIGN_IN_SESSION) {
      expect(messages[key]?.()).toMatch(SESSION)
    }

    for (const key of Object.keys(KEEPS)) {
      expect(messages[key]?.()).toMatch(CLASSIC_NOUNS)
    }
  })
})

test("persisted data never takes shell wording: a host-less remote stores main's default label", () => {
  for (const shell of [CLASSIC, REDESIGN]) {
    adoptShellPreference(shell)

    const registry = migrateV1ToRegistry({ mode: 'remote', remote: { url: 'not a url', authMode: 'token' } })

    expect(registry.connections.find(connection => connection.kind === 'remote')?.label).toBe('Remote gateway')
  }
})
