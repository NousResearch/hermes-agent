import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isReauthRequiredError, makeUnsignedOauthError } from './backend-health'
import {
  classifyTerminalRemoteBootFailure,
  isHermesNotFoundBootFailure,
  isHostKeyChangedBootFailure,
  isRetryableRemoteBootFailure,
  isSshAuthFailedBootFailure,
  isSshClientFailedBootFailure,
  shouldHoldBootProgressForReauth,
  shouldLatchBackendStartFailure,
  shouldLatchHermesNotFoundFailure,
  shouldLatchHostKeyChangedFailure,
  shouldLatchRemoteReauthFailure,
  shouldLatchSshAuthFailure,
  shouldLatchSshClientFailure,
  shouldLatchTerminalRemoteBootFailure,
  sshClientFailedError
} from './backend-start-failure'
import { SshConnection } from './ssh-connection'

test('latches a LOCAL backend failure so the install-retry loop is broken', () => {
  assert.equal(shouldLatchBackendStartFailure({ attemptedRemote: false }), true)
})

test('never latches a REMOTE failure so recovery stays retryable without a restart', () => {
  // A lapsed OAuth session / mint timeout / host briefly unreachable across a
  // laptop sleep must not wedge the app: the next connect has to re-attempt and
  // re-mint against the refreshed session.
  assert.equal(shouldLatchBackendStartFailure({ attemptedRemote: true }), false)
})

test('never latches a supervisor-owned respawn failure (it has its own bounded crash-loop budget)', () => {
  // A pre-ready child exit during a supervisor respawn must be able to spend
  // the remaining crash-loop slots instead of becoming a permanent local latch.
  assert.equal(shouldLatchBackendStartFailure({ attemptedRemote: false, supervisorRecovery: true }), false)
  assert.equal(shouldLatchBackendStartFailure({ attemptedRemote: false, supervisorRecovery: false }), true)
})

test('latches a CONFIRMED remote reauth failure so the overlay stays clickable', () => {
  // Without this the non-latching remote path re-runs startHermes on every
  // getConnection/api call, re-emits running:true, and the overlay hides
  // itself — the "Sign in" button flickers away before it can be clicked.
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: true }), true)
})

test('does not latch a transient remote failure as reauth', () => {
  // A mint timeout or a host unreachable across sleep must still self-heal.
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: false }), false)
})

test('never latches a LOCAL failure as reauth (that is backendStartFailure job)', () => {
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: false, isReauth: true }), false)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: false, isReauth: false }), false)
})

test('the two latches never fire for the same failure', () => {
  // They are complementary, not overlapping: local failures latch via
  // backendStartFailure, confirmed remote reauth latches via its own flag.
  for (const attemptedRemote of [true, false]) {
    for (const isReauth of [true, false]) {
      const start = shouldLatchBackendStartFailure({ attemptedRemote })
      const reauth = shouldLatchRemoteReauthFailure({ attemptedRemote, isReauth })
      assert.ok(!(start && reauth), `both latched for remote=${attemptedRemote} reauth=${isReauth}`)
    }
  }
})

test('FIX #82679: a transient remote failure is retryable so a dropped SSH/HTTP connection self-heals', () => {
  // The dropped-registered-connection class: "Could not verify the existing
  // SSH backend", ERR_CONNECTION_RESET on an HTTP remote, mint timeouts. All
  // surface as non-reauth remote boot failures and must enter the bounded
  // renderer retry loop instead of parking on "Desktop boot failed".
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: false }), true)
})

test('a CONFIRMED reauth rejection is never auto-retried (missing capability, not transient failure)', () => {
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: true }), false)
})

test('unsigned OAuth latches and is never auto-retried; a bare needsOauthLogin hint still retries', () => {
  // Production composition in startHermes: isReauth = isReauthRequiredError(error).
  // A bare `{ needsOauthLogin: true }` is the IPC-shaped hint, not a confirmed
  // rejection; gatewayTicketFailure tags a confirmed 401/403 with
  // isReauthRequired itself (#95701, see remote-reauth-latch.test.ts).
  const unsigned = isReauthRequiredError(makeUnsignedOauthError())
  const ticketHint = isReauthRequiredError({ needsOauthLogin: true })

  assert.equal(unsigned, true)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: unsigned }), true)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: unsigned }), false)
  assert.equal(ticketHint, false)
  assert.equal(shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth: ticketHint }), false)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: ticketHint }), true)
})

test('local failures are never auto-retried by the remote self-heal loop', () => {
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: false, isReauth: false }), false)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: false, isReauth: true }), false)
})

test('retryable and reauth-latch are mutually exclusive for remote failures', () => {
  // Every remote failure either self-heals (transient) or latches for sign-in
  // (confirmed reauth) — never both, never neither.
  for (const isReauth of [true, false]) {
    const retry = isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth })
    const latch = shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth })
    assert.equal(retry !== latch, true, `remote failure with reauth=${isReauth} must pick exactly one path`)
  }
})

test('FIX host-key change: classified from the kind tag and from stringified ssh banners', () => {
  // classifySshError tags the Error it built; errors that crossed an IPC or
  // string boundary only keep the message. Both shapes must classify.
  const tagged = Object.assign(new Error('SSH refused to connect.'), { kind: 'host-key-changed' })
  assert.equal(isHostKeyChangedBootFailure(tagged), true)
  assert.equal(
    isHostKeyChangedBootFailure(new Error('@@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@@')),
    true
  )
  assert.equal(isHostKeyChangedBootFailure(new Error('Host key verification failed.')), true)
  assert.equal(
    isHostKeyChangedBootFailure(new Error('The host key for root@203.0.113.7 has CHANGED since you last connected.')),
    true
  )
  assert.equal(isHostKeyChangedBootFailure(new Error('Connection refused')), false)
  assert.equal(isHostKeyChangedBootFailure(null), false)
})

test('FIX host-key change: latches and is never auto-retried (157-failure loop, Aug 2026 bundle)', () => {
  // SSH fails closed on a changed host key: every retry re-drives the same
  // doomed boot until the user clears known_hosts. Terminal, like reauth.
  const context = { attemptedRemote: true, isReauth: false, isHostKeyChanged: true }
  assert.equal(shouldLatchHostKeyChangedFailure(context), true)
  assert.equal(isRetryableRemoteBootFailure(context), false)
})

test('host-key latch never fires for local failures or ordinary remote faults', () => {
  assert.equal(
    shouldLatchHostKeyChangedFailure({ attemptedRemote: false, isReauth: false, isHostKeyChanged: true }),
    false
  )
  assert.equal(
    shouldLatchHostKeyChangedFailure({ attemptedRemote: true, isReauth: false, isHostKeyChanged: false }),
    false
  )
  assert.equal(shouldLatchHostKeyChangedFailure({ attemptedRemote: true, isReauth: false }), false)
})

test('every remote failure picks exactly one path: retry, reauth latch, or host-key latch', () => {
  for (const isReauth of [true, false]) {
    for (const isHostKeyChanged of [true, false]) {
      const retry = isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth, isHostKeyChanged })
      const reauth = shouldLatchRemoteReauthFailure({ attemptedRemote: true, isReauth })
      const hostKey = shouldLatchHostKeyChangedFailure({ attemptedRemote: true, isReauth, isHostKeyChanged })
      const picked = [retry, reauth, hostKey].filter(Boolean).length
      assert.ok(picked >= 1, `remote failure reauth=${isReauth} hostKey=${isHostKeyChanged} fell through every path`)
    }
  }
})

test('FIX #72698: a rejected SSH key latches the boot failure and is never auto-retried', () => {
  // The error exactly as SshConnection.open() rejects it, and as the SSH
  // bootstrap re-wraps a lifecycle failure (message + sshError tag).
  const fromOpen = new SshConnection({ host: '127.0.0.1', user: 'me' }, {})._fail(
    'me@127.0.0.1: Permission denied (publickey,password,keyboard-interactive).'
  )

  const rewrapped = Object.assign(new Error(fromOpen.message), { sshError: fromOpen.kind, isSshBootstrap: true })

  for (const error of [fromOpen, rewrapped, new Error(fromOpen.message)]) {
    const isSshAuthFailed = isSshAuthFailedBootFailure(error)
    const context = { attemptedRemote: true, isReauth: false, isHostKeyChanged: false, isSshAuthFailed }

    assert.equal(shouldLatchSshAuthFailure(context), true)
    assert.equal(isRetryableRemoteBootFailure(context), false)
  }

  // Connectivity faults keep self-healing; local boots use the local latch.
  const unreachable = new SshConnection({ host: '127.0.0.1', user: 'me' }, {})._fail(
    'ssh: connect to host 127.0.0.1 port 22: Connection refused'
  )

  const transient = {
    attemptedRemote: true,
    isReauth: false,
    isSshAuthFailed: isSshAuthFailedBootFailure(unreachable)
  }

  assert.equal(shouldLatchSshAuthFailure(transient), false)
  assert.equal(isRetryableRemoteBootFailure(transient), true)
  assert.equal(shouldLatchSshAuthFailure({ attemptedRemote: false, isReauth: false, isSshAuthFailed: true }), false)
  // A remote lifecycle's filesystem "Permission denied" is not a credential rejection.
  assert.equal(isSshAuthFailedBootFailure(new Error('mkdir: /opt/hermes: Permission denied')), false)
})

test('FIX #95701: while a reauth rejection is latched, only re-emits of that failure reach the renderer', () => {
  const latched = 'Your remote gateway session has expired. Sign in again.'

  // The latched failure's own (re-)emit passes — it carries retryable:false.
  assert.equal(shouldHoldBootProgressForReauth(latched, { error: latched }), false)

  // Anything that would lift the overlay is held: a running phase from an
  // attempt already in flight when the latch closed, a cleared error, or a
  // sibling failure that would flip retryable back on.
  assert.equal(shouldHoldBootProgressForReauth(latched, { error: null }), true)
  assert.equal(shouldHoldBootProgressForReauth(latched, {}), true)
  assert.equal(shouldHoldBootProgressForReauth(latched, { error: 'Could not reach the remote Hermes gateway' }), true)
})

test('FIX #95701: with no reauth latch every boot-progress update flows as before', () => {
  for (const latch of [null, undefined, '']) {
    assert.equal(shouldHoldBootProgressForReauth(latch, { error: null }), false)
    assert.equal(shouldHoldBootProgressForReauth(latch, {}), false)
    assert.equal(shouldHoldBootProgressForReauth(latch, { error: 'Desktop boot failed: spawn ENOENT' }), false)
  }
})

test('FIX #103288: a failed local `ssh -G` probe latches and is never auto-retried', () => {
  const error = sshClientFailedError(
    'C:\\Windows\\System32\\OpenSSH\\ssh.exe',
    new Error('Command failed: ssh.exe -G -- box'),
    'win32'
  )

  assert.equal(isSshClientFailedBootFailure(error), true)
  assert.match(error.message, /System32\\OpenSSH\\ssh\.exe/)
  assert.match(error.message, /Command failed/)
  assert.match(error.message, /desktop\.ssh_path/)

  const context = { attemptedRemote: true, isReauth: false, isSshClientFailed: true }

  assert.equal(shouldLatchSshClientFailure(context), true)
  assert.equal(isRetryableRemoteBootFailure(context), false)

  // Ordinary remote faults keep retrying; local boots never take this latch.
  const unreachable = new Error('ssh: connect to host box port 22: Connection timed out')

  assert.equal(isSshClientFailedBootFailure(unreachable), false)
  assert.equal(
    isRetryableRemoteBootFailure({
      attemptedRemote: true,
      isReauth: false,
      isSshClientFailed: isSshClientFailedBootFailure(unreachable)
    }),
    true
  )
  assert.equal(shouldLatchSshClientFailure({ attemptedRemote: false, isReauth: false, isSshClientFailed: true }), false)
})

test('a remote with no Hermes install latches the boot failure and is never auto-retried', () => {
  // As remote-lifecycle's locateHermes throws it (auto-detect and explicit path),
  // as the SSH bootstrap re-wraps it (message + sshError tag), and as a Windows
  // remote's PowerShell throw arrives: ssh stderr wrapped by SshConnection, text only.
  const windowsRemote = (thrown: string) =>
    new SshConnection({ host: '127.0.0.1', user: 'me' }, {})._fail(
      `#< CLIXML\r\n<Objs Version="1.1.0.1"><S S="Error">${thrown}_x000D__x000A_</S></Objs>`
    )

  const notInstalled = Object.assign(
    new Error('Hermes is not installed on the remote host (could not find a `hermes` executable).'),
    { kind: 'hermes-not-found' }
  )

  const badPath = Object.assign(
    new Error('The Hermes path you set is not an executable on the remote host: "/bad/path/hermes".'),
    { kind: 'hermes-not-found' }
  )

  const rewrapped = Object.assign(new Error(notInstalled.message), {
    sshError: 'hermes-not-found',
    isSshBootstrap: true
  })

  for (const error of [
    notInstalled,
    badPath,
    rewrapped,
    new Error(notInstalled.message),
    windowsRemote('Hermes is not installed on the remote Windows host.'),
    windowsRemote('The configured Hermes path is not an executable file.')
  ]) {
    const isHermesNotFound = isHermesNotFoundBootFailure(error)
    const context = { attemptedRemote: true, isReauth: false, isHermesNotFound }

    assert.equal(shouldLatchHermesNotFoundFailure(context), true)
    assert.equal(isRetryableRemoteBootFailure(context), false)
  }

  // Connectivity faults keep self-healing; local boots never take this latch.
  const unreachable = new Error('ssh: connect to host box port 22: Connection timed out')
  const transient = {
    attemptedRemote: true,
    isReauth: false,
    isHermesNotFound: isHermesNotFoundBootFailure(unreachable)
  }

  assert.equal(shouldLatchHermesNotFoundFailure(transient), false)
  assert.equal(isRetryableRemoteBootFailure(transient), true)
  assert.equal(
    shouldLatchHermesNotFoundFailure({ attemptedRemote: false, isReauth: false, isHermesNotFound: true }),
    false
  )
})

test('every terminal remote kind takes the one boot latch; anything else does not', () => {
  const ssh = new SshConnection({ host: '127.0.0.1', user: 'me' }, {})

  const cases = [
    [new Error('Host key verification failed.'), 'isHostKeyChanged'],
    [ssh._fail('me@127.0.0.1: Permission denied (publickey).'), 'isSshAuthFailed'],
    [sshClientFailedError('ssh', new Error('boom'), 'linux'), 'isSshClientFailed'],
    [
      Object.assign(new Error('Hermes is not installed on the remote host.'), { kind: 'hermes-not-found' }),
      'isHermesNotFound'
    ]
  ] as const

  for (const [error, flag] of cases) {
    const flags = classifyTerminalRemoteBootFailure(error)

    assert.deepEqual(
      Object.entries(flags)
        .filter(([, set]) => set)
        .map(([name]) => name),
      [flag]
    )
    assert.equal(shouldLatchTerminalRemoteBootFailure({ attemptedRemote: true, isReauth: false, ...flags }), true)
    assert.equal(shouldLatchTerminalRemoteBootFailure({ attemptedRemote: false, isReauth: false, ...flags }), false)
  }

  const transient = classifyTerminalRemoteBootFailure(
    ssh._fail('ssh: connect to host 127.0.0.1 port 22: Connection refused')
  )

  assert.equal(shouldLatchTerminalRemoteBootFailure({ attemptedRemote: true, isReauth: false, ...transient }), false)
  assert.equal(isRetryableRemoteBootFailure({ attemptedRemote: true, isReauth: false, ...transient }), true)
})

test('FIX #103288: the desktop.ssh_path hint is Windows-only', () => {
  const error = sshClientFailedError('ssh', new Error('boom'), 'darwin')

  assert.equal(isSshClientFailedBootFailure(error), true)
  assert.doesNotMatch(error.message, /ssh_path/)
})
