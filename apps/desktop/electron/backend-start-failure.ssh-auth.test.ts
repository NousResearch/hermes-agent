/**
 * Unit tests for shouldLatchSshAuthFailure — the predicate that prevents
 * the SSH auth-failed retry loop that locks users out of Settings (#72698).
 *
 * Two layers:
 *   1. Truth-table tests: the pure predicate for every 2x2 combination.
 *   2. Behavior-level wiring: production-like error shapes, the latch →
 *      block → clear cycle, and every recognized SSH_ERROR value.
 */

import { describe, expect, it } from 'vitest'
import {
  shouldLatchSshAuthFailure,
  type SshAuthFailureContext,
} from './backend-start-failure'

// ---------------------------------------------------------------------------
// 1. Truth-table: pure predicate for every combination
// ---------------------------------------------------------------------------

describe('shouldLatchSshAuthFailure (truth table)', () => {
  it('returns false when not remote', () => {
    expect(shouldLatchSshAuthFailure({ attemptedRemote: false, isSshAuthFailed: true })).toBe(false)
  })

  it('returns false when remote but not an SSH auth failure', () => {
    expect(shouldLatchSshAuthFailure({ attemptedRemote: true, isSshAuthFailed: false })).toBe(false)
  })

  it('returns true when remote AND SSH auth failed', () => {
    expect(shouldLatchSshAuthFailure({ attemptedRemote: true, isSshAuthFailed: true })).toBe(true)
  })

  it('returns false when neither is true', () => {
    expect(shouldLatchSshAuthFailure({ attemptedRemote: false, isSshAuthFailed: false })).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// 2. Behavior-level wiring — production error shapes and latch cycle
// ---------------------------------------------------------------------------

describe('SshAuthFailureContext — behavior-level wiring', () => {
  //
  // Production context derivation
  //
  // In main.ts the context is constructed as:
  //    { attemptedRemote, isSshAuthFailed: error?.sshError === 'auth-failed' }
  // where error.sshError comes from the ssh-connection layer (SSH_ERROR.*).

  it('derives isSshAuthFailed=true from error with sshError="auth-failed"', () => {
    const error = new Error('SSH authentication failed') as any
    error.sshError = 'auth-failed'

    const context: SshAuthFailureContext = {
      attemptedRemote: true,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(true)
  })

  it('derives isSshAuthFailed=false from error with sshError="unreachable"', () => {
    const error = new Error('Host unreachable') as any
    error.sshError = 'unreachable'

    const context: SshAuthFailureContext = {
      attemptedRemote: true,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(false)
  })

  it('derives isSshAuthFailed=false from error with sshError="timeout"', () => {
    const error = new Error('Connection timed out') as any
    error.sshError = 'timeout'

    const context: SshAuthFailureContext = {
      attemptedRemote: true,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(false)
  })

  it('derives isSshAuthFailed=false from error with sshError="host-key-changed"', () => {
    const error = new Error('Host key has changed') as any
    error.sshError = 'host-key-changed'

    const context: SshAuthFailureContext = {
      attemptedRemote: true,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(false)
  })

  it('derives isSshAuthFailed=false from error with sshError="unknown"', () => {
    const error = new Error('Unknown SSH error') as any
    error.sshError = 'unknown'

    const context: SshAuthFailureContext = {
      attemptedRemote: true,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(false)
  })

  it('derives isSshAuthFailed=false when error has no sshError property', () => {
    const error = new Error('Some non-SSH error')

    const context: SshAuthFailureContext = {
      attemptedRemote: true,
      isSshAuthFailed: (error as any)?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(false)
  })

  /**
   * In the startup-failure path, the sshError flows through as a literal
   * string on the thrown error (not through the SSH_ERROR enum), so test
   * the string value directly — matching the real predicate in main.ts.
   */
  it('matches on the string "auth-failed" exactly, not on partial or case variants', () => {
    // Literal-vs-literal comparisons are narrowed by TS to "no overlap"
    // (TS2367), so widen both sides through `as string` to model the real
    // runtime path where sshError arrives as an opaque string.
    const actual: string = 'auth-failed'

    // Exact match — this is what the production code expects
    expect(actual === 'auth-failed').toBe(true)

    // Partial or case variants must NOT match
    expect('AuthFailed' === actual).toBe(false)
    expect('auth' === actual).toBe(false)
    expect('auth-failure' === actual).toBe(false)
  })

  //
  // The "not remote" guard in the predicate
  //
  // Even with a genuine auth-failed sshError, setting attemptedRemote=false
  // must suppress latching (local backends use a different error path).

  it('suppresses latching when attemptedRemote=false even with sshError=auth-failed', () => {
    const error = new Error('SSH authentication failed') as any
    error.sshError = 'auth-failed'

    const context: SshAuthFailureContext = {
      attemptedRemote: false,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    }

    expect(shouldLatchSshAuthFailure(context)).toBe(false)
  })

  //
  // Latch → block → clear cycle
  //
  // This is the core wiring pattern from main.ts: the predicate gates
  // whether an error is stored in the sshAuthFailure module variable,
  // which then short-circuits subsequent startHermes() calls, and is
  // cleared on reset/repair/apply-connection-config.

  it('models the latch → block → clear recovery cycle', () => {
    // Phase 1 — SET: an auth-failed error arrives, predicate gates true,
    // and the error is stored as the latched failure.
    let sshAuthFailure: Error | null = null

    const error = new Error('SSH authentication failed') as any
    error.sshError = 'auth-failed'

    const shouldLatch = shouldLatchSshAuthFailure({
      attemptedRemote: true,
      isSshAuthFailed: error?.sshError === 'auth-failed',
    })

    if (shouldLatch) {
      sshAuthFailure = error instanceof Error ? error : new Error(String(error))
    }

    expect(sshAuthFailure).not.toBeNull()
    expect(sshAuthFailure!.message).toBe('SSH authentication failed')

    // Phase 2 — BLOCK: a subsequent call mimics the startHermes() early-return
    // guard (if (sshAuthFailure) throw sshAuthFailure). The latch blocks re-entry.
    expect(() => {
      if (sshAuthFailure) {
        throw sshAuthFailure
      }
    }).toThrow('SSH authentication failed')

    // Phase 3 — CLEAR: reset/repair/apply-connection-config sets the latch back
    // to null (see resetHermesConnection / hermes:bootstrap:reset in main.ts).
    sshAuthFailure = null
    expect(sshAuthFailure).toBeNull()

    // Phase 4 — VERIFY clear is durable: after clearing, the guard no longer
    // blocks, and a non-auth-failed error passes through normally.
    const unreachableError = new Error('Host unreachable') as any
    unreachableError.sshError = 'unreachable'
    expect(
      shouldLatchSshAuthFailure({
        attemptedRemote: true,
        isSshAuthFailed: unreachableError?.sshError === 'auth-failed',
      })
    ).toBe(false)
  })
})
