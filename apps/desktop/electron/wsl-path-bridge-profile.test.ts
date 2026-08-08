/**
 * Profile-scoped eligibility for the WSL path bridge (#66447).
 *
 * The single-profile tests in wsl-path-bridge.test.ts and the Windows-platform
 * gate tests in wsl-path-bridge-gate.test.ts cover the *what* (paths pass
 * through unchanged when bridging is disabled) but not the *which profile*. The
 * desktop is multi-profile: the renderer can swap the live gateway onto any
 * profile (primary or pool) without reloading the window — so the bridge
 * eligibility MUST be keyed off the **currently active profile's** backend
 * configuration, not a process-global boolean. This file proves the
 * per-profile contract:
 *
 *   1. local primary profile  → bridge ON  (preserved)
 *   2. remote primary profile → bridge OFF (preserved)
 *   3. local primary  + remote non-primary  → bridge OFF when the non-primary
 *      is active; bridge ON when the primary is active again — **no bleed**.
 *   4. remote primary + local non-primary  → bridge ON when the non-primary
 *      is active; bridge OFF when the primary is active again — **no bleed**.
 *   5. profile-scoped calls accept a profile argument; absent it falls back
 *      to the live gateway profile that the renderer announced.
 *   6. afterEach resets state so tests don't bleed into each other.
 *
 * These tests are pure behavior: they assert the eligibility function's output
 * and the public selectors' output against observable calls. They do NOT read
 * the implementation source — only the public surface exported from
 * `./wsl-path-bridge`.
 */
import assert from 'node:assert/strict'

import { afterEach, describe, test } from 'vitest'

import {
  isWslBridgeActive,
  resolveLocalReadPath,
  resolvePickerDefaultPath,
  setActiveGatewayProfile,
  setWslBridgeActive,
  setWslBridgeProfileState,
  wslPosixToWindowsAccessible
} from './wsl-path-bridge'

// ── helpers ──────────────────────────────────────────────────────────

const PROFILE_PRIMARY = 'default'
const PROFILE_LOCAL = 'team-local'
const PROFILE_REMOTE = 'team-remote'

/** Reset every profile key the bridge knows about to a clean state. */
afterEach(() => {
  setWslBridgeProfileState(PROFILE_PRIMARY, true)
  setWslBridgeProfileState(PROFILE_LOCAL, true)
  setWslBridgeProfileState(PROFILE_REMOTE, false)
  setActiveGatewayProfile(PROFILE_PRIMARY)
  setWslBridgeActive(true)
})

// ── single-profile contract (preserved behaviour) ────────────────────

describe('WSL bridge profile eligibility — single-profile contract preserved', () => {
  test('primary local → bridge ON (preserved)', () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setActiveGatewayProfile(PROFILE_PRIMARY)
    assert.equal(isWslBridgeActive(), true)
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
  })

  test('primary remote → bridge OFF (preserved)', () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, false)
    setActiveGatewayProfile(PROFILE_PRIMARY)
    assert.equal(isWslBridgeActive(), false)
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY), '/home/alex')
    assert.equal(resolveLocalReadPath('/home/alex/proj', undefined, PROFILE_PRIMARY), '/home/alex/proj')
  })
})

// ── multi-profile regression (the gap) ────────────────────────────────

describe('WSL bridge profile eligibility — multi-profile (no cross-profile bleed)', () => {
  test('local primary + remote non-primary → non-primary OFF, primary ON', () => {
    // Primary is a local backend (WSL on this Windows host). A second profile
    // points at a remote host whose POSIX paths the Windows host CANNOT open
    // via wsl.exe — bridging must be OFF for it.
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setWslBridgeProfileState(PROFILE_REMOTE, false)

    // Non-primary remote is foregrounded — bridge must be OFF for it.
    setActiveGatewayProfile(PROFILE_REMOTE)
    assert.equal(isWslBridgeActive(), false)
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_REMOTE), '/home/alex')
    assert.equal(resolveLocalReadPath('/home/alex/proj', undefined, PROFILE_REMOTE), '/home/alex/proj')

    // Swap back to local primary — bridge must be ON again, no bleed.
    setActiveGatewayProfile(PROFILE_PRIMARY)
    assert.equal(isWslBridgeActive(), true)
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
  })

  test('remote primary + local non-primary → non-primary ON, primary OFF', () => {
    // Primary is remote (no local WSL paths). A second profile is local —
    // bridging should be ON for it because its paths CAN be opened locally.
    setWslBridgeProfileState(PROFILE_PRIMARY, false)
    setWslBridgeProfileState(PROFILE_LOCAL, true)

    // Local non-primary foregrounded — bridge ON for it.
    setActiveGatewayProfile(PROFILE_LOCAL)
    assert.equal(isWslBridgeActive(), true)
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_LOCAL),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )

    // Swap back to remote primary — bridge OFF for it, no bleed.
    setActiveGatewayProfile(PROFILE_PRIMARY)
    assert.equal(isWslBridgeActive(), false)
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY), '/home/alex')
  })

  test('three profiles: each profile behaves independently', () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setWslBridgeProfileState(PROFILE_LOCAL, true)
    setWslBridgeProfileState(PROFILE_REMOTE, false)

    // Same path, different profiles, different outcomes.
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_LOCAL),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_REMOTE), '/home/alex')
  })
})

// ── profile-argument contract ────────────────────────────────────────

describe('WSL bridge profile eligibility — selector argument contract', () => {
  test("selector with explicit profile → that profile's bridge state", () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setWslBridgeProfileState(PROFILE_REMOTE, false)
    setActiveGatewayProfile(PROFILE_PRIMARY)

    // Explicit profile wins over the live gateway.
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_REMOTE), '/home/alex')
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
  })

  test("selector with no profile → live gateway profile's bridge state", () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setWslBridgeProfileState(PROFILE_REMOTE, false)
    setActiveGatewayProfile(PROFILE_REMOTE)

    // No profile argument → fallback to active gateway profile (remote → OFF).
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu'), '/home/alex')
    assert.equal(resolveLocalReadPath('/home/alex/proj'), '/home/alex/proj')

    setActiveGatewayProfile(PROFILE_PRIMARY)
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
  })

  test('selector with unknown profile → bridge ON (defaults to active for new profiles)', () => {
    // A profile that has never been seeded should default to the safe
    // "bridge ON" behaviour so a brand-new local profile isn't accidentally
    // disabled. The renderer seeds the state at boot; an unknown key here is
    // either a renderer race or a profile created mid-session.
    setWslBridgeProfileState(PROFILE_PRIMARY, false)
    setActiveGatewayProfile(PROFILE_PRIMARY)

    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', 'unknown-profile'),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
  })
})

// ── legacy back-compat: setWslBridgeActive targets the active profile ─

describe('WSL bridge profile eligibility — legacy toggle targets active profile', () => {
  test('setWslBridgeActive(false) flips the active profile, not a process-global', () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setWslBridgeProfileState(PROFILE_REMOTE, true)
    setActiveGatewayProfile(PROFILE_REMOTE)

    setWslBridgeActive(false)

    // Active profile (REMOTE) flipped to OFF; primary untouched.
    assert.equal(isWslBridgeActive(), false)
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
    assert.equal(resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_REMOTE), '/home/alex')
  })

  test('setWslBridgeActive(true) restores the active profile only', () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, true)
    setWslBridgeProfileState(PROFILE_REMOTE, false)
    setActiveGatewayProfile(PROFILE_REMOTE)

    setWslBridgeActive(true)

    // Active (REMOTE) restored; primary unchanged.
    assert.equal(isWslBridgeActive(), true)
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_REMOTE),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
    assert.equal(
      resolvePickerDefaultPath('/home/alex', 'Ubuntu', PROFILE_PRIMARY),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex'
    )
  })
})

// ── wslPosixToWindowsAccessible stays pure / unchanged ───────────────

describe('WSL bridge profile eligibility — POSIX translation stays pure', () => {
  test('wslPosixToWindowsAccessible ignores bridge state (pure translation)', () => {
    setWslBridgeProfileState(PROFILE_PRIMARY, false)
    setActiveGatewayProfile(PROFILE_PRIMARY)

    // The translator does NOT consult the bridge state — it's pure POSIX → UNC.
    // Tests elsewhere assert that the bridge gate short-circuits BEFORE this
    // function is reached. A regression here would mean coupling leaked into
    // a helper that should stay side-effect-free.
    assert.equal(
      wslPosixToWindowsAccessible('/home/alex/proj', 'Ubuntu'),
      '\\\\wsl.localhost\\Ubuntu\\home\\alex\\proj'
    )
    assert.equal(wslPosixToWindowsAccessible('/mnt/c/Users/alex', 'Ubuntu'), 'C:\\Users\\alex')
  })
})
