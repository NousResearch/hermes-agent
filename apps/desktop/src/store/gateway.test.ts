import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { HermesGateway } from '@/hermes'
import { $gatewayState, setGatewayState } from '@/store/session'

import {
  $gateway,
  closeSecondaryGateways,
  ensureGatewayForProfile,
  ensureProfileSocketOpen,
  gatewayForProfile,
  pruneSecondaryGateways,
  setPrimaryGateway
} from './gateway'

// No window.hermesDesktop in jsdom, so openSecondary() is a no-op: sockets are
// created + registered but never actually connect — exactly what these tests
// need (routing/pointer semantics, not transport).

describe('gateway registry profile-socket accessors', () => {
  const primary = new HermesGateway()

  beforeEach(async () => {
    setPrimaryGateway(primary, 'default')
    // Pin the active pointer to the primary so each test starts from the
    // single-profile baseline.
    await ensureGatewayForProfile('default')
  })

  afterEach(async () => {
    closeSecondaryGateways()
    await ensureGatewayForProfile('default')
    setPrimaryGateway(null)
    $gateway.set(null)
    setGatewayState('idle')
  })

  describe('gatewayForProfile', () => {
    it('passes the primary through for the primary key and blank/null keys', () => {
      expect(gatewayForProfile('default')).toBe(primary)
      expect(gatewayForProfile(null)).toBe(primary)
      expect(gatewayForProfile(undefined)).toBe(primary)
      expect(gatewayForProfile('   ')).toBe(primary)
    })

    it('returns null for a profile with no live secondary', () => {
      expect(gatewayForProfile('beta')).toBeNull()
    })

    it('returns the secondary once one exists', async () => {
      const socket = await ensureProfileSocketOpen('beta')

      expect(gatewayForProfile('beta')).toBe(socket)
    })
  })

  describe('ensureProfileSocketOpen', () => {
    it('opens a secondary WITHOUT moving the active gateway or its reported state', async () => {
      setGatewayState('open')
      const activeBefore = $gateway.get()

      const socket = await ensureProfileSocketOpen('beta')

      expect(socket).toBeInstanceOf(HermesGateway)
      expect(socket).not.toBe(primary)
      // The active pointer and the composer's gateway-state mirror are exactly
      // where they were — a background pane's socket must be invisible to the
      // foreground surfaces.
      expect($gateway.get()).toBe(activeBefore)
      expect($gatewayState.get()).toBe('open')
    })

    it('reuses the existing secondary on repeat calls', async () => {
      const first = await ensureProfileSocketOpen('beta')
      const second = await ensureProfileSocketOpen('beta')

      expect(second).toBe(first)
    })

    it('is a passthrough for the primary profile', async () => {
      setGatewayState('open')

      expect(await ensureProfileSocketOpen('default')).toBe(primary)
      expect(await ensureProfileSocketOpen(null)).toBe(primary)
      expect($gatewayState.get()).toBe('open')
    })
  })

  describe('ensureGatewayForProfile (the swap, by contrast)', () => {
    it('moves the active gateway to the profile socket', async () => {
      const socket = await ensureProfileSocketOpen('beta')

      await ensureGatewayForProfile('beta')

      expect($gateway.get()).toBe(socket)

      await ensureGatewayForProfile('default')

      expect($gateway.get()).toBe(primary)
    })
  })

  describe('pruneSecondaryGateways', () => {
    it('spares profiles in the keep set and evicts the rest', async () => {
      const kept = await ensureProfileSocketOpen('beta')
      await ensureProfileSocketOpen('gamma')

      pruneSecondaryGateways(new Set(['beta']))

      expect(gatewayForProfile('beta')).toBe(kept)
      expect(gatewayForProfile('gamma')).toBeNull()

      pruneSecondaryGateways(new Set())

      expect(gatewayForProfile('beta')).toBeNull()
    })
  })
})
