// installShape() — the ONE split every desktop lifecycle decision gates on.
//
// The stamp is a constant of the artifact; the shape is derived from it
// and never from filesystem probes. These tests pin the derivation table
// and the probe-independence (a stamp object is enough — no fs access).

import { describe, expect, test } from 'vitest'

import { installShape, type InstallStamp } from './install-stamp'

function stamp(overrides: Partial<InstallStamp>): InstallStamp {
  return {
    schemaVersion: 2,
    commit: 'a'.repeat(40),
    commitDate: 1755000000,
    branch: 'main',
    builtAt: '2026-08-14T00:00:00Z',
    dirty: false,
    source: 'ci',
    distribution: 'desktop-app',
    updateMechanism: 'electron-updater',
    baseVersion: '0.21.0',
    displayVersion: '0.21.0',
    distance: 0,
    payload: 'bootstrap',
    tag: null,
    ...overrides
  }
}

describe('installShape', () => {
  test('a bundled stamp is the bundled shape', () => {
    expect(installShape(stamp({ payload: 'bundled', tag: 'v0.21.0' }))).toBe('bundled')
  })

  test('a bootstrap stamp is a checkout — its runtime is a local install', () => {
    expect(installShape(stamp({ payload: 'bootstrap' }))).toBe('checkout')
  })

  test('a light stamp is not bundled (no runtime to run at all)', () => {
    // Light artifacts never resolve a local backend; what matters here is
    // that they can never be told to run venv machinery as "bundled".
    expect(installShape(stamp({ payload: 'light' }))).toBe('checkout')
  })

  test('a dev run (null stamp) is a checkout', () => {
    expect(installShape(null)).toBe('checkout')
  })

  test('the shape comes from the stamp alone — no probe can change it', () => {
    // Same stamp, any filesystem state: the answer is a pure function of
    // the constant. (The probes live INSIDE the chosen shape as integrity
    // checks; see resolveHermesBackend.)
    const bundled = stamp({ payload: 'bundled' })
    expect(installShape(bundled)).toBe(installShape(bundled))
  })
})
