import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'

import { resolveVersionStatus } from './version-status'

const copy = en.shell.statusbar

const client = (over: Partial<Parameters<typeof resolveVersionStatus>[0]> = {}) =>
  resolveVersionStatus({ applying: false, copy, remote: false, restarting: false, target: 'client', ...over })

const backend = (over: Partial<Parameters<typeof resolveVersionStatus>[0]> = {}) =>
  resolveVersionStatus({ applying: false, copy, remote: true, restarting: false, target: 'backend', ...over })

describe('resolveVersionStatus', () => {
  it('labels a current local client with its version and sha detail', () => {
    const status = client({ sha: 'abc1234', version: '0.4.2' })

    expect(status.label).toBe('v0.4.2')
    expect(status.detail).toBe('abc1234')
    expect(status.hasUpdate).toBe(false)
    expect(status.unknown).toBe(false)
  })

  it('appends the commit diff when the client is behind', () => {
    const status = client({ behind: 12, branch: 'main', version: '0.4.2' })

    expect(status.label).toBe('v0.4.2 (+12)')
    expect(status.hasUpdate).toBe(true)
    expect(status.tooltip).toContain('12 commits behind main')
  })

  it('names the client as one of two versions in remote mode', () => {
    expect(client({ remote: true, version: '0.4.2' }).label).toBe('client v0.4.2')
  })

  it('falls back to the sha, then to unknown, when there is no version', () => {
    expect(client({ sha: 'abc1234' }).label).toBe('abc1234')
    expect(client({ sha: 'abc1234' }).unknown).toBe(false)
    expect(client().label).toBe(copy.unknown)
    expect(client().unknown).toBe(true)
  })

  it('drops the diff and the sha detail while an apply is in flight', () => {
    const applying = client({ applying: true, behind: 3, sha: 'abc1234', version: '0.4.2' })

    expect(applying.label).toBe('v0.4.2 · update')
    expect(applying.detail).toBeUndefined()
    expect(applying.hasUpdate).toBe(false)

    expect(client({ applying: true, restarting: true, version: '0.4.2' }).label).toBe('v0.4.2 · restart')
  })

  it('leads the tooltip with the apply message while applying', () => {
    expect(client({ applyMessage: 'Pulling…', applying: true, version: '0.4.2' }).tooltip).toBe(
      'Pulling… · Hermes Desktop v0.4.2'
    )
    expect(client({ applying: true, version: '0.4.2' }).tooltip).toBe(
      `${copy.updateInProgress} · Hermes Desktop v0.4.2`
    )
  })

  it('labels the backend target distinctly and never claims a client sha', () => {
    const status = backend({ sha: 'abc1234', version: '0.4.2' })

    expect(status.label).toBe('backend v0.4.2')
    expect(status.detail).toBeUndefined()
    expect(status.tooltip).toBe('Backend v0.4.2')
  })

  it('falls back to (update) for a backend that cannot count commits', () => {
    const status = backend({ updateAvailable: true, version: '0.4.2' })

    expect(status.label).toBe('backend v0.4.2 (update)')
    expect(status.hasUpdate).toBe(true)
  })

  it('prefers the exact commit diff over the generic (update) hint', () => {
    expect(backend({ behind: 4, updateAvailable: true, version: '0.4.2' }).label).toBe('backend v0.4.2 (+4)')
  })

  it('hides a backend row that has no version at all', () => {
    expect(backend().unknown).toBe(true)
  })
})
