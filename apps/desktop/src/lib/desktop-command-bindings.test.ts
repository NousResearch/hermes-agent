import { describe, expect, it } from 'vitest'

import {
  projectDesktopClientCommandBindings,
  resolveDesktopClientCommandBinding
} from './desktop-command-bindings'

const LOCAL_COMMANDS = [
  '/new',
  '/branch',
  '/yolo',
  '/wake',
  '/handoff',
  '/profile',
  '/skin',
  '/title',
  '/help',
  '/browser',
  '/journey',
  '/model',
  '/resume',
  '/compress',
  '/pet',
  '/hatch',
  '/save',
  '/status'
] as const

describe('desktop client command bindings', () => {
  it('projects only genuinely client-owned action, picker, and RPC surfaces', () => {
    const bindings = projectDesktopClientCommandBindings(LOCAL_COMMANDS.map(name => ({ name })))

    expect(bindings).toHaveLength(18)
    expect(bindings.filter(binding => binding.surface.kind === 'action')).toHaveLength(14)
    expect(bindings.filter(binding => binding.surface.kind === 'picker')).toHaveLength(2)
    expect(bindings.filter(binding => binding.surface.kind === 'rpc')).toHaveLength(2)
  })

  it('uses command_id from v2 catalogs and the canonical name for mixed-version fallback', () => {
    expect(resolveDesktopClientCommandBinding({ command_id: 'session.new', name: '/new' })).toMatchObject({
      canonicalName: '/new',
      commandId: 'session.new'
    })
    expect(resolveDesktopClientCommandBinding({ name: '/new' })).toMatchObject({
      canonicalName: '/new',
      commandId: '/new'
    })
  })

  it('canonicalizes aliases through the existing resolver instead of owning another alias table', () => {
    expect(resolveDesktopClientCommandBinding({ command_id: 'session.new', name: '/reset' })).toMatchObject({
      canonicalName: '/new',
      commandId: 'session.new',
      surface: { action: 'new', kind: 'action' }
    })
    expect(resolveDesktopClientCommandBinding({ name: '/commands' })).toMatchObject({
      canonicalName: '/help',
      surface: { action: 'help', kind: 'action' }
    })
  })

  it('does not claim server-executed, unavailable, or unknown commands', () => {
    expect(resolveDesktopClientCommandBinding({ name: '/usage' })).toBeNull()
    expect(resolveDesktopClientCommandBinding({ name: '/clear' })).toBeNull()
    expect(resolveDesktopClientCommandBinding({ name: '/does-not-exist' })).toBeNull()
  })

  it('preserves dedicated RPC execution without copying catalog semantics', () => {
    const binding = resolveDesktopClientCommandBinding({ command_id: 'session.save', name: '/save' })

    expect(binding).not.toBeNull()
    expect(binding && Object.keys(binding).sort()).toEqual(['canonicalName', 'commandId', 'surface'])
    expect(binding?.surface.kind).toBe('rpc')

    if (binding?.surface.kind !== 'rpc') {
      return
    }

    expect(binding.surface.rpc).toBe('session.save')
    expect(binding.surface.buildParams({ arg: '', command: '/save', name: 'save', sessionId: 's-1' })).toEqual({
      session_id: 's-1'
    })
  })

  it('returns immutable bindings, surfaces, and projections', () => {
    const binding = resolveDesktopClientCommandBinding({ name: '/model' })
    const bindings = projectDesktopClientCommandBindings([{ name: '/new' }, { name: '/model' }])

    expect(Object.isFrozen(binding)).toBe(true)
    expect(Object.isFrozen(binding?.surface)).toBe(true)
    expect(Object.isFrozen(bindings)).toBe(true)
  })

  it('fails closed on duplicate stable identities', () => {
    expect(() =>
      projectDesktopClientCommandBindings([
        { command_id: 'desktop.shared', name: '/new' },
        { command_id: 'desktop.shared', name: '/model' }
      ])
    ).toThrow('Desktop command_id collision')
  })

  it('fails closed when aliases project the same canonical command twice', () => {
    expect(() =>
      projectDesktopClientCommandBindings([
        { command_id: 'session.new', name: '/new' },
        { command_id: 'session.reset', name: '/reset' }
      ])
    ).toThrow('Desktop canonical command collision')
  })

  it('rejects blank explicit command ids instead of silently falling back', () => {
    expect(() => resolveDesktopClientCommandBinding({ command_id: '   ', name: '/new' })).toThrow('empty command_id')
  })
})