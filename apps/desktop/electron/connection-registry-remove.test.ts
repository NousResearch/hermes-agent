import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, it, onTestFinished } from 'vitest'

import { type ConnectionRegistry, normalizeRegistry, reconcileRegistryDrift } from './connection-registry'
import { removeSavedConnection } from './connection-registry-remove'
import { writeSecretFileAtomic } from './hardening'

function fixture(kind: 'cloud' | 'remote' | 'ssh', primary = 'saved-ssh') {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-connection-removal-'))
  onTestFinished(() => fs.rmSync(root, { recursive: true, force: true }))
  const configPath = path.join(root, 'connection.json')
  const registryPath = path.join(root, 'connections.json')

  const remote =
    kind === 'ssh'
      ? { mode: 'ssh', host: 'old.example.invalid', user: 'hermes', keyPath: '/offline/old-key' }
      : { url: 'http://127.0.0.1:9119', authMode: 'oauth' }

  const config = {
    mode: kind,
    remote,
    profiles: {
      research: { mode: 'remote', url: 'https://research.example.invalid', authMode: 'oauth' },
      writing: { mode: 'ssh', host: 'writing.example.invalid', user: 'writer' }
    }
  }

  const registry = normalizeRegistry({
    version: 2,
    primary,
    launchMode: 'last-used',
    lastUsed: 'saved-ssh',
    connections: [
      { id: 'local', kind: 'local', label: 'This device' },
      { id: 'old-gateway', kind, label: 'Old gateway', ...remote },
      { id: 'saved-ssh', kind: 'ssh', label: 'Saved SSH', host: 'saved.example.invalid', user: 'hermes' }
    ]
  })

  const writers = {
    writeConfig: (value: Record<string, any>) => writeSecretFileAtomic(configPath, JSON.stringify(value)),
    writeRegistry: (value: ConnectionRegistry) => writeSecretFileAtomic(registryPath, JSON.stringify(value))
  }

  writers.writeConfig(config)
  writers.writeRegistry(registry)

  return {
    config,
    registry,
    writers,
    readConfig: () => JSON.parse(fs.readFileSync(configPath, 'utf8')),
    coldRead: () =>
      reconcileRegistryDrift(
        normalizeRegistry(JSON.parse(fs.readFileSync(registryPath, 'utf8'))),
        JSON.parse(fs.readFileSync(configPath, 'utf8'))
      )
  }
}

it.each(['remote', 'cloud', 'ssh'] as const)(
  'keeps an explicitly removed legacy %s route absent across cold reads without disabling future Settings drift repair',
  async kind => {
    for (const primary of ['saved-ssh', 'old-gateway']) {
      const state = fixture(kind, primary)
      await removeSavedConnection('old-gateway', state.registry, state.config, state.writers)

      const restart = state.coldRead()
      expect(restart.changed).toBe(false)
      expect(restart.registry.connections.map(connection => connection.id)).toEqual(['local', 'saved-ssh'])
      expect(restart.registry.primary).toBe(primary === 'old-gateway' ? 'local' : 'saved-ssh')
      expect(restart.registry.lastUsed).toBe('saved-ssh')
      expect(restart.registry.launchMode).toBe('last-used')
      expect(state.readConfig().profiles).toEqual(state.config.profiles)
      expect(state.readConfig().mode).toBe('local')

      // An explicit later Settings save must still register its route.
      state.writers.writeConfig(state.config)
      const reapplied = state.coldRead()
      expect(reapplied.changed).toBe(true)
      expect(reapplied.registry.connections).toHaveLength(3)
    }

    // Removing a different saved connection must not clear the global route.
    const unrelated = fixture(kind)
    await removeSavedConnection('saved-ssh', unrelated.registry, unrelated.config, unrelated.writers)
    expect(unrelated.readConfig()).toEqual(unrelated.config)
    expect(unrelated.coldRead().changed).toBe(false)

    // A surviving equivalent route still owns the global configuration.
    const duplicate = fixture(kind)
    duplicate.registry.connections.push({
      ...duplicate.registry.connections.find(connection => connection.id === 'old-gateway')!,
      id: 'equivalent-route',
      ...(kind === 'ssh'
        ? { port: 22 }
        : { kind: kind === 'cloud' ? 'remote' : 'cloud', url: 'http://127.0.0.1:9119/' })
    })
    duplicate.writers.writeRegistry(duplicate.registry)
    await removeSavedConnection('old-gateway', duplicate.registry, duplicate.config, duplicate.writers)
    expect(duplicate.readConfig()).toEqual(duplicate.config)
    expect(duplicate.coldRead().changed).toBe(false)

    // Removing another entry must not erase a pre-existing Settings-only route.
    const drift = fixture(kind)
    drift.registry.connections = drift.registry.connections.filter(connection => connection.id !== 'old-gateway')
    drift.writers.writeRegistry(drift.registry)
    await removeSavedConnection('saved-ssh', drift.registry, drift.config, drift.writers)
    expect(drift.readConfig()).toEqual(drift.config)
    expect(drift.coldRead().changed).toBe(true)
  }
)

it.each(['config', 'registry'] as const)(
  'preserves both files and the original error when the %s write fails during legacy-route removal',
  async failingStore => {
    const state = fixture('remote')
    const failure = new Error('injected storage failure')
    let failed = false

    const write = (store: 'config' | 'registry', value: any) => {
      if (store === failingStore && !failed) {
        failed = true
        throw failure
      }

      if (store === 'config') {
        state.writers.writeConfig(value)
      } else {
        state.writers.writeRegistry(value)
      }
    }

    await expect(
      removeSavedConnection('old-gateway', state.registry, state.config, {
        writeConfig: value => write('config', value),
        writeRegistry: value => write('registry', value)
      })
    ).rejects.toBe(failure)
    expect(state.readConfig()).toEqual(state.config)
    expect(state.coldRead()).toEqual({ changed: false, registry: state.registry })
  }
)

it('does not overwrite a subsequent registry edit while removal finishes', async () => {
  const state = fixture('remote')
  const removing = removeSavedConnection('old-gateway', state.registry, state.config, state.writers)
  const edited = state.coldRead().registry
  edited.connections.push({ id: 'new-ssh', kind: 'ssh', label: 'New SSH', host: 'new.example.invalid' })
  state.writers.writeRegistry(edited)
  await removing

  expect(state.coldRead().registry.connections.map(connection => connection.id)).toEqual([
    'local',
    'saved-ssh',
    'new-ssh'
  ])
})
