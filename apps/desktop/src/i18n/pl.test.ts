import { describe, expect, it } from 'vitest'

import { en } from './en'
import { pl, plOverrides } from './pl'

function ownLeafPaths(value: unknown, prefix = ''): string[] {
  if (
    typeof value === 'function' ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    return [prefix]
  }

  if (Array.isArray(value)) {
    return value.flatMap((item, index) => ownLeafPaths(item, `${prefix}[${index}]`))
  }

  if (value && typeof value === 'object') {
    return Object.keys(value as Record<string, unknown>).flatMap(key =>
      ownLeafPaths((value as Record<string, unknown>)[key], prefix ? `${prefix}.${key}` : key)
    )
  }

  return [prefix]
}

describe('Polish desktop catalog', () => {
  it('overrides every English leaf without relying on the merge fallback', () => {
    expect(ownLeafPaths(plOverrides).sort()).toEqual(ownLeafPaths(en).sort())
  })

  it('uses Polish for core visible actions', () => {
    expect(pl.common.save).toBe('Zapisz')
    expect(pl.common.cancel).toBe('Anuluj')
    expect(pl.common.delete).toBe('Usuń')
    expect(pl.settings.nav.providers).toBe('Dostawcy')
    expect(pl.boot.ready).toBe('Hermes Desktop jest gotowy')
    expect(pl.commandCenter.commandCenter).toBe('Centrum poleceń')
    expect(pl.commandCenter.generatePet.hatch).toBe('Wykluj')
    expect(pl.assistant.approval.run).toBe('Uruchom')
    expect(pl.install.setupChoiceDesc).toBe(
      'Połącz tę aplikację z działającą bramą Hermesa albo zainstaluj Hermesa lokalnie na tym komputerze.'
    )
    expect(pl.install.connectExistingTitle).toBe('Połącz z istniejącą bramą Hermesa')
    expect(pl.install.connectExistingShort).toBe('Połącz z bramą')
    expect(pl.install.installLocalTitle).toBe('Zainstaluj Hermesa lokalnie')
    expect(pl.install.installLocalDesc).toBe(
      'Pobierz Hermesa, utwórz jego środowisko Python i uruchom backend na tym komputerze.'
    )
    expect(pl.install.remoteSetupTitle).toBe('Połącz z istniejącą bramą Hermesa')
    expect(pl.settings.gateway.sshHermesPathTitle).toBe('Ścieżka Hermesa (opcjonalnie)')
    expect(pl.settings.gateway.sshReachable('host', 'linux')).toBe('Host osiągalny: host (linux) — znaleziono Hermesa')
    expect(pl.settings.gateway.sshErrNotInstalled).toContain('ustaw ścieżkę Hermesa.')
    expect(pl.settings.gateway.sshErrUpdateRequired).toBe(
      'Przed połączeniem z Hermes Desktop zaktualizuj Hermesa na zdalnym hoście.'
    )
    expect(pl.settings.providers.localEndpoint.description).toBe(
      'Połącz Hermesa z dowolnym endpointem zgodnym z OpenAI (Zyphra, vLLM, llama.cpp, Ollama itd.).'
    )
    expect(pl.install.activeDesc).toContain('Instalator Hermesa')
    expect(pl.install.remoteUrlDesc).toBe('Użyj bazowego URL bramy Hermesa, wraz z https:// dla połączenia zdalnego.')
    expect(pl.install.probeError).toBe('Nie można połączyć się z tą bramą Hermesa.')
    expect(pl.assistant.approval.gatewayDisconnected).toBe('Brama Hermesa nie jest podłączona')
    expect(pl.assistant.clarify.gatewayDisconnected).toBe('Brama Hermesa nie jest podłączona')
    expect(pl.prompts.gatewayDisconnected).toBe('Brama Hermesa nie jest podłączona')
  })

  it('preserves state-specific toggle labels', () => {
    expect(pl.settings.mcp.toggleFailed('serwer', true)).not.toBe(pl.settings.mcp.toggleFailed('serwer', false))
    expect(pl.skills.toggleToolset('web', true)).not.toBe(pl.skills.toggleToolset('web', false))
    expect(pl.sidebar.projects.toggle('Projekt', true)).not.toBe(pl.sidebar.projects.toggle('Projekt', false))
    expect(pl.webhooks.toggleFailed('subskrypcja', true)).not.toBe(pl.webhooks.toggleFailed('subskrypcja', false))
  })
})
