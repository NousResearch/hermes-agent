import { expect, it } from 'vitest'

import { BOTS_LOCALES } from './i18n'
import { bind } from './i18n-bind'
import type { BotsMessages } from './i18n-messages'
import { translateBotsIn } from './i18n-test-helper'

it('resolves Brazilian Portuguese Bot Mode messages through the plugin translator', () => {
  const b = bind(translateBotsIn('pt-br'), BOTS_LOCALES.en as BotsMessages)

  expect(b.bot.newTitle).toBe('Novo bot')
  expect(b.editor.skillsEnabled(2, 3)).toBe('Habilidades (2/3 ativadas)')
  expect(b.canonical.you).toBe('Você')
  expect(b.canonical.reasonProviderAuth).toBe('O provedor recusou o acesso')
  expect(b.canonical.roundOf).toBe('Rodada {n} de {max}')
  expect(b.crossGateway.pickBots(6)).toBe('Escolha de 2 a 6 bots.')
  expect(b.crossGateway.gatewayPeer('Servidor', true)).toContain('neste dispositivo')
  expect(b.crossGateway.gatewayPeer('Servidor')).toContain('Servidor')
  expect(b.group.createAndJoin).toBe('Criar e entrar')
  expect(b.group.attachmentTooLarge('foto.png')).toBe('foto.png: arquivo muito grande (máx. 15 MB).')
  expect(b.roster.filtersActive(2)).toBe('Filtrar lista, 2 filtros ativos')
  expect(b.sections.deleted('Clientes', 1)).toBe('Seção “Clientes” excluída — 1 bot movido para Sem seção')
  expect(b.group.replyCount(1)).toBe('1 resposta')
  expect(b.group.replyCount(2)).toBe('2 respostas')
  expect(b.group.compressDone('Ana', 1, '')).toBe('1 sessão de chat em grupo comprimida para Ana')
  expect(b.group.compressDone('Ana', 2, 'ok')).toBe('2 sessões de chat em grupo comprimidas para Ana — ok')
  expect(b.screen.takeOver).toBe('Assumir o controle')
  expect(b.tools.saveTest).toBe('Salvar e testar')
  expect(b.cron.runsWeekly('segunda-feira', '09:00')).toBe('Executa toda segunda-feira às 09:00')
})

it('resolves every English plugin message in pt-br without fallback and preserves named placeholders', () => {
  const t = translateBotsIn('pt-br')
  const pt = BOTS_LOCALES['pt-br']
  // Product/technical names that are also natural Portuguese labels.
  const unchanged = new Set(['roster.title', 'avatar.tabBot', 'cron.status', 'group.memberCount'])

  function verify(english: unknown, localized: unknown, path: string) {
    expect(typeof localized, path).toBe(typeof english)

    if (english && typeof english === 'object') {
      for (const [key, value] of Object.entries(english)) {
        verify(value, (localized as Record<string, unknown> | undefined)?.[key], path ? `${path}.${key}` : key)
      }

      return
    }

    const args = ['«ARG1»', 2, '«ARG3»']
    const source = typeof english === 'function' ? english(...args) : String(english)
    const translated = t(path, ...args)
    expect(translated, path).not.toBe(path)
    expect(translated.trim(), path).not.toBe('')
    expect(translated.match(/(?<!\$)\{\w+\}/g)?.sort() ?? [], path).toEqual(
      source.match(/(?<!\$)\{\w+\}/g)?.sort() ?? []
    )

    if (!unchanged.has(path)) {
      expect(translated, path).not.toBe(source)
    }
  }

  verify(BOTS_LOCALES.en, pt, '')
})
