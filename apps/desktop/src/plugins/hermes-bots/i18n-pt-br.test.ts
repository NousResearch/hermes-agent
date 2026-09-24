import { describe, expect, it } from 'vitest'

import { translateBotsIn } from './i18n-test-helper'

const ptBr = translateBotsIn('pt-br')

describe('Bot Mode pt-BR context menu', () => {
  it('localizes every visible bot-row action', () => {
    expect({
      autoOpenScreen: ptBr('screen.autoOpenMenu'),
      duplicate: ptBr('bot.duplicate'),
      edit: ptBr('bot.editMenu'),
      hide: ptBr('bot.hide'),
      manageGroups: ptBr('bot.manageGroups'),
      moveToSection: ptBr('sections.moveTo'),
      newChat: ptBr('bot.newChatWith'),
      openChat: ptBr('bot.openBotChat'),
      openRecent: ptBr('bot.openRecentSession'),
      openScreen: ptBr('screen.menu'),
      pin: ptBr('bot.pinToTop')
    }).toEqual({
      autoOpenScreen: 'Abrir a tela quando o bot a utilizar',
      duplicate: 'Duplicar',
      edit: 'Editar…',
      hide: 'Ocultar',
      manageGroups: 'Gerenciar grupos…',
      moveToSection: 'Mover para seção',
      newChat: 'Nova conversa com este bot',
      openChat: 'Abrir conversa do bot',
      openRecent: 'Abrir sessão recente',
      openScreen: 'Abrir tela',
      pin: 'Fixar no topo'
    })
  })

  it('localizes the Bot roster search', () => {
    expect(ptBr('roster.searchPlaceholder')).toBe('Buscar bots e conversas em grupo…')
  })
})
