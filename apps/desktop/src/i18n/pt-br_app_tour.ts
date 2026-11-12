import type { Translations } from './types'

export const ptBrAppTour: Translations['appTour'] = {
  sessions: {
    title: 'Suas conversas',
    text: 'Todas as conversas ficam aqui. Pesquise, fixe ou reabra qualquer uma delas.'
  },
  composer: { title: 'Peça aqui', text: 'Diga o que você quer fazer. Digite @ para incluir um arquivo.' },
  newSession: { title: 'Comece do zero', text: 'Cada nova sessão tem seu próprio contexto. Use uma por tarefa.' },
  model: { title: 'Seletor de modelo', text: 'Escolha qual modelo responde a você.' },
  modelLocal: 'Este computador pode executar um modelo localmente: Configurações > Provedores > Modelos locais.',
  capabilities: {
    title: 'Recursos',
    text: 'Habilidades, ferramentas e plugins que o Hermes pode usar. Adicione mais aqui.'
  },
  messaging: { title: 'Mensagens', text: 'Converse com o Hermes pelo Telegram, Slack, Discord e outros aplicativos.' },
  rightPane: {
    title: 'Painel de trabalho',
    text: 'Abre arquivos, terminal, revisão e o navegador integrado à direita.'
  }
}
