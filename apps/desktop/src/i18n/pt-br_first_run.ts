import type { FirstRunCopy } from './types_first_run'

export const ptBrFirstRun: FirstRunCopy = {
  beats: {
    name: 'Nome',
    theme: 'Cor de destaque',
    layout: 'Layout',
    connectors: 'Aplicativos',
    plugins: 'Plugins',
    tour: 'Visita guiada',
    build: 'Primeira tarefa'
  },
  namePlaceholder: 'Seu nome',
  useName: name => `Usar “${name}”`,
  install: 'Instalar',
  installed: 'Instalado',
  loading: 'Carregando…',
  noPlugins: 'Nenhum plugin para mostrar no momento.',
  noneChosen: 'Nenhum',
  noAnswer: 'Sem resposta',
  answerRefused: 'Resposta não aceita',
  tours: {
    quick: 'Visita rápida',
    full: 'Mostre tudo',
    skip: 'Pular, vamos criar algo'
  },
  skippedAll: 'Configuração pulada',
  skip: 'Pular configuração',
  skipTip: 'Encerrar a configuração guiada e abrir uma nova conversa.',
  skipFailed: 'Não foi possível pular a configuração',
  slashOffTitle: 'Comandos de barra estão desativados durante a configuração',
  slashOffBody: 'Eles funcionam normalmente após a conclusão da configuração.',
  localModel: {
    title: 'Este computador pode executar um modelo localmente',
    text: (model, size) =>
      `${model} (${size}) é compatível com seu hardware. Ele é gratuito e suas conversas ficam neste computador.`,
    setUp: 'Configurar',
    notNow: 'Agora não'
  },
  intro: {
    greeting: 'Oi, eu sou o Hermes.',
    plan: 'Vamos deixar tudo pronto para você. Depois, vamos fazer algo legal.'
  },
  rerun: {
    title: 'Refazer a configuração inicial',
    description: 'Abra uma nova conversa de configuração que percorre os primeiros passos com seu modelo atual.',
    action: 'Executar novamente',
    failed: 'Não foi possível iniciar a configuração inicial'
  }
}
