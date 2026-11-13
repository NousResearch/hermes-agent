// Textos das configurações de notificação usados pelo idioma do aplicativo.
export const ptBrNotifications = {
  title: 'Notificações',
  intro: 'Notificações do sistema (não são os avisos dentro do aplicativo). Por dispositivo.',
  enableAll: 'Ativar notificações',
  enableAllDesc: 'Desativar silencia todas as notificações abaixo.',
  focusedHint: 'Os alertas de conclusão só aparecem quando o Hermes está em segundo plano.',
  kinds: {
    approval: {
      label: 'Aprovação necessária',
      description: 'Um comando está esperando você aprová-lo ou rejeitá-lo.'
    },
    input: {
      label: 'Sua resposta é necessária',
      description: 'O Hermes fez uma pergunta ou precisa de uma senha ou segredo.'
    },
    turnDone: {
      label: 'Resposta pronta',
      description: 'Um turno foi concluído enquanto o Hermes estava em segundo plano.'
    },
    turnError: {
      label: 'Falha no turno',
      description: 'Erros em turnos executados em segundo plano.'
    },
    backgroundDone: {
      label: 'Tarefa em segundo plano concluída',
      description: 'Um comando de terminal em segundo plano foi concluído.'
    },
    credits: {
      label: 'Alertas de créditos',
      description: 'O acesso aos créditos foi pausado ou restaurado.'
    },
    plugin: {
      label: 'Notificações de plugins',
      description: 'Um plugin do aplicativo enviou uma notificação enquanto o Hermes estava em segundo plano.'
    }
  },
  browserPermission: {
    title: 'Notificações do navegador',
    allow: 'Permitir',
    states: {
      default:
        'Mostre alertas deste navegador enquanto a aba estiver em segundo plano. Até você permitir, apenas o título da aba piscará.',
      granted: 'Permitidas. Os alertas aparecem quando esta aba está em segundo plano; o título da aba também pisca.',
      denied: 'Bloqueadas nas configurações de site deste navegador. O título da aba ainda pisca.',
      insecure: 'Requer HTTPS, como o Tailscale serve. Com HTTP simples, o título da aba ainda pisca.',
      unsupported: 'Este navegador não pode mostrar notificações. O título da aba ainda pisca.'
    }
  },
  test: 'Enviar notificação de teste',
  testTitle: 'Hermes',
  testBody: 'As notificações estão funcionando.',
  testSent:
    'Teste enviado. Se nada aparecer, confira as permissões de notificação do sistema e o modo Foco/Não perturbe.',
  testUnsupported: 'Este sistema não oferece suporte a notificações nativas.',
  muteSoundsTitle: 'Silenciar sons da interface',
  muteSoundsDesc:
    'Silencia os sons automáticos de conclusão, palavra de ativação e raciocínio, o retorno tátil e os sons de notificação do sistema. As prévias de som e as respostas faladas continuam sendo reproduzidas.',
  completionSoundTitle: 'Som de conclusão',
  completionSoundDesc: 'Toca quando um turno do agente termina. Escolha uma predefinição e ouça uma prévia aqui.',
  completionSoundPreview: 'Ouvir prévia'
}
