import type { CronControlCopy } from './types_cron_control'

export const ptBrCronControl: CronControlCopy = {
  stop: 'Parar',
  stopping: 'Parando…',
  stopped: 'Execução cancelada',
  finishedFirst: 'A execução terminou antes que pudesse ser interrompida',
  nothingToStop: 'Nenhuma execução em andamento',
  failedStop: 'Não foi possível parar a execução',
  lastCancelled: (reason: string) => (reason ? `Última execução cancelada: ${reason}` : 'Última execução cancelada'),
  catchUpLabel: 'Execuções perdidas',
  catchUpHint: 'O que fazer com as execuções perdidas enquanto o Hermes não estava em funcionamento.',
  catchUpChoices: {
    default: 'Usar o padrão',
    run: 'Executar uma vez ao retornar',
    skip: 'Ignorar'
  }
}
