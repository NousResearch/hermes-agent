import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const enUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes is still working',
  busyBody: 'Updating restarts Hermes, which stops:',
  chats: (n: number) => (n === 1 ? '1 chat is working' : `${n} chats are working`),
  cron: (jobs: string) => `Scheduled job running (${jobs})`,
  delegation: 'Background subagents are running',
  input: 'A prompt is waiting for your answer',
  unknown: 'Hermes could not confirm it is idle',
  updateNow: (n: number) => `Update now (interrupts ${n})`,
  updateWhenIdle: 'Update when idle',
  waitingTitle: 'Update waits for Hermes to finish',
  waitingBody: 'The update starts on its own once nothing is running. Keep working; Hermes stays usable until then.',
  waitingFor: 'Waiting for:',
  waitingIdle: 'Nothing is running. Starting the update…',
  waitingUntil: (time: string) => `If Hermes is still busy at ${time}, the update is skipped.`,
  cancel: 'Cancel update',
  timedOut: 'Hermes stayed busy, so the update did not run. Nothing was changed; try again later.'
}
