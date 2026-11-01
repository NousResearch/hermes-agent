import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const jaUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermesはまだ作業中です',
  busyBody: '更新するとHermesが再起動し、次の作業が止まります:',
  chats: (n: number) => `${n}件のチャットが作業中`,
  cron: (jobs: string) => `スケジュールジョブを実行中 (${jobs})`,
  delegation: 'バックグラウンドのサブエージェントが実行中',
  input: '確認待ちのプロンプトがあります',
  unknown: 'Hermesがアイドル状態か確認できませんでした',
  updateNow: (n: number) => `今すぐ更新(${n}件を中断)`,
  updateWhenIdle: 'アイドル時に更新',
  waitingTitle: 'Hermesの作業完了を待って更新します',
  waitingBody: '実行中の作業がなくなると自動で更新が始まります。それまでHermesは通常どおり使えます。',
  waitingFor: '待機中:',
  waitingIdle: '実行中の作業はありません。更新を開始しています…',
  waitingUntil: (time: string) => `${time}になってもHermesが作業中なら、更新は見送られます。`,
  cancel: '更新をキャンセル',
  timedOut:
    'Hermesが作業中のままだったため、更新は実行されませんでした。何も変更されていません。後でもう一度お試しください。'
}
