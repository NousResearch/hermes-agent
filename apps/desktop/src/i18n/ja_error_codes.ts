import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const jaErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: 'このチャットはスケジュールされたジョブ用です',
    body: 'このチャットはそのジョブの実行専用のため、メッセージは実行されませんでした。送信するには新しいチャットを開始してください。'
  },
  workspace_unavailable: {
    title: 'このチャットのフォルダーが見つかりません',
    body: 'このチャットの作業フォルダーが移動または削除されたため、メッセージは実行されませんでした。フォルダーを作り直すか、「Move to project」で別のフォルダーを割り当ててから、もう一度お試しください。'
  },
  launch_credentials_unavailable: {
    title: 'このチャットの API キーが変更されました',
    body: 'このチャットの開始時の API キーがその後変更または削除されたため、メッセージは送信されませんでした。開いているチャットが自動で別のキーに切り替わることはありません。「新しいセッションを開始」で現在のキーを使ってください。'
  }
}
