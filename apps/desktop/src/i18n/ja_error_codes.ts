import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const jaErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: 'このチャットはスケジュールされたジョブ用です',
    body: 'このチャットはそのジョブの実行専用のため、メッセージは実行されませんでした。送信するには新しいチャットを開始してください。'
  },
  session_read_only: {
    title: 'このチャットは読み取り専用です',
    body: 'Hermes が表示はできても続行できない保存済みの履歴のため、メッセージは実行されませんでした。送信するには新しいチャットを開始してください。'
  },
  workspace_unavailable: {
    title: 'このチャットのフォルダーが見つかりません',
    body: 'このチャットの作業フォルダーが移動または削除されたため、メッセージは実行されませんでした。フォルダーを作り直すか、「Move to project」で別のフォルダーを割り当ててから、もう一度お試しください。'
  },
  auth: {
    title: provider => `${provider} が API キーを拒否しました`,
    body: () => '保存されたキーが無効か、取り消されています。キーを更新してから、メッセージを再送してください。'
  },
  auth_permanent: {
    title: provider => `${provider} が API キーを拒否しました`,
    body: () => '保存されたキーが無効か、取り消されています。キーを更新してから、メッセージを再送してください。'
  },
  billing: {
    title: provider => `${provider} のクレジットが不足しています`,
    body: provider => `${provider} で有料モデルを使い続けるには、クレジットを追加してください。`
  },
  no_reply: {
    title: '返信が届きませんでした',
    body: 'Hermes はこのターンを完了しましたが、返信がこのウィンドウに届きませんでした。再試行してもう一度送信してください。'
  },
  upstream_blocked: {
    title: provider => `${provider} がこのリクエストをブロックしました`,
    body: provider =>
      `${provider} の手前にあるファイアウォールが、モデルに届く前にリクエストを拒否しました。プロバイダーを切り替えるか、カスタムの User-Agent を設定してください。`
  },
  context_overflow: {
    title: 'このチャットは長すぎます',
    body: (_provider, model) =>
      `会話が${model ? ` ${model} の` : 'モデルの'}コンテキストウィンドウに収まらなくなりました。続けるには会話を圧縮してください。`
  },
  payload_too_large: {
    title: 'このメッセージは大きすぎます',
    body: provider =>
      `添付ファイルを含むリクエストが ${provider} のサイズ上限を超えています。会話を圧縮するか、添付ファイルを削除してください。`
  },
  model_not_found: {
    title: (_provider, model) => `${model || 'このモデル'} は利用できません`,
    body: (provider, model) =>
      `${provider} はお使いのアカウントで ${model || 'このモデル'} を提供していません。別のモデルを選んでください。`
  },
  truncated: {
    title: '返信が途中で切れました',
    body: (_provider, model) =>
      `${model || 'モデル'} の返答が完了前に止まりました。「続行」で中断したところから再開できます。`
  },
  loop_error: {
    title: 'Hermes はこのステップを完了できませんでした',
    body: '返信の処理中に問題が発生しました。再試行してください。再び発生する場合は、詳細から診断情報を送信してください。'
  },
  agent_init_failed: {
    title: 'このチャットを開始できませんでした',
    body: provider =>
      `Hermes はこのチャットで ${provider} に接続できませんでした。再試行してください。失敗が続く場合は、設定でプロバイダーを確認してください。`
  },
  SESSION_NOT_OWNED: {
    title: 'チャットが別の場所で開かれています',
    body: 'このチャットは別の Hermes ウィンドウまたはターミナルで開かれています。ここで新しいチャットを開始するか、先にそちらで閉じてください。'
  },
  disk_full: {
    title: 'ディスクがいっぱいです',
    body: 'Hermes はこの会話を保存できませんでした。空き容量を確保してから再試行してください。'
  },
  free_tier_disabled: {
    title: 'サインインしてチャットを続ける',
    body: 'Nous アカウントでサインインすると、チャットを続けられます。無料です。'
  },
  free_tier_rate_limited: {
    title: '無料枠を使い切りました',
    body: 'まもなくリセットされます。Nous アカウントでサインインすると、より多くの利用枠を無料で使えます。'
  },
  free_tier_at_capacity: {
    title: '無料枠が混み合っています',
    body: 'サインインすると待たずに使えます（無料）。または、しばらくしてからもう一度お試しください。'
  },
  free_tier_model_not_free: {
    title: 'このモデルにはサインインが必要です',
    body: '現在は無料モデルを使用しています。Nous アカウントでサインインすると、より多くのモデルを無料で使えます。'
  },
  free_tier_route: {
    title: '無料モデルに接続できません',
    body: 'Nous アカウントで無料でサインインするか、NOUS_INFERENCE_BASE_URL の設定を確認してください。'
  },
  free_tier_outage: {
    title: '無料モデルで問題が発生しています',
    body: '1 分ほどしてからメッセージを再送してください。'
  },
  free_tier_refused: {
    title: 'サインインしてチャットを続ける',
    body: 'Nous アカウントでのサインインは無料です。'
  },
  launch_credentials_unavailable: {
    title: 'このチャットの API キーが変更されました',
    body: 'このチャットの開始時の API キーがその後変更または削除されたため、メッセージは送信されませんでした。開いているチャットが自動で別のキーに切り替わることはありません。「新しいセッションを開始」で現在のキーを使ってください。'
  },
  local_runtime_unavailable: {
    title: 'ローカルモデルサーバーに接続できません',
    body: 'ローカルモデルサーバーはまだ動作していません。起動中の可能性があります。少し待ってから再試行するか、別のモデルに切り替えてください。',
    bodyNoRetry:
      'ローカルモデルサーバーはオフになっています。Hermes デスクトップアプリか、ターミナルの `hermes model` から起動するか、別のモデルに切り替えてください。'
  }
}
