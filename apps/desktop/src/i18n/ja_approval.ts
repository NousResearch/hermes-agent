import type { ApprovalCopy } from './types_approval'

export const jaApproval: Partial<ApprovalCopy> = {
  gatewayDisconnected: 'Hermes ゲートウェイが接続されていません',
  sendFailed: '承認応答を送信できませんでした',
  run: '実行',
  command: 'コマンド',
  moreOptions: 'その他の承認オプション',
  allowSession: 'このセッションで許可',
  alwaysAllowMenu: '常に許可…',
  jumpToApproval: '承認が必要',
  reject: '拒否',
  alwaysTitle: 'このコマンドを常に許可しますか？',
  alwaysDescription: pattern =>
    `これにより "${pattern}" パターンが永続的な許可リスト (~/.hermes/config.yaml) に追加されます。Hermes はこのセッションや将来のセッションで、このようなコマンドについて再度尋ねません。`,
  alwaysAllow: '常に許可',
  deny: '拒否',
  allow: '許可',
  allowHint: 'サンドボックスをオフにするまで',
  denyReason: '拒否の理由',
  denyReasonPlaceholder: 'Hermes に理由を伝える（任意）',
  access: {
    title: 'フォルダーアクセス',
    readOnly: '読み取り専用',
    readWrite: '読み取り/書き込み',
    alwaysTitle: 'このフォルダーを常に許可しますか？',
    alwaysDescription:
      'フォルダーがこのプロファイルのサンドボックスに追加されます。Hermes はどのチャットでも再度尋ねません。'
  },
  network: {
    title: 'サイトへのアクセス',
    alwaysTitle: 'このサイトを常に許可しますか？',
    alwaysDescription:
      'サイトがこのプロファイルの許可済みサイトに追加されます。Hermes はどのチャットでも再度尋ねません。'
  },
  schedule: {
    title: 'このジョブをスケジュールしますか？',
    allowSession: 'このチャットで許可',
    alwaysAllowMenu: '新しいジョブを常に許可…',
    alwaysTitle: '新しいスケジュールジョブを常に許可しますか？',
    alwaysDescription:
      'Hermes は確認せずにスケジュールジョブを作成します。config.yaml の command_allowlist から cron_create を削除すると、確認が再び有効になります。',
    timedOutSystemLine:
      '承認がタイムアウトしました — ジョブは作成されませんでした。Hermes にもう一度依頼するか、設定 → 安全 → 承認タイムアウトで上限を引き上げてください。',
    editTitle: 'このジョブを変更しますか？',
    editTimedOutSystemLine:
      '承認がタイムアウトしました — ジョブは変更されませんでした。Hermes にもう一度依頼するか、設定 → 安全 → 承認タイムアウトで上限を引き上げてください。'
  }
}
