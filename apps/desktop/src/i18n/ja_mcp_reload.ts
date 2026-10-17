import type { TranslationOverrides } from './define-locale'

export const jaMcpReload: NonNullable<TranslationOverrides['connectorsPage']>['reload'] = {
  action: 'MCP サーバーを再読み込み',
  title: 'MCP サーバーを再読み込みしますか？',
  description:
    'このプロファイルの MCP サーバーを設定と照合し、このチャットのツールを更新します。次のメッセージでは完全なプロンプトが再送信され、プロンプトキャッシュのトークンを消費します。',
  confirm: '今すぐ再読み込み',
  deferred:
    'このプロファイルの MCP サーバーを更新しました。このチャットは実行中のターンが終わるまで現在のツールを保持します。その後にもう一度再読み込みするか、チャットを開き直してください。',
  unchanged: 'MCP サーバーはすでに最新です',
  success: 'MCP サーバーを再読み込みしました',
  failed: servers => `MCP サーバーを再読み込みしましたが、次のサーバーに接続できませんでした: ${servers}`,
  error: 'MCP サーバーは再読み込みされませんでした',
  confirmRequired: '/reload-mcp には確認が必要です',
  reloadedAlways: 'MCP サーバーを再読み込みしました · このプロファイルでは /reload-mcp で確認を求めなくなります',
  notConnected: servers => `接続できませんでした: ${servers}`,
  completionNow: '今すぐ再読み込み（今回だけ確認を省略）',
  completionAlways: '今すぐ再読み込みし、このプロファイルでは今後確認しない',
  skillsRemote:
    'スキル一覧を更新しました。ライブ再スキャンはこのプロファイルを所有するゲートウェイで実行され、このウィンドウからは実行されません。',
  skillsFailed: 'スキルの再読み込みに失敗しました'
}
