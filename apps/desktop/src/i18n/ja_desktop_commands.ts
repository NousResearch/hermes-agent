// Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines
// (en_desktop_commands.ts is the source).
export const jaDesktopCommands = {
  error: (message: string) => `エラー: ${message}`,
  rollbackUsage: '使い方: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled:
    'このセッションではチェックポイントが有効になっていません（config.yaml で checkpoints.enabled を設定してください）。',
  checkpointsEmpty: 'この作業ディレクトリにはまだチェックポイントがありません。',
  checkpointsHeader: 'チェックポイント（新しい順）:',
  checkpointsHint:
    '/rollback <n> [file] で復元します（--all を付けると自分の編集も上書きします）。/rollback diff <n> でプレビューできます。',
  noDiff: 'そのチェックポイント以降の変更はありません。',
  rollbackFailed: (error: string) => `ロールバックに失敗しました: ${error}`,
  unknownError: '不明なエラー',
  restored: (hash: string) => (hash ? `チェックポイント ${hash} を復元しました。` : 'チェックポイントを復元しました。'),
  rewound: (count: string) => `この会話から ${count} 件のメッセージを巻き戻しました。`,
  reopenForRewind: '巻き戻した会話を表示するには、このチャットを開き直してください。',
  agentsEmpty: 'このチャットに実行中のサブエージェントやバックグラウンドプロセスはありません。',
  subagentsHeader: (count: string) => `サブエージェント（${count}）:`,
  subagentFallback: 'サブエージェント',
  statusRunning: '実行中',
  toolCalls: (count: string) => `ツール呼び出し ${count} 回`,
  processesHeader: (count: string) => `バックグラウンドプロセス（${count}）:`,
  configHeader: '設定（表示、承認、貼り付け、音声の設定のみ。それ以外は表示されません）:',
  configDefaults: '（既定値）',
  configFooter:
    '設定画面で変更するか、ゲートウェイのホストで `hermes config edit` を使って config.yaml を編集してください。',
  historyHeader: (count: string) => `会話履歴（${count} 件のメッセージ）:`,
  historyEmpty: 'まだメッセージはありません。',
  historyYou: 'あなた',
  historyHermes: 'Hermes',
  saveUsage: '使い方: /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    '保存されませんでした: このゲートウェイは秘匿化・ファイル名指定・JSON 以外のコピーを書き出せません。ゲートウェイを更新するか、オプションなしで /save を実行してゲートウェイのホストに生の JSON を書き込んでください。',
  savedOnHost: (path: string) => `ゲートウェイのホストに保存しました: ${path}`,
  unknownPath: '（パス不明）',
  saveUnsupported: 'エラー: このデスクトップ版はファイルを保存できません',
  saveDialogTitle: '会話を保存',
  savedTo: (path: string) => `会話を ${path} に保存しました`,
  saveCancelled: '保存されませんでした（キャンセル）。',
  exportRemote:
    '/export はゲートウェイのホストからプロファイルをパッケージ化しますが、そのホストはこのコンピューターではありません。そのホストで `hermes profile export` を実行してください。',
  importRemote:
    '/import はゲートウェイのホストにプロファイルをインストールしますが、そのホストはこのコンピューターではありません。そのホストで `hermes profile import <archive>` を実行してください。',
  timestampsUsage: '使い方: /timestamps [on|off|status]',
  timestampsOn: 'タイムスタンプ: オン（このウィンドウのみ。維持するには config.yaml で display.timestamps を設定）',
  timestampsOff: 'タイムスタンプ: オフ（このウィンドウのみ。維持するには config.yaml で display.timestamps を設定）'
}
