// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const jaSessionCommands = {
  undoUsage: '使い方: /undo [N]',
  undoNothing: '元に戻すものはありません',
  undidOne: (count: string) => `${count} 件のメッセージを元に戻しました — 削除したメッセージは入力欄にあります`,
  undidOther: (count: string) => `${count} 件のメッセージを元に戻しました — 削除したメッセージは入力欄にあります`,
  retryNothing: '再試行するものはありません',
  retryUnsupported: '/retry: メディアや不明なコンテンツを含むターンは再試行できません',
  busy: (command: string) =>
    `セッションが使用中です — Hermes はまだ応答中です。先に現在の応答を停止してから（停止ボタン、またはターミナルでは Ctrl+C）/${command} を実行してください。`,
  busyCompressing: (command: string) =>
    `セッションが使用中です — /compress を実行中です。完了したら /${command} を実行してください。`,
  steerUsage: '使い方: /steer <プロンプト>',
  steerQueued: (preview: string) => `誘導を予約しました — 次のツール呼び出しの後に届きます: 「${preview}」`,
  steerNoActiveTurn: (preview: string) => `実行中のターンがありません — 次のターンに追加しました: 「${preview}」`,
  steerRejected: '誘導は拒否されました — 実行中のターンがないため、次のターンに追加しました',
  queued: (preview: string) => `キューに追加: 「${preview}」`,
  queueFailed: 'メッセージをキューに追加できませんでした',
  countOne: (count: string) => `キュー内のメッセージ ${count} 件`,
  countOther: (count: string) => `キュー内のメッセージ ${count} 件`,
  queueEditKeptTitle: 'キュー内のメッセージは変更されていません',
  queueEditKept: '編集を保存する前に開始されたか、拒否されました。編集したテキストは入力欄にあります。',
  queueEditUnconfirmedTitle: '編集を確認できません',
  queueEditUnconfirmed:
    'Hermes は編集を確認できませんでした。もう一度保存すると同じ編集を再試行します。二重にキューに入ることはありません。'
}
