// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const jaSessionCommands = {
  undoUsage: '使い方: /undo [N]',
  undoNothing: '元に戻すものはありません',
  undidOne: (count: string) => `${count} 件のメッセージを元に戻しました — 削除したメッセージは入力欄にあります`,
  undidOther: (count: string) => `${count} 件のメッセージを元に戻しました — 削除したメッセージは入力欄にあります`,
  retryNothing: '再試行するものはありません',
  retryUnsupported: '/retry: メディアや不明なコンテンツを含むターンは再試行できません',
  steerUsage: '使い方: /steer <プロンプト>',
  steerQueued: (preview: string) => `誘導を予約しました — 次のツール呼び出しの後に届きます: 「${preview}」`,
  steerNoActiveTurn: (preview: string) => `実行中のターンがありません — 次のターンに追加しました: 「${preview}」`,
  steerRejected: '誘導は拒否されました — 実行中のターンがないため、次のターンに追加しました',
  queued: (preview: string) => `キューに追加: 「${preview}」`,
  queueFailed: 'メッセージをキューに追加できませんでした',
  countOne: (count: string) => `キュー内のメッセージ ${count} 件`,
  countOther: (count: string) => `キュー内のメッセージ ${count} 件`
}
