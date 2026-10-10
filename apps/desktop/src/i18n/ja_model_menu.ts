export const jaModelMenu = {
  search: 'モデルを検索',
  noModels: 'モデルが見つかりません',
  editModels: 'モデルを編集…',
  followDefault: '設定のデフォルトを使用',
  refreshModels: 'モデルを更新',
  favorites: 'お気に入り',
  addFavorite: 'お気に入りに追加',
  removeFavorite: 'お気に入りから削除',
  favoriteShortcut: '⇧ クリック',
  fast: '高速',
  free: '無料',
  cacheRead: 'キャッシュ読み取り',
  priceTitle: (input: string, output: string, cache: string) =>
    `入力 ${input}/Mtok · 出力 ${output}/Mtok` + (cache ? ` · キャッシュ読み取り ${cache}/Mtok` : ''),
  catalogPrice: 'models.dev カタログの定価です（このプロバイダーは価格を公開していません）',
  contextTitle: (context: string) => `コンテキストウィンドウ ${context} トークン`,
  vision: '画像入力に対応',
  maxOutputLabel: (tokens: string) => `出力 ${tokens}`,
  maxOutputTitle: (tokens: string) => `最大 ${tokens} トークンまで応答`,
  perThousandTitle: (input: string, output: string) => `1K トークンあたり: 入力 ${input} · 出力 ${output}`,
  tools: 'ツール呼び出しに対応',
  localSetup: {
    title: 'ローカルで実行 · 無料・プライベート',
    text: (model: string, size: string) => `${model} はこのマシンで動きます · ${size} をダウンロード`,
    action: '設定する'
  },
  limited: '制限中',
  limitedUntil: (time: string) => `${time} まで制限中`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} が利用上限に達しました。${time} にリセットされます。リセット後に使うモデルは今のうちに選べます。`
      : `${provider} が利用上限に達しました。リセット後に使うモデルは今のうちに選べます。`,
  modelResets: (time: string) => `${time} に再開`,
  modelLimitedTip: (time: string) =>
    `このモデルは個別の上限に達しており、${time} に再開します。ここにある他のモデルは引き続き使えます。`,
  usageLeft: (percent: number, time: null | string) =>
    time ? `残り ${percent}% · ${time} にリセット` : `残り ${percent}%`,
  poolAccounts: (count: number) => `${count} アカウント`,
  poolLimited: (limited: number, total: number) => `${total} アカウント中 ${limited} 件が制限中`,
  poolAccount: (number: number) => `アカウント ${number}`,
  poolUnknown: '使用量を取得できません',
  poolUnavailable: '再ログインしてください',
  usageTip: (provider: string) => `${provider} は利用上限に近づいています。`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label}: 残り ${percent}%、${time} にリセット` : `${label}: 残り ${percent}%`
}
