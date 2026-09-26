export type BrowserMessages = {
  title: string
  readOnly: string
  browse: string
  back: string
  sources: string
  choose: string
  emptySources: string
  emptyBoards: string
  refresh: string
  retry: string
  available: string
  auth: string
  unsupported: string
  unavailable: string
  removed: string
  stale: string
  never: string
  refreshed: (at: string) => string
  ownership: string
}

export const browserEn: BrowserMessages = {
  title: 'Boards by host',
  readOnly: 'Read-only',
  browse: 'Browse hosts',
  back: 'Active board',
  sources: 'Registered hosts',
  choose: 'Choose a host and board to inspect',
  emptySources: 'No registered hosts',
  emptyBoards: 'No boards reported by this source',
  refresh: 'Refresh',
  retry: 'Retry',
  available: 'Available',
  auth: 'Access denied. Check this connection’s authentication.',
  unsupported: 'Boards unavailable on this backend. The plugin may be disabled or unsupported.',
  unavailable: 'Source unavailable. Check the connection, then retry.',
  removed: 'The selected source or board is no longer registered.',
  stale: 'Stale: last successful snapshot, not current state.',
  never: 'Not refreshed yet',
  refreshed: at => `Last success: ${at}`,
  ownership: 'Boards stay on their owning host. Sources may expose the same physical board; they are not merged.'
}

export const browserJa: BrowserMessages = {
  title: 'ホスト別ボード',
  readOnly: '読み取り専用',
  browse: 'ホストを参照',
  back: 'アクティブなボード',
  sources: '登録済みホスト',
  choose: 'ホストとボードを選択してください',
  emptySources: '登録済みホストはありません',
  emptyBoards: 'このソースにはボードがありません',
  refresh: '更新',
  retry: '再試行',
  available: '利用可能',
  auth: 'アクセスが拒否されました。接続の認証を確認してください。',
  unsupported: 'このバックエンドではボードを利用できません。プラグインが無効または非対応の可能性があります。',
  unavailable: 'ソースを利用できません。接続を確認して再試行してください。',
  removed: '選択したソースまたはボードは登録されていません。',
  stale: '古いデータ：最後に取得した状態で、現在の状態ではありません。',
  never: '未更新',
  refreshed: at => `最終取得成功：${at}`,
  ownership: 'ボードは各ホストが所有します。同じ実体を公開するソースもありますが、統合はしません。'
}

export const browserZh: BrowserMessages = {
  title: '按主机浏览看板',
  readOnly: '只读',
  browse: '浏览主机',
  back: '当前看板',
  sources: '已注册主机',
  choose: '选择主机和看板以查看详情',
  emptySources: '没有已注册的主机',
  emptyBoards: '此来源未报告看板',
  refresh: '刷新',
  retry: '重试',
  available: '可用',
  auth: '访问被拒绝。请检查此连接的身份验证。',
  unsupported: '此后端无法提供看板。插件可能已禁用或不受支持。',
  unavailable: '来源不可用。请检查连接后重试。',
  removed: '所选来源或看板已不再注册。',
  stale: '数据已过时：显示最后成功获取的快照，而非当前状态。',
  never: '尚未刷新',
  refreshed: at => `上次成功：${at}`,
  ownership: '看板由各自主机管理。不同来源可能提供同一个实体看板，但不会合并。'
}

export const browserZhHant: BrowserMessages = {
  title: '依主機瀏覽看板',
  readOnly: '唯讀',
  browse: '瀏覽主機',
  back: '目前看板',
  sources: '已註冊主機',
  choose: '選擇主機與看板以查看詳情',
  emptySources: '沒有已註冊的主機',
  emptyBoards: '此來源未回報看板',
  refresh: '重新整理',
  retry: '重試',
  available: '可用',
  auth: '存取遭拒。請檢查此連線的驗證。',
  unsupported: '此後端無法提供看板。外掛可能已停用或不受支援。',
  unavailable: '來源無法使用。請檢查連線後重試。',
  removed: '所選來源或看板已不再註冊。',
  stale: '資料已過時：顯示最後成功取得的快照，而非目前狀態。',
  never: '尚未重新整理',
  refreshed: at => `上次成功：${at}`,
  ownership: '看板由各自主機管理。不同來源可能提供同一個實體看板，但不會合併。'
}
