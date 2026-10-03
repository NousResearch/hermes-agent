import type { TranslationOverrides } from './define-locale'

export const zhHantProjects = {
  projects: {
    search: '搜尋專案...',
    refresh: '重新整理專案',
    refreshing: '正在重新整理專案',
    loading: '正在載入專案',
    emptyTitle: '還沒有專案',
    emptyDesc: '建立專案，把它的資料夾、儲存庫和工作階段集中在一起。',
    noMatchesTitle: '沒有符合的專案',
    unavailableTitle: '無法使用專案',
    unavailableDesc: '目前連線的 Hermes 後端尚不支援專案。請更新 Hermes 後使用。',
    loadFailedTitle: '無法載入專案',
    loadFailedDesc: 'Hermes 未傳回專案清單。請檢查連線後再試一次。',
    partialFailed: '未能重新整理所有專案詳細資料，正在顯示上次載入的內容。',
    incompleteProfiles: '部分設定檔無法讀取，因此此清單可能不完整。',
    selectTitle: '選擇一個專案',
    selectDesc: '選擇專案以檢視其資料夾、儲存庫和工作階段。',
    autoDiscovered: '自動探索的儲存庫',
    sessionCount: count => (count === 1 ? '1 個工作階段' : `${count} 個工作階段`),
    primaryPath: '主要資料夾',
    noPath: '沒有資料夾',
    folders: '資料夾',
    primaryFolder: '主要',
    repositories: '儲存庫',
    noRepositories: '此專案中還沒有 git 儲存庫。',
    laneMain: '主要簽出',
    laneWorktree: '工作樹',
    laneKanban: '看板任務工作樹',
    activeSessions: '正在進行',
    noActiveSessions: '目前沒有代理在此專案中工作。',
    activityUnknown: '在載入此專案的所有工作階段之前，無法確認哪些代理正在工作。',
    sessions: '工作階段',
    noSessions: '此專案中還沒有工作階段。',
    sessionsFailed: '無法載入此專案的所有工作階段，僅顯示最近的工作階段。',
    allProfilesLimited:
      '「全部設定檔」無法列出單一專案的工作階段。請在側邊欄選擇一個設定檔，以查看此專案的所有工作階段。',
    untitledSession: '未命名的工作階段',
    status: {
      background: '在背景執行',
      'needs-input': '需要輸入',
      stalled: '已停滯',
      working: '運作中'
    },
    openArtifacts: '成品',
    openKanban: '看板',
    showInSidebar: '在側邊欄中顯示',
    openOverview: '開啟專案概覽'
  }
} satisfies Pick<TranslationOverrides, 'projects'>
