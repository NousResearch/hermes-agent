import type { ConnectionsCopy } from './connections-copy'

export const ja: ConnectionsCopy = {
  badges: { destructive: '破壊的', off: 'オフ', read: '読み取り', write: '書き込み' },
  bulk: {
    discard: '破棄',
    save: '保存',
    someOn: '一部オン',
    turnAllOff: 'すべてオフ',
    turnAllOn: 'すべてオン'
  },
  card: {
    add: '追加',
    connect: '接続',
    connected: '接続済み',
    disconnect: '切断',
    manage: '管理',
    menu: 'その他のオプション',
    notConnected: '未接続',
    reconnect: '再接続',
    refreshTools: 'ツール一覧を更新',
    remove: '削除',
    rename: '名前を変更',
    toolsOff: count => `ツール ${count} 件がオフ`,
    toolsOn: count => `ツール ${count} 件がオン`,
    toolsSomeOn: (on, total) => `ツール ${on}/${total} 件がオン`,
    waiting: 'サインイン待ち'
  },
  categories: {
    'customers-growth': {
      name: '顧客と成長',
      subtitle: 'マーケティング、営業、サポート。'
    },
    'data-analytics': {
      name: 'データと分析',
      subtitle: 'プロダクトとデータを理解する。'
    },
    'design-content': {
      name: 'デザインとコンテンツ',
      subtitle: '作成、デザイン、公開。'
    },
    engineering: {
      name: 'エンジニアリング',
      subtitle: '構築、ホスティング、監視。'
    },
    money: {
      name: 'お金',
      subtitle: '支払い、銀行、投資。'
    },
    reference: {
      name: 'リファレンス',
      subtitle: 'チャットを離れずに調べる。'
    },
    'travel-fitness': {
      name: '旅行とフィットネス',
      subtitle: '旅行の計画とトレーニングの記録。'
    },
    'work-productivity': {
      name: '仕事と生産性',
      subtitle: '書類、プロジェクト、会議、チームのアプリ。'
    }
  },
  disconnect: {
    body: name => `再接続するまで、${name} はチャットで使えなくなります。`,
    cancel: 'キャンセル',
    confirm: '切断',
    title: name => `${name} を切断しますか？`
  },
  errors: {
    connectBody: 'サインインを完了したか確認して、もう一度お試しください。',
    connectTitle: name => `${name} に接続できませんでした`,
    dismiss: '閉じる',
    loadBody: 'アプリを確認できませんでした。以前に読み込まれた内容はそのまま表示されています。',
    loadTitle: '接続を読み込めませんでした',
    tryAgain: '再試行'
  },
  facetTooltips: {
    destructive: '破壊的：データを削除したり、作業を取り消したりできます。',
    read: '読み取り：データを見るだけで、変更はできません。',
    write: '書き込み：データを作成・変更できます。'
  },
  filters: {
    activeFilters: count => `${count} 個のフィルター`,
    addMcpServer: 'MCP サーバーを追加',
    all: 'すべて',
    allCategories: 'すべてのカテゴリ',
    apply: '結果を表示',
    backToConnections: '接続',
    button: 'フィルター',
    categories: 'カテゴリ',
    categoryChipLabel: 'カテゴリ',
    chooseOne: '1つ選択',
    clear: 'すべてクリア',
    close: '閉じる',
    findCategory: 'カテゴリを検索',
    noResultsBody: 'このフィルターに一致するものはありません。クリアするとすべてのアプリが表示されます。',
    noResultsQuery: query => `「${query}」に一致するものはありません。別の検索語を試すか、フィルターをクリアしてください。`,
    noResultsTitle: '一致する接続がありません',
    resultCount: count => `${count} 件の接続`,
    searchCategory: name => `${name} を検索`,
    selectedCount: count => `${count} 件選択中`,
    showConnections: count => `${count} 件の接続を表示`,
    status: 'ステータス',
    statusAny: 'すべて',
    statusChipLabel: 'ステータス',
    statusConnected: '接続済み',
    statusNotConnected: '未接続',
    title: 'フィルター'
  },
  page: {
    subtitle: 'Hermes があなたの代わりに使えるアプリとツール。',
    title: '接続'
  },
  removeAccount: {
    body: name => `${name} はこの接続から削除されます。アプリは他のアカウントで接続されたままです。`,
    cancel: 'キャンセル',
    confirm: '削除',
    title: 'このアカウントを削除しますか？'
  },
  search: { clear: '検索をクリア', placeholder: '接続を検索' },
  sections: {
    connected: '接続済み',
    other: 'その他のアプリ',
    seeAll: count => `すべて表示（${count}）`,
    suggested: 'おすすめ'
  },
  sheet: {
    account: 'アカウント',
    accounts: 'アカウント',
    addAnAccount: '＋ アカウントを追加',
    addAnotherAccount: '＋ 別のアカウントを追加',
    availableSubtitle: 'すべてのエージェントで一時停止します。サインインは維持されます。',
    availableToAgents: 'エージェントで利用可能',
    cancelConnection: '接続をキャンセル',
    changeAfterConnect: '接続後に変更できます。',
    close: '閉じる',
    connected: '接続済み',
    connectedOn: date => `${date} に接続`,
    disconnectFailed: app => `${app} を切断できませんでした。接続を確認して、`,
    facetDescriptions: {
      destructive: '完全削除',
      read: '閲覧のみ',
      write: '作成と編集'
    },
    finishSignIn: 'サインインを完了',
    finishSignInBody: app =>
      `ブラウザで ${app} のページのアクセスを承認してから、ここに戻ってください。`,
    notConnected: '未接続',
    permissionsTooltip: app =>
      `${app} が各ツールを読み取り・書き込み・破壊的のどれにするかを決めます。オンにするものはあなたが選びます。`,
    previewAll: count => `すべてプレビュー（${count}）`,
    removeAccountBody: app =>
      `エージェントはこの ${app} アカウントを使えなくなります。他のアカウントは接続されたままです。`,
    removeAccountTitle: alias => `「${alias}」を削除しますか？`,
    removeFailed: alias => `「${alias}」を削除できませんでした。接続を確認して、`,
    removedSuccess: alias => `「${alias}」を削除しました`,
    renamePlaceholder: 'アカウント名',
    reopenSignIn: 'サインインページを再度開く',
    reviewAll: count => `すべて確認（${count}）`,
    reviewTools: 'すべてのツールを確認',
    signInBody: app => `${app} をブラウザで開きます。Hermes がパスワードを見ることはありません。`,
    signInWith: app => `${app} でサインイン`,
    startsOff: '最初からオフ',
    startsOn: '最初からオン',
    toolPermissions: 'ツールの権限',
    tools: 'ツール',
    toolsCount: count => `${count} 個のツール`,
    waiting: 'サインイン待ち',
    waitingFor: app => `${app} を待機中…`
  },
  subgroups: {
    'ai-media-presentations': 'AI・メディア・プレゼン',
    'automation-ai-apis': '自動化・AI・API',
    'bi-search-warehouses': 'BI・検索・ウェアハウス',
    'code-review-ci': 'コードレビューと CI',
    communication: 'コミュニケーション',
    'customer-support': 'カスタマーサポート',
    'design-whiteboards': 'デザインとホワイトボード',
    'docs-files': '書類とファイル',
    'docs-knowledge': '書類とナレッジ',
    'hiring-people': '採用と人材',
    'hosting-databases': 'ホスティングとデータベース',
    'marketing-seo': 'マーケティングと SEO',
    'meetings-scheduling': '会議とスケジュール',
    'monitoring-incidents': '監視とインシデント',
    'outdoors-fitness': 'アウトドアとフィットネス',
    'payments-commerce': '支払いとコマース',
    'product-analytics': 'プロダクト分析',
    'projects-tasks': 'プロジェクトとタスク',
    'sales-crm': '営業と CRM',
    'spend-investing-crypto': '支出・投資・暗号資産',
    travel: '旅行',
    'websites-assets': 'ウェブサイトと素材'
  },
  toast: {
    connected: name => `${name} を接続しました。`,
    disconnected: name => `${name} を切断しました。`,
    removed: name => `${name} を削除しました。`,
    renamed: name => `${name} に名前を変更しました。`
  },
  toolFilters: {
    all: 'すべて',
    apply: '結果を表示',
    areas: '領域',
    behavior: '動作',
    clear: 'すべてクリア',
    close: '閉じる',
    status: 'ステータス',
    title: 'ツールを絞り込む',
    uncategorised: '未分類'
  },
  tools: {
    allAreas: 'すべての領域',
    areaChipLabel: '領域',
    back: '接続に戻る',
    emptyBody: 'このアプリはツールを公開していません。',
    emptyTitle: 'ツールはまだありません',
    errorBody: 'ツール一覧を読み取れませんでした。もう一度お試しください。',
    errorTitle: 'ツールを読み込めませんでした',
    filter: 'フィルター',
    findArea: '領域を検索',
    loading: 'ツールを読み込み中',
    lockedByOrg: '組織によってオフ',
    noResultsBody: 'この検索やフィルターに一致するものはありません。',
    noResultsTitle: '一致するツールがありません',
    off: 'オフ',
    on: 'オン',
    pageSummary: (on, total, app) =>
      `${total} 個中 ${on} 個のツールがオンです。変更はまとめて保存され、${app} を使うすべてのエージェントに適用されます。`,
    pageTitle: app => `${app} のツール`,
    permissionChipLabel: '権限',
    searchAppTools: app => `${app} のツールを検索`,
    searchPlaceholder: 'ツールを検索',
    showTools: count => `${count} 個のツールを表示`,
    subtitle: name => `${name} にできることを選びます。`,
    summary: (on, total) => `${on}/${total} 件オン`,
    title: 'すべてのツール',
    toolNamePrefix: 'ツール名:',
    unsavedChanges: count => `${count} 件の未保存の変更`,
    whatItDoes: 'できること',
    whatItDoesHint: '1つ以上選択してください。「すべて」を選ぶと他の選択は解除されます。'
  }
}


export const zh: ConnectionsCopy = {
  badges: { destructive: '破坏性', off: '已关闭', read: '读取', write: '写入' },
  bulk: {
    discard: '放弃',
    save: '保存',
    someOn: '部分开启',
    turnAllOff: '全部关闭',
    turnAllOn: '全部开启'
  },
  card: {
    add: '添加',
    connect: '连接',
    connected: '已连接',
    disconnect: '断开连接',
    manage: '管理',
    menu: '更多选项',
    notConnected: '未连接',
    reconnect: '重新连接',
    refreshTools: '刷新工具列表',
    remove: '移除',
    rename: '重命名',
    toolsOff: count => `${count} 个工具已关闭`,
    toolsOn: count => `${count} 个工具已开启`,
    toolsSomeOn: (on, total) => `${total} 个工具中有 ${on} 个已开启`,
    waiting: '等待登录'
  },
  categories: {
    'customers-growth': {
      name: '客户与增长',
      subtitle: '营销、销售和支持。'
    },
    'data-analytics': {
      name: '数据与分析',
      subtitle: '了解你的产品和数据。'
    },
    'design-content': {
      name: '设计与内容',
      subtitle: '创作、设计并发布。'
    },
    engineering: {
      name: '工程',
      subtitle: '构建、托管和监控。'
    },
    money: {
      name: '财务',
      subtitle: '支付、银行和投资。'
    },
    reference: {
      name: '参考',
      subtitle: '无需离开对话即可查阅。'
    },
    'travel-fitness': {
      name: '旅行与健身',
      subtitle: '规划行程、记录训练。'
    },
    'work-productivity': {
      name: '工作与效率',
      subtitle: '文档、项目、会议，以及团队日常使用的应用。'
    }
  },
  disconnect: {
    body: name => `在你重新连接之前，${name} 将无法在对话中使用。`,
    cancel: '取消',
    confirm: '断开连接',
    title: name => `断开 ${name}？`
  },
  errors: {
    connectBody: '请确认已完成登录，然后重试。',
    connectTitle: name => `无法连接 ${name}`,
    dismiss: '关闭',
    loadBody: '无法检查你的应用。之前加载的内容仍会保留。',
    loadTitle: '无法加载连接',
    tryAgain: '重试'
  },
  facetTooltips: {
    destructive: '破坏性：可以删除数据或撤销操作。',
    read: '读取：可以查看你的数据，但不能更改。',
    write: '写入：可以创建和更改数据。'
  },
  filters: {
    activeFilters: count => `${count} 个筛选条件`,
    addMcpServer: '添加 MCP 服务器',
    all: '全部',
    allCategories: '所有类别',
    apply: '显示结果',
    backToConnections: '连接',
    button: '筛选',
    categories: '类别',
    categoryChipLabel: '类别',
    chooseOne: '选择一项',
    clear: '清除全部',
    close: '关闭',
    findCategory: '查找类别',
    noResultsBody: '没有符合这些筛选条件的内容。清除筛选即可查看所有应用。',
    noResultsQuery: query => `没有与“${query}”匹配的内容。请尝试其他搜索，或清除筛选条件。`,
    noResultsTitle: '没有匹配的连接',
    resultCount: count => `${count} 个连接`,
    searchCategory: name => `搜索 ${name}`,
    selectedCount: count => `已选择 ${count} 项`,
    showConnections: count => `显示 ${count} 个连接`,
    status: '状态',
    statusAny: '全部',
    statusChipLabel: '状态',
    statusConnected: '已连接',
    statusNotConnected: '未连接',
    title: '筛选'
  },
  page: {
    subtitle: 'Hermes 可以为你使用的应用和工具。',
    title: '连接'
  },
  removeAccount: {
    body: name => `${name} 将从此连接中移除。应用会继续通过你的其他账户保持连接。`,
    cancel: '取消',
    confirm: '移除',
    title: '移除此账户？'
  },
  search: { clear: '清除搜索', placeholder: '搜索连接' },
  sections: {
    connected: '已连接',
    other: '其他应用',
    seeAll: count => `查看全部 ${count} 个`,
    suggested: '为你推荐'
  },
  sheet: {
    account: '账户',
    accounts: '账户',
    addAnAccount: '+ 添加账户',
    addAnotherAccount: '+ 添加其他账户',
    availableSubtitle: '对所有智能体暂停；保持登录状态。',
    availableToAgents: '可供所有智能体使用',
    cancelConnection: '取消连接',
    changeAfterConnect: '连接后可以更改这些设置。',
    close: '关闭',
    connected: '已连接',
    connectedOn: date => `连接于 ${date}`,
    disconnectFailed: app => `无法断开 ${app}。请检查连接并 `,
    facetDescriptions: {
      destructive: '永久删除',
      read: '仅查看',
      write: '创建和编辑'
    },
    finishSignIn: '完成登录',
    finishSignInBody: app => `在浏览器中的 ${app} 页面批准访问，然后回到这里。`,
    notConnected: '未连接',
    permissionsTooltip: app =>
      `${app} 决定每个工具是读取、写入还是破坏性。你选择开启哪些。`,
    previewAll: count => `预览全部 ${count} 个`,
    removeAccountBody: app => `智能体将无法再使用此 ${app} 账户。其他账户仍保持连接。`,
    removeAccountTitle: alias => `移除“${alias}”？`,
    removeFailed: alias => `无法移除“${alias}”。请检查连接并 `,
    removedSuccess: alias => `“${alias}”已成功移除`,
    renamePlaceholder: '账户名称',
    reopenSignIn: '重新打开登录页面',
    reviewAll: count => `查看全部 ${count} 个`,
    reviewTools: '查看所有工具',
    signInBody: app => `在浏览器中打开 ${app}。Hermes 永远不会看到你的密码。`,
    signInWith: app => `使用 ${app} 登录`,
    startsOff: '默认关闭',
    startsOn: '默认开启',
    toolPermissions: '工具权限',
    tools: '工具',
    toolsCount: count => `${count} 个工具`,
    waiting: '等待登录',
    waitingFor: app => `正在等待 ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'AI、媒体与演示',
    'automation-ai-apis': '自动化、AI 与 API',
    'bi-search-warehouses': 'BI、搜索与数据仓库',
    'code-review-ci': '代码审查与 CI',
    communication: '沟通',
    'customer-support': '客户支持',
    'design-whiteboards': '设计与白板',
    'docs-files': '文档与文件',
    'docs-knowledge': '文档与知识',
    'hiring-people': '招聘与人员',
    'hosting-databases': '托管与数据库',
    'marketing-seo': '营销与 SEO',
    'meetings-scheduling': '会议与日程',
    'monitoring-incidents': '监控与事件',
    'outdoors-fitness': '户外与健身',
    'payments-commerce': '支付与商务',
    'product-analytics': '产品分析',
    'projects-tasks': '项目与任务',
    'sales-crm': '销售与 CRM',
    'spend-investing-crypto': '支出、投资与加密货币',
    travel: '旅行',
    'websites-assets': '网站与素材'
  },
  toast: {
    connected: name => `${name} 已连接。`,
    disconnected: name => `${name} 已断开连接。`,
    removed: name => `已移除 ${name}。`,
    renamed: name => `已重命名为 ${name}。`
  },
  toolFilters: {
    all: '全部',
    apply: '显示结果',
    areas: '领域',
    behavior: '行为',
    clear: '清除全部',
    close: '关闭',
    status: '状态',
    title: '筛选工具',
    uncategorised: '未分类'
  },
  tools: {
    allAreas: '所有领域',
    areaChipLabel: '领域',
    back: '返回连接',
    emptyBody: '此应用尚未提供任何工具。',
    emptyTitle: '还没有工具',
    errorBody: '无法读取工具列表。请重试。',
    errorTitle: '无法加载工具',
    filter: '筛选',
    findArea: '查找领域',
    loading: '正在加载工具',
    lockedByOrg: '已被你的组织关闭',
    noResultsBody: '没有符合此搜索或筛选条件的内容。',
    noResultsTitle: '没有匹配的工具',
    off: '已关闭',
    on: '已开启',
    pageSummary: (on, total, app) =>
      `${total} 个工具中有 ${on} 个已开启。更改会一起保存，并应用于每个使用 ${app} 的智能体。`,
    pageTitle: app => `${app} 工具`,
    permissionChipLabel: '权限',
    searchAppTools: app => `搜索 ${app} 工具`,
    searchPlaceholder: '搜索工具',
    showTools: count => `显示 ${count} 个工具`,
    subtitle: name => `选择 ${name} 可以做什么。`,
    summary: (on, total) => `${total} 个中有 ${on} 个已开启`,
    title: '所有工具',
    toolNamePrefix: '工具名称：',
    unsavedChanges: count => `${count} 项未保存的更改`,
    whatItDoes: '功能',
    whatItDoesHint: '选择一个或多个。“全部”会清除其他选择。'
  }
}


export const zhHant: ConnectionsCopy = {
  badges: { destructive: '破壞性', off: '已關閉', read: '讀取', write: '寫入' },
  bulk: {
    discard: '捨棄',
    save: '儲存',
    someOn: '部分開啟',
    turnAllOff: '全部關閉',
    turnAllOn: '全部開啟'
  },
  card: {
    add: '新增',
    connect: '連線',
    connected: '已連線',
    disconnect: '中斷連線',
    manage: '管理',
    menu: '更多選項',
    notConnected: '未連線',
    reconnect: '重新連線',
    refreshTools: '重新整理工具清單',
    remove: '移除',
    rename: '重新命名',
    toolsOff: count => `${count} 個工具已關閉`,
    toolsOn: count => `${count} 個工具已開啟`,
    toolsSomeOn: (on, total) => `${total} 個工具中有 ${on} 個已開啟`,
    waiting: '等待登入'
  },
  categories: {
    'customers-growth': {
      name: '客戶與成長',
      subtitle: '行銷、銷售與支援。'
    },
    'data-analytics': {
      name: '資料與分析',
      subtitle: '了解你的產品與資料。'
    },
    'design-content': {
      name: '設計與內容',
      subtitle: '創作、設計並發布。'
    },
    engineering: {
      name: '工程',
      subtitle: '建置、託管與監控。'
    },
    money: {
      name: '財務',
      subtitle: '付款、銀行與投資。'
    },
    reference: {
      name: '參考',
      subtitle: '不必離開對話即可查閱。'
    },
    'travel-fitness': {
      name: '旅行與健身',
      subtitle: '規劃行程、記錄訓練。'
    },
    'work-productivity': {
      name: '工作與效率',
      subtitle: '文件、專案、會議，以及團隊日常使用的應用程式。'
    }
  },
  disconnect: {
    body: name => `在你重新連線之前，${name} 將無法在對話中使用。`,
    cancel: '取消',
    confirm: '中斷連線',
    title: name => `中斷 ${name} 的連線？`
  },
  errors: {
    connectBody: '請確認已完成登入，然後再試一次。',
    connectTitle: name => `無法連線至 ${name}`,
    dismiss: '關閉',
    loadBody: '無法檢查你的應用程式。先前載入的內容仍會保留。',
    loadTitle: '無法載入連線',
    tryAgain: '重試'
  },
  facetTooltips: {
    destructive: '破壞性：可以刪除資料或復原操作。',
    read: '讀取：可以查看你的資料，但不能變更。',
    write: '寫入：可以建立與變更資料。'
  },
  filters: {
    activeFilters: count => `${count} 個篩選條件`,
    addMcpServer: '新增 MCP 伺服器',
    all: '全部',
    allCategories: '所有類別',
    apply: '顯示結果',
    backToConnections: '連線',
    button: '篩選',
    categories: '類別',
    categoryChipLabel: '類別',
    chooseOne: '選擇一項',
    clear: '清除全部',
    close: '關閉',
    findCategory: '尋找類別',
    noResultsBody: '沒有符合這些篩選條件的內容。清除篩選即可查看所有應用程式。',
    noResultsQuery: query => `沒有符合「${query}」的內容。請嘗試其他搜尋，或清除篩選條件。`,
    noResultsTitle: '沒有符合的連線',
    resultCount: count => `${count} 個連線`,
    searchCategory: name => `搜尋 ${name}`,
    selectedCount: count => `已選擇 ${count} 項`,
    showConnections: count => `顯示 ${count} 個連線`,
    status: '狀態',
    statusAny: '全部',
    statusChipLabel: '狀態',
    statusConnected: '已連線',
    statusNotConnected: '未連線',
    title: '篩選'
  },
  page: {
    subtitle: 'Hermes 可以為你使用的應用程式與工具。',
    title: '連線'
  },
  removeAccount: {
    body: name => `${name} 將從此連線中移除。應用程式會繼續透過你的其他帳戶保持連線。`,
    cancel: '取消',
    confirm: '移除',
    title: '移除此帳戶？'
  },
  search: { clear: '清除搜尋', placeholder: '搜尋連線' },
  sections: {
    connected: '已連線',
    other: '其他應用程式',
    seeAll: count => `查看全部 ${count} 個`,
    suggested: '為你推薦'
  },
  sheet: {
    account: '帳戶',
    accounts: '帳戶',
    addAnAccount: '＋ 新增帳戶',
    addAnotherAccount: '＋ 新增其他帳戶',
    availableSubtitle: '對所有智慧代理暫停；保持登入狀態。',
    availableToAgents: '可供所有智慧代理使用',
    cancelConnection: '取消連線',
    changeAfterConnect: '連線後可以變更這些設定。',
    close: '關閉',
    connected: '已連線',
    connectedOn: date => `已於 ${date} 連線`,
    disconnectFailed: app => `無法中斷 ${app} 的連線。請檢查你的連線並 `,
    facetDescriptions: {
      destructive: '永久刪除',
      read: '僅檢視',
      write: '建立與編輯'
    },
    finishSignIn: '完成登入',
    finishSignInBody: app => `在瀏覽器中的 ${app} 頁面核准存取，然後回到這裡。`,
    notConnected: '未連線',
    permissionsTooltip: app =>
      `${app} 決定每個工具是讀取、寫入還是破壞性。你選擇要開啟哪些。`,
    previewAll: count => `預覽全部 ${count} 個`,
    removeAccountBody: app => `智慧代理將無法再使用此 ${app} 帳戶。其他帳戶仍保持連線。`,
    removeAccountTitle: alias => `移除「${alias}」？`,
    removeFailed: alias => `無法移除「${alias}」。請檢查你的連線並 `,
    removedSuccess: alias => `已成功移除「${alias}」`,
    renamePlaceholder: '帳戶名稱',
    reopenSignIn: '重新開啟登入頁面',
    reviewAll: count => `檢視全部 ${count} 個`,
    reviewTools: '檢視所有工具',
    signInBody: app => `在瀏覽器中開啟 ${app}。Hermes 永遠不會看到你的密碼。`,
    signInWith: app => `使用 ${app} 登入`,
    startsOff: '預設關閉',
    startsOn: '預設開啟',
    toolPermissions: '工具權限',
    tools: '工具',
    toolsCount: count => `${count} 個工具`,
    waiting: '等待登入',
    waitingFor: app => `正在等待 ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'AI、媒體與簡報',
    'automation-ai-apis': '自動化、AI 與 API',
    'bi-search-warehouses': 'BI、搜尋與資料倉儲',
    'code-review-ci': '程式碼審查與 CI',
    communication: '溝通',
    'customer-support': '客戶支援',
    'design-whiteboards': '設計與白板',
    'docs-files': '文件與檔案',
    'docs-knowledge': '文件與知識',
    'hiring-people': '招募與人員',
    'hosting-databases': '託管與資料庫',
    'marketing-seo': '行銷與 SEO',
    'meetings-scheduling': '會議與排程',
    'monitoring-incidents': '監控與事件',
    'outdoors-fitness': '戶外與健身',
    'payments-commerce': '付款與商務',
    'product-analytics': '產品分析',
    'projects-tasks': '專案與任務',
    'sales-crm': '銷售與 CRM',
    'spend-investing-crypto': '支出、投資與加密貨幣',
    travel: '旅行',
    'websites-assets': '網站與素材'
  },
  toast: {
    connected: name => `${name} 已連線。`,
    disconnected: name => `${name} 已中斷連線。`,
    removed: name => `已移除 ${name}。`,
    renamed: name => `已重新命名為 ${name}。`
  },
  toolFilters: {
    all: '全部',
    apply: '顯示結果',
    areas: '領域',
    behavior: '行為',
    clear: '清除全部',
    close: '關閉',
    status: '狀態',
    title: '篩選工具',
    uncategorised: '未分類'
  },
  tools: {
    allAreas: '所有領域',
    areaChipLabel: '領域',
    back: '返回連線',
    emptyBody: '此應用程式尚未提供任何工具。',
    emptyTitle: '尚無工具',
    errorBody: '無法讀取工具清單。請再試一次。',
    errorTitle: '無法載入工具',
    filter: '篩選',
    findArea: '尋找領域',
    loading: '正在載入工具',
    lockedByOrg: '已被你的組織關閉',
    noResultsBody: '沒有符合此搜尋或篩選條件的內容。',
    noResultsTitle: '沒有符合的工具',
    off: '已關閉',
    on: '已開啟',
    pageSummary: (on, total, app) =>
      `${total} 個工具中有 ${on} 個已開啟。變更會一起儲存，並套用至每個使用 ${app} 的智慧代理。`,
    pageTitle: app => `${app} 工具`,
    permissionChipLabel: '權限',
    searchAppTools: app => `搜尋 ${app} 工具`,
    searchPlaceholder: '搜尋工具',
    showTools: count => `顯示 ${count} 個工具`,
    subtitle: name => `選擇 ${name} 可以做什麼。`,
    summary: (on, total) => `${total} 個中有 ${on} 個已開啟`,
    title: '所有工具',
    toolNamePrefix: '工具名稱：',
    unsavedChanges: count => `${count} 項未儲存的變更`,
    whatItDoes: '功能',
    whatItDoesHint: '選擇一或多個。「全部」會清除其他選擇。'
  }
}


function ruPlural(count: number, one: string, few: string, many: string): string {
  const mod10 = count % 10
  const mod100 = count % 100

  if (mod10 === 1 && mod100 !== 11) {
    return one
  }

  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) {
    return few
  }

  return many
}


export const ru: ConnectionsCopy = {
  badges: { destructive: 'Разрушительное', off: 'Выключено', read: 'Чтение', write: 'Запись' },
  bulk: {
    discard: 'Отменить',
    save: 'Сохранить',
    someOn: 'Часть включена',
    turnAllOff: 'Выключить все',
    turnAllOn: 'Включить все'
  },
  card: {
    add: 'Добавить',
    connect: 'Подключить',
    connected: 'Подключено',
    disconnect: 'Отключить',
    manage: 'Управлять',
    menu: 'Другие действия',
    notConnected: 'Не подключено',
    reconnect: 'Переподключить',
    refreshTools: 'Обновить список инструментов',
    remove: 'Удалить',
    rename: 'Переименовать',
    toolsOff: count =>
      `${count} ${ruPlural(count, 'инструмент', 'инструмента', 'инструментов')} выключено`,
    toolsOn: count => `${count} ${ruPlural(count, 'инструмент', 'инструмента', 'инструментов')} включено`,
    toolsSomeOn: (on, total) => `${on} из ${total} инструментов включено`,
    waiting: 'Ожидание входа'
  },
  categories: {
    'customers-growth': {
      name: 'Клиенты и рост',
      subtitle: 'Маркетинг, продажи и поддержка.'
    },
    'data-analytics': {
      name: 'Данные и аналитика',
      subtitle: 'Понимайте свой продукт и свои данные.'
    },
    'design-content': {
      name: 'Дизайн и контент',
      subtitle: 'Создавайте, оформляйте и публикуйте.'
    },
    engineering: {
      name: 'Инженерия',
      subtitle: 'Создавайте, размещайте и отслеживайте.'
    },
    money: {
      name: 'Деньги',
      subtitle: 'Платежи, банки и инвестиции.'
    },
    reference: {
      name: 'Справочник',
      subtitle: 'Находите ответы, не выходя из чата.'
    },
    'travel-fitness': {
      name: 'Путешествия и спорт',
      subtitle: 'Планируйте поездки и следите за тренировками.'
    },
    'work-productivity': {
      name: 'Работа и продуктивность',
      subtitle: 'Документы, проекты, встречи и приложения вашей команды.'
    }
  },
  disconnect: {
    body: name => `${name} перестанет работать в чате, пока вы снова его не подключите.`,
    cancel: 'Отмена',
    confirm: 'Отключить',
    title: name => `Отключить ${name}?`
  },
  errors: {
    connectBody: 'Убедитесь, что вход завершён, и повторите попытку.',
    connectTitle: name => `Не удалось подключить ${name}`,
    dismiss: 'Закрыть',
    loadBody: 'Не удалось проверить приложения. Загруженное ранее остаётся на месте.',
    loadTitle: 'Не удалось загрузить подключения',
    tryAgain: 'Повторить'
  },
  facetTooltips: {
    destructive: 'Разрушительное: может удалять данные или отменять работу.',
    read: 'Чтение: может просматривать ваши данные, но не изменять их.',
    write: 'Запись: может создавать и изменять данные.'
  },
  filters: {
    activeFilters: count => `${count} ${ruPlural(count, 'фильтр', 'фильтра', 'фильтров')}`,
    addMcpServer: 'Добавить сервер MCP',
    all: 'Все',
    allCategories: 'Все категории',
    apply: 'Показать результаты',
    backToConnections: 'Подключения',
    button: 'Фильтры',
    categories: 'Категории',
    categoryChipLabel: 'Категория',
    chooseOne: 'Выберите один',
    clear: 'Очистить всё',
    close: 'Закрыть',
    findCategory: 'Найти категорию',
    noResultsBody: 'Ничего не соответствует этим фильтрам. Очистите их, чтобы увидеть все приложения.',
    noResultsQuery: query =>
      `Ничего не соответствует «${query}». Попробуйте другой запрос или очистите фильтры.`,
    noResultsTitle: 'Нет подходящих подключений',
    resultCount: count => `${count} ${ruPlural(count, 'подключение', 'подключения', 'подключений')}`,
    searchCategory: name => `Поиск ${name}`,
    selectedCount: count => `Выбрано: ${count}`,
    showConnections: count =>
      `Показать ${count} ${ruPlural(count, 'подключение', 'подключения', 'подключений')}`,
    status: 'Статус',
    statusAny: 'Все',
    statusChipLabel: 'Статус',
    statusConnected: 'Подключено',
    statusNotConnected: 'Не подключено',
    title: 'Фильтры'
  },
  page: {
    subtitle: 'Приложения и инструменты, которые Hermes может использовать за вас.',
    title: 'Подключения'
  },
  removeAccount: {
    body: name => `${name} будет удалён из этого подключения. Приложение останется подключённым через другие аккаунты.`,
    cancel: 'Отмена',
    confirm: 'Удалить',
    title: 'Удалить этот аккаунт?'
  },
  search: { clear: 'Очистить поиск', placeholder: 'Поиск подключений' },
  sections: {
    connected: 'Подключено',
    other: 'Другие приложения',
    seeAll: count => `Показать все (${count})`,
    suggested: 'Рекомендуем вам'
  },
  sheet: {
    account: 'Аккаунт',
    accounts: 'Аккаунты',
    addAnAccount: '+ Добавить аккаунт',
    addAnotherAccount: '+ Добавить другой аккаунт',
    availableSubtitle: 'Пауза для всех агентов; вход сохраняется.',
    availableToAgents: 'Доступно агентам',
    cancelConnection: 'Отменить подключение',
    changeAfterConnect: 'Их можно изменить после подключения.',
    close: 'Закрыть',
    connected: 'Подключено',
    connectedOn: date => `Подключено ${date}`,
    disconnectFailed: app => `Не удалось отключить ${app}. Проверьте подключение и `,
    facetDescriptions: {
      destructive: 'Безвозвратное удаление',
      read: 'Только просмотр',
      write: 'Создание и изменение'
    },
    finishSignIn: 'Завершить вход',
    finishSignInBody: app => `Подтвердите доступ на странице ${app} в браузере, затем вернитесь сюда.`,
    notConnected: 'Не подключено',
    permissionsTooltip: app =>
      `${app} решает, будет инструмент на чтение, запись или разрушительным. Вы выбираете, какие включены.`,
    previewAll: count => `Просмотреть все (${count})`,
    removeAccountBody: app =>
      `Агенты больше не смогут использовать этот аккаунт ${app}. Остальные аккаунты останутся подключены.`,
    removeAccountTitle: alias => `Удалить «${alias}»?`,
    removeFailed: alias => `Не удалось удалить «${alias}». Проверьте подключение и `,
    removedSuccess: alias => `«${alias}» успешно удалён`,
    renamePlaceholder: 'Название аккаунта',
    reopenSignIn: 'Снова открыть страницу входа',
    reviewAll: count => `Проверить все (${count})`,
    reviewTools: 'Проверить все инструменты',
    signInBody: app => `Откроет ${app} в браузере. Hermes никогда не видит ваш пароль.`,
    signInWith: app => `Войти через ${app}`,
    startsOff: 'Выключено изначально',
    startsOn: 'Включено изначально',
    toolPermissions: 'Разрешения инструментов',
    tools: 'Инструменты',
    toolsCount: count => `${count} ${ruPlural(count, 'инструмент', 'инструмента', 'инструментов')}`,
    waiting: 'Ожидание входа',
    waitingFor: app => `Ожидание ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'ИИ, медиа и презентации',
    'automation-ai-apis': 'Автоматизация, ИИ и API',
    'bi-search-warehouses': 'BI, поиск и хранилища',
    'code-review-ci': 'Проверка кода и CI',
    communication: 'Общение',
    'customer-support': 'Поддержка клиентов',
    'design-whiteboards': 'Дизайн и доски',
    'docs-files': 'Документы и файлы',
    'docs-knowledge': 'Документы и знания',
    'hiring-people': 'Найм и люди',
    'hosting-databases': 'Хостинг и базы данных',
    'marketing-seo': 'Маркетинг и SEO',
    'meetings-scheduling': 'Встречи и расписание',
    'monitoring-incidents': 'Мониторинг и инциденты',
    'outdoors-fitness': 'Активный отдых и фитнес',
    'payments-commerce': 'Платежи и торговля',
    'product-analytics': 'Продуктовая аналитика',
    'projects-tasks': 'Проекты и задачи',
    'sales-crm': 'Продажи и CRM',
    'spend-investing-crypto': 'Расходы, инвестиции и крипто',
    travel: 'Путешествия',
    'websites-assets': 'Сайты и ресурсы'
  },
  toast: {
    connected: name => `${name} подключено.`,
    disconnected: name => `${name} отключено.`,
    removed: name => `${name} удалено.`,
    renamed: name => `Переименовано в ${name}.`
  },
  toolFilters: {
    all: 'Все',
    apply: 'Показать результаты',
    areas: 'Области',
    behavior: 'Поведение',
    clear: 'Очистить всё',
    close: 'Закрыть',
    status: 'Статус',
    title: 'Фильтр инструментов',
    uncategorised: 'Без категории'
  },
  tools: {
    allAreas: 'Все области',
    areaChipLabel: 'Область',
    back: 'Назад к подключениям',
    emptyBody: 'Это приложение не предоставило инструментов.',
    emptyTitle: 'Инструментов пока нет',
    errorBody: 'Не удалось прочитать список инструментов. Повторите попытку.',
    errorTitle: 'Не удалось загрузить инструменты',
    filter: 'Фильтры',
    findArea: 'Найти область',
    loading: 'Загрузка инструментов',
    lockedByOrg: 'Отключено вашей организацией',
    noResultsBody: 'Ничего не соответствует этому поиску или фильтрам.',
    noResultsTitle: 'Нет подходящих инструментов',
    off: 'Выключено',
    on: 'Включено',
    pageSummary: (on, total, app) =>
      `${on} из ${total} инструментов включено. Изменения сохраняются вместе и применяются ко всем агентам, использующим ${app}.`,
    pageTitle: app => `Инструменты ${app}`,
    permissionChipLabel: 'Разрешение',
    searchAppTools: app => `Поиск инструментов ${app}`,
    searchPlaceholder: 'Поиск инструментов',
    showTools: count =>
      `Показать ${count} ${ruPlural(count, 'инструмент', 'инструмента', 'инструментов')}`,
    subtitle: name => `Выберите, что может делать ${name}.`,
    summary: (on, total) => `${on} из ${total} включено`,
    title: 'Все инструменты',
    toolNamePrefix: 'Название инструмента:',
    unsavedChanges: count =>
      `${count} ${ruPlural(count, 'несохранённое изменение', 'несохранённых изменения', 'несохранённых изменений')}`,
    whatItDoes: 'Что делает',
    whatItDoesHint: 'Выберите один или несколько. «Все» сбрасывает остальные варианты.'
  }
}


export const ar: ConnectionsCopy = {
  badges: { destructive: 'تدميرية', off: 'متوقفة', read: 'قراءة', write: 'كتابة' },
  bulk: {
    discard: 'تجاهل',
    save: 'حفظ',
    someOn: 'بعضها مفعّل',
    turnAllOff: 'إيقاف الكل',
    turnAllOn: 'تشغيل الكل'
  },
  card: {
    add: 'إضافة',
    connect: 'اتصال',
    connected: 'متصل',
    disconnect: 'قطع الاتصال',
    manage: 'إدارة',
    menu: 'خيارات أخرى',
    notConnected: 'غير متصل',
    reconnect: 'إعادة الاتصال',
    refreshTools: 'تحديث قائمة الأدوات',
    remove: 'إزالة',
    rename: 'إعادة التسمية',
    toolsOff: count => `${count} أداة متوقفة`,
    toolsOn: count => `${count} أداة مفعّلة`,
    toolsSomeOn: (on, total) => `${on} من ${total} أداة مفعّلة`,
    waiting: 'بانتظار تسجيل الدخول'
  },
  categories: {
    'customers-growth': {
      name: 'العملاء والنمو',
      subtitle: 'التسويق والمبيعات والدعم.'
    },
    'data-analytics': {
      name: 'البيانات والتحليلات',
      subtitle: 'افهم منتجك وبياناتك.'
    },
    'design-content': {
      name: 'التصميم والمحتوى',
      subtitle: 'أنشئ وصمّم وانشر.'
    },
    engineering: {
      name: 'الهندسة',
      subtitle: 'ابنِ واستضف وراقب.'
    },
    money: {
      name: 'المال',
      subtitle: 'المدفوعات والخدمات المصرفية والاستثمار.'
    },
    reference: {
      name: 'المراجع',
      subtitle: 'ابحث عن إجابات دون مغادرة المحادثة.'
    },
    'travel-fitness': {
      name: 'السفر واللياقة',
      subtitle: 'خطّط للرحلات وتابع تدريبك.'
    },
    'work-productivity': {
      name: 'العمل والإنتاجية',
      subtitle: 'المستندات والمشاريع والاجتماعات وتطبيقات فريقك.'
    }
  },
  disconnect: {
    body: name => `سيتوقف ${name} عن العمل في المحادثة حتى تعيد الاتصال به.`,
    cancel: 'إلغاء',
    confirm: 'قطع الاتصال',
    title: name => `قطع اتصال ${name}؟`
  },
  errors: {
    connectBody: 'تأكد من إكمال تسجيل الدخول ثم أعد المحاولة.',
    connectTitle: name => `تعذّر اتصال ${name}`,
    dismiss: 'إغلاق',
    loadBody: 'تعذّر فحص تطبيقاتك. ما حُمّل سابقًا ما زال هنا.',
    loadTitle: 'تعذّر تحميل الاتصالات',
    tryAgain: 'إعادة المحاولة'
  },
  facetTooltips: {
    destructive: 'تدميرية: يمكنها حذف البيانات أو التراجع عن العمل.',
    read: 'قراءة: يمكنها الاطلاع على بياناتك دون تغييرها.',
    write: 'كتابة: يمكنها إنشاء البيانات وتغييرها.'
  },
  filters: {
    activeFilters: count => `${count} عامل تصفية`,
    addMcpServer: 'إضافة خادم MCP',
    all: 'الكل',
    allCategories: 'كل الفئات',
    apply: 'عرض النتائج',
    backToConnections: 'الاتصالات',
    button: 'عوامل التصفية',
    categories: 'الفئات',
    categoryChipLabel: 'الفئة',
    chooseOne: 'اختر واحدًا',
    clear: 'مسح الكل',
    close: 'إغلاق',
    findCategory: 'ابحث عن فئة',
    noResultsBody: 'لا شيء يطابق عوامل التصفية هذه. امسحها لعرض كل التطبيقات.',
    noResultsQuery: query => `لا شيء يطابق «${query}». جرّب بحثًا آخر أو امسح عوامل التصفية.`,
    noResultsTitle: 'لا توجد اتصالات مطابقة',
    resultCount: count => `${count} ${count === 1 ? 'اتصال' : 'اتصالات'}`,
    searchCategory: name => `ابحث في ${name}`,
    selectedCount: count => `تم تحديد ${count}`,
    showConnections: count => `عرض ${count} ${count === 1 ? 'اتصال' : 'اتصالات'}`,
    status: 'الحالة',
    statusAny: 'الكل',
    statusChipLabel: 'الحالة',
    statusConnected: 'متصل',
    statusNotConnected: 'غير متصل',
    title: 'عوامل التصفية'
  },
  page: {
    subtitle: 'التطبيقات والأدوات التي يمكن لـ Hermes استخدامها نيابةً عنك.',
    title: 'الاتصالات'
  },
  removeAccount: {
    body: name => `ستُزال ${name} من هذا الاتصال. سيبقى التطبيق متصلًا بحساباتك الأخرى.`,
    cancel: 'إلغاء',
    confirm: 'إزالة',
    title: 'إزالة هذا الحساب؟'
  },
  search: { clear: 'مسح البحث', placeholder: 'ابحث في الاتصالات' },
  sections: {
    connected: 'متصل',
    other: 'تطبيقات أخرى',
    seeAll: count => `عرض الكل (${count})`,
    suggested: 'مقترح لك'
  },
  sheet: {
    account: 'الحساب',
    accounts: 'الحسابات',
    addAnAccount: '+ إضافة حساب',
    addAnotherAccount: '+ إضافة حساب آخر',
    availableSubtitle: 'إيقاف مؤقت لكل الوكلاء؛ ابقَ مسجّل الدخول.',
    availableToAgents: 'متاح للوكلاء',
    cancelConnection: 'إلغاء الاتصال',
    changeAfterConnect: 'يمكنك تغييرها بعد الاتصال.',
    close: 'إغلاق',
    connected: 'متصل',
    connectedOn: date => `متصل في ${date}`,
    disconnectFailed: app => `تعذّر قطع اتصال ${app}. تحقق من اتصالك و`,
    facetDescriptions: {
      destructive: 'حذف دائم',
      read: 'عرض فقط',
      write: 'إنشاء وتحرير'
    },
    finishSignIn: 'إكمال تسجيل الدخول',
    finishSignInBody: app => `وافق على الوصول في صفحة ${app} في متصفحك، ثم عد إلى هنا.`,
    notConnected: 'غير متصل',
    permissionsTooltip: app =>
      `${app} يحدد ما إذا كانت كل أداة للقراءة أو الكتابة أو تدميرية. أنت تختار أيها مفعّل.`,
    previewAll: count => `معاينة الكل (${count})`,
    removeAccountBody: app =>
      `لن يتمكن الوكلاء من استخدام حساب ${app} هذا بعد الآن. تبقى الحسابات الأخرى متصلة.`,
    removeAccountTitle: alias => `إزالة «${alias}»؟`,
    removeFailed: alias => `تعذّرت إزالة «${alias}». تحقق من اتصالك و`,
    removedSuccess: alias => `تمت إزالة «${alias}» بنجاح`,
    renamePlaceholder: 'اسم الحساب',
    reopenSignIn: 'إعادة فتح صفحة تسجيل الدخول',
    reviewAll: count => `مراجعة الكل (${count})`,
    reviewTools: 'مراجعة كل الأدوات',
    signInBody: app => `يفتح ${app} في متصفحك. لا يرى Hermes كلمة مرورك أبدًا.`,
    signInWith: app => `تسجيل الدخول باستخدام ${app}`,
    startsOff: 'يبدأ متوقفًا',
    startsOn: 'يبدأ مفعّلًا',
    toolPermissions: 'أذونات الأدوات',
    tools: 'الأدوات',
    toolsCount: count => `${count} أداة`,
    waiting: 'بانتظار تسجيل الدخول',
    waitingFor: app => `بانتظار ${app}…`
  },
  subgroups: {
    'ai-media-presentations': 'الذكاء الاصطناعي والوسائط والعروض',
    'automation-ai-apis': 'الأتمتة والذكاء الاصطناعي وواجهات API',
    'bi-search-warehouses': 'ذكاء الأعمال والبحث والمستودعات',
    'code-review-ci': 'مراجعة الشيفرة والتكامل المستمر',
    communication: 'التواصل',
    'customer-support': 'دعم العملاء',
    'design-whiteboards': 'التصميم والسبورات',
    'docs-files': 'المستندات والملفات',
    'docs-knowledge': 'المستندات والمعرفة',
    'hiring-people': 'التوظيف والأفراد',
    'hosting-databases': 'الاستضافة وقواعد البيانات',
    'marketing-seo': 'التسويق وتحسين محركات البحث',
    'meetings-scheduling': 'الاجتماعات والجدولة',
    'monitoring-incidents': 'المراقبة والحوادث',
    'outdoors-fitness': 'الهواء الطلق واللياقة',
    'payments-commerce': 'المدفوعات والتجارة',
    'product-analytics': 'تحليلات المنتج',
    'projects-tasks': 'المشاريع والمهام',
    'sales-crm': 'المبيعات وإدارة العملاء',
    'spend-investing-crypto': 'الإنفاق والاستثمار والعملات الرقمية',
    travel: 'السفر',
    'websites-assets': 'المواقع والأصول'
  },
  toast: {
    connected: name => `تم اتصال ${name}.`,
    disconnected: name => `تم قطع اتصال ${name}.`,
    removed: name => `تمت إزالة ${name}.`,
    renamed: name => `أُعيدت التسمية إلى ${name}.`
  },
  toolFilters: {
    all: 'الكل',
    apply: 'عرض النتائج',
    areas: 'المجالات',
    behavior: 'السلوك',
    clear: 'مسح الكل',
    close: 'إغلاق',
    status: 'الحالة',
    title: 'تصفية الأدوات',
    uncategorised: 'غير مصنّفة'
  },
  tools: {
    allAreas: 'كل المجالات',
    areaChipLabel: 'المجال',
    back: 'العودة إلى الاتصالات',
    emptyBody: 'لم يشارك هذا التطبيق أي أدوات.',
    emptyTitle: 'لا أدوات بعد',
    errorBody: 'تعذّرت قراءة قائمة الأدوات. أعد المحاولة.',
    errorTitle: 'تعذّر تحميل الأدوات',
    filter: 'عوامل التصفية',
    findArea: 'ابحث عن مجال',
    loading: 'جارٍ تحميل الأدوات',
    lockedByOrg: 'أوقفتها مؤسستك',
    noResultsBody: 'لا شيء يطابق هذا البحث أو عوامل التصفية.',
    noResultsTitle: 'لا أدوات مطابقة',
    off: 'متوقفة',
    on: 'مفعّلة',
    pageSummary: (on, total, app) =>
      `${on} من ${total} أداة مفعّلة. تُحفظ التغييرات معًا وتُطبّق على كل وكيل يستخدم ${app}.`,
    pageTitle: app => `أدوات ${app}`,
    permissionChipLabel: 'الإذن',
    searchAppTools: app => `ابحث في أدوات ${app}`,
    searchPlaceholder: 'ابحث في الأدوات',
    showTools: count => `عرض ${count} أداة`,
    subtitle: name => `اختر ما يمكن لـ ${name} فعله.`,
    summary: (on, total) => `${on} من ${total} مفعّلة`,
    title: 'كل الأدوات',
    toolNamePrefix: 'اسم الأداة:',
    unsavedChanges: count => `${count} تغيير غير محفوظ`,
    whatItDoes: 'ما تفعله',
    whatItDoesHint: 'اختر واحدة أو أكثر. «الكل» يمسح الخيارات الأخرى.'
  }
}

