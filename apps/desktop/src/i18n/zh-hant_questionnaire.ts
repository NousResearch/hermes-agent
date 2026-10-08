import type { Translations } from './types'

export const zhHantQuestionnaire: Translations['questionnaire'] = {
  skipSetup: '略過設定',
  back: '返回',
  trailLabel: '你的回答',
  changeAnswer: '修改此回答',
  retry: '重試',
  saveFailed: '無法將設定標記為完成',
  skipped: '已略過',
  otherLabel: '其他回答',
  kinds: {
    accent: '強調色',
    layout: '版面',
    local: '本機模型',
    apps: '這部電腦上的 App',
    connectors: '連接器'
  },
  name: {
    greeting: '你好，我是 Hermes。',
    question: '我該怎麼稱呼你？',
    placeholder: '你的名字'
  },
  accent: {
    title: '選一個顏色',
    custom: '自訂顏色'
  },
  layout: {
    title: '你想怎麼使用？',
    basic: '基本',
    basicDetail: '用來和 Hermes 對話。',
    elite: '進階',
    eliteDetail: '給開發者：終端機、檔案、差異。'
  },
  local: {
    titleSpark: '你的 Spark 可以離線執行 Hermes',
    title: kind => `這台${kind}可以在本機執行模型`,
    download: model => `下載 ${model}`,
    downloadDetail: '第一次對話開始後在背景下載。',
    notNow: '暫不',
    notNowDetail: '會保留在模型選單中，顯示為「在本機執行」。',
    chip: '晶片',
    gpu: 'GPU',
    memory: '記憶體',
    os: '作業系統',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: '在這些 App 中使用 Hermes？',
    plugin: '外掛',
    pluginNotRunning: '外掛 · App 未執行',
    approve: '每次安裝都會在第一次對話中由你確認。',
    found: '在這部電腦上找到。未安裝的 App 不會列出。'
  },
  connectors: {
    title: '連接你的 App？',
    checking: '正在檢查你的帳號可以連接哪些 App…',
    offered: '這是你的帳號目前提供的清單。你會在第一次對話中逐一登入。'
  },
  task: {
    title: '先做什麼？',
    question: '選一個，或自己輸入。',
    blender: '在 Blender 中建立場景',
    nvidia: '調校我的 GPU',
    broadcast: '設定麥克風和攝影機',
    brief: '簡報今天的安排',
    tidy: '整理下載資料夾'
  },
  tour: {
    title: '要看看導覽嗎？',
    question: '之後快速看一圈？',
    quick: '帶我看看',
    quickDetail: '幾個站點，不到一分鐘。',
    none: '我自己摸索',
    noneDetail: '之後隨時可以請 Hermes 帶你導覽。'
  },
  start: {
    title: name => (name ? `準備好了，${name}。` : '準備好了。'),
    sub: '這是 Hermes 收到的第一則訊息。請求顯示在最前面，其餘內容在下方。',
    aboutMe: '關於我：',
    beforeYouStart: '開始之前：',
    noTask: '未選擇任務：Hermes 會問你想做什麼。',
    action: '開始',
    waiting: '正在準備你的免費帳號…',
    failed: '無法開始第一次對話'
  },
  summary: {
    custom: '自訂',
    local: model => `本機：${model}`,
    localNo: '本機：暫不',
    noApps: '無 App',
    noConnectors: '無連接器',
    more: (label, count) => `${label} +${count}`,
    tour: '導覽',
    noTour: '不導覽'
  },
  status: {
    settingUp: '正在設定你的免費帳號',
    stillSettingUp: '仍在設定你的免費帳號',
    unavailable: '免費帳號無法使用',
    ready: 'Nous · 免費方案',
    downloadingEngine: '正在下載本機引擎',
    downloading: model => `正在下載 ${model}`
  },
  settings: {
    title: '重新執行設定',
    description: '再次開啟首次啟動時的問題。你的回答會套用到預設設定檔。',
    action: '執行設定',
    failed: '無法開始設定'
  }
}
