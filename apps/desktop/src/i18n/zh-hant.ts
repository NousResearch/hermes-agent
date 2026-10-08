import { defineLocale, type TranslationOverrides } from './define-locale'
import { introZhHant } from './intro-zh-hant'
import { zhHantArtifacts } from './zh-hant_artifacts'
import { zhHantAssistant } from './zh-hant_assistant'
import { zhHantBoot } from './zh-hant_boot'
import { zhHantCapabilities } from './zh-hant_capabilities'
import { zhHantChat } from './zh-hant_chat'
import { zhHantChrome } from './zh-hant_chrome'
import { zhHantCommandCenter } from './zh-hant_command_center'
import { zhHantCommon } from './zh-hant_common'
import { zhHantConnectors } from './zh-hant_connectors'
import { zhHantDiagnostics } from './zh-hant_diagnostics'
import { zhHantMemoryDiscovery } from './zh-hant_memory_discovery'
import { zhHantSettings } from './zh-hant_settings'

export const zhHantOverrides = {
  memoryDiscovery: zhHantMemoryDiscovery,
  skillDeepLink: {
    installTitle: (name: string) => `安裝「${name}」？`,
    installDescription: '此技能將於新的工作階段中可用。請僅安裝可信來源的內容。',
    installTo: '安裝至',
    thisComputer: '這部電腦',
    installing: '正在安裝…',
    installComplete: (name: string) => `已安裝「${name}」`,
    destinationChanged: '安裝目標已變更。請關閉此對話框並重新開啟安裝連結。',
    installed: '已安裝',
    source: '來源'
  },
  externalOpenFailed: {
    title: '無法開啟此連結',
    message: '沒有註冊用於開啟此位址的瀏覽器。請複製連結並手動開啟。',
    copyUrl: '複製連結',
    close: '關閉'
  },
  sharedMetrics: {
    consentTitle: '分享使用統計？',
    dialogTitle: '使用統計',
    consentBody:
      'Hermes 可以統計你的使用方式：工作階段長度、執行了哪些模型和工具，以及何時發生失敗。它絕不記錄你的訊息、檔案、路徑或錯誤文字。',
    whatIsCollected: '統計哪些內容',
    collectedActivity: '工作階段：長度、結果、錯誤類型、每日活躍時間',
    collectedModels: '模型：使用哪些模型、token 總量',
    collectedNames: '功能：使用或關閉的內建工具、指令、應用程式區域和設定',
    collectedMilestones: '設定：完成了哪些步驟、提供方連線、技能、外掛和排程工作的數量',
    collectedReliability: '應用程式健康狀態：當機、啟動與回覆速度、更新、訊息平台連線',
    collectedUsage: '代理品質：未成功的編輯、損壞的工具呼叫、卡住的迴圈、每個任務的成本',
    collectedMachine: '機器：作業系統、記憶體範圍、GPU 類型、Hermes 版本、本機模型使用情況',
    sending:
      '除非你選擇分享，否則統計資料只會保留在這台電腦上。分享的統計資料每天傳送給 Nous 一次，並附帶此設定檔的隨機 ID。除了一則 Hermes 已安裝的一次性記錄（僅在你同意後計入）之外，你同意之前的統計資料永遠不會被傳送。你可以隨時在設定中變更。',
    readDocs: '查看完整說明',
    share: '與 Nous 分享',
    local: '保留在這部電腦上',
    off: '不用了',
    saveFailed: '無法儲存你的選擇',
    collectLabel: '收集使用統計',
    collectDesc: '僅限計數，保留在這部電腦上。絕不包含你的訊息、檔案、路徑或錯誤文字。',
    sendLabel: '與 Nous 分享使用統計',
    sendDesc:
      '每天一次將統計資料連同此設定檔的隨機 ID 傳送給 Nous。除一次性的安裝記錄外，你同意之前的統計資料絕不會被傳送。需要先開啟收集。',
    unavailable: '請更新 Hermes 後端以變更此設定。',
    stripBody: '僅限計數，絕不包含你的訊息或檔案。',
    stripReaskBody: '再次詢問：舊版本可能在你看到此問題之前就已儲存了「不用了」。',
    stripChoices: { share: '與 Nous 分享', local: '保留在這部電腦上', off: '不用了' },
    stripDetails: '詳細資訊'
  },
  intro: introZhHant,
  sessionImport: zhHantConnectors.sessionImport,
  common: zhHantCommon.common,
  fileMenu: zhHantChrome.fileMenu,
  boot: zhHantBoot.boot,
  notifications: zhHantDiagnostics.notifications,
  remoteDisplayBanner: zhHantBoot.remoteDisplayBanner,
  butterbar: zhHantBoot.butterbar,
  billingBlock: zhHantCommon.billingBlock,
  sendDiagnostics: zhHantDiagnostics.sendDiagnostics,
  titlebar: zhHantChrome.titlebar,
  language: zhHantSettings.language,
  settings: zhHantSettings.settings,
  skills: zhHantCapabilities.skills,
  starmap: zhHantCapabilities.starmap,
  agents: zhHantCapabilities.agents,
  commandCenter: zhHantCommandCenter.commandCenter,
  messaging: zhHantCommandCenter.messaging,
  profiles: zhHantCommandCenter.profiles,
  modelAssignment: {
    saveFailed: 'Hermes 未儲存該模型變更。',
    confirmTitle: '模型選擇警告',
    confirmDetail: '僅在你接受此權衡時確認。',
    confirmAction: '確認',
    declined: '已取消模型變更 — 你拒絕了資料訓練層級警告。'
  },
  cron: zhHantCommandCenter.cron,
  artifacts: zhHantArtifacts.artifacts,
  artifactCard: zhHantArtifacts.artifactCard,
  artifactPreview: zhHantArtifacts.artifactPreview,
  sidebar: zhHantChrome.sidebar,
  composer: zhHantChat.composer,
  statusStack: zhHantChat.statusStack,
  updates: zhHantBoot.updates,
  install: zhHantBoot.install,
  onboarding: zhHantBoot.onboarding,
  modelPicker: zhHantSettings.modelPicker,
  modelVisibility: zhHantSettings.modelVisibility,
  shell: zhHantChrome.shell,
  rightSidebar: zhHantChrome.rightSidebar,
  preview: zhHantArtifacts.preview,
  interfaceMode: {
    title: '介面模式',
    hint: '只改變顯示的內容，不改變 Hermes 的能力。',
    sessionNote: '由簡潔模式設定。此處的變更僅在本次工作階段內生效；切換到進階模式即可保留為你的設定。',
    simple: {
      label: '簡潔',
      description: '用於與 Hermes 對話。只有側邊欄和聊天；沒有終端機、檔案或差異面板。'
    },
    advanced: {
      label: '進階',
      description: '面向開發者。終端機、檔案、差異、狀態列和版面配置，按你的設定顯示。'
    }
  },
  zones: zhHantChrome.zones,
  contextMenu: zhHantChrome.contextMenu,
  assistant: zhHantAssistant.assistant,
  prompts: zhHantChat.prompts,
  desktop: zhHantChat.desktop,
  errors: zhHantDiagnostics.errors,
  tips: zhHantChat.tips,
  ui: zhHantCommon.ui,
  handoffTour: {
    profileTitle: '你的第一個任務在預設設定檔中執行',

    profileText:
      '這條欄用來切換設定檔。現在亮著的是 default，任務工作階段就在這裡。另一個是設定用的設定檔，歡迎聊天在那裡。',

    sessionsTitle: '每個設定檔都有自己的工作階段',

    sessionsText:
      '這個清單屬於 default 設定檔。「新工作階段」會在目前選取的設定檔中開始。在欄上切換設定檔，清單也會跟著改變。',

    stayTitle: 'Hermes 一鍵可及',

    stayText: '需要幫忙時，切換到設定用的設定檔並開啟「歡迎使用 Hermes」。它會一直在那裡。',
    localTitle: '這台電腦可以在本機執行模型',
    localText: (model: string) =>
      `${model} 適合你的硬體。免費執行，對話不會離開你的電腦。隨時在這裡的模型選單中選擇它。`
  },
  freeTier: {
    offer: {
      heading: '繼續使用 Hermes',
      body: '你正在使用免費額度。繼續使用 Hermes 的話，你會開始遇到限制。登入免費的 Nous 帳戶，即可獲得更多額度。',
      signIn: '登入',
      notNow: '暫不'
    }
  }
} satisfies TranslationOverrides

export const zhHant = defineLocale(zhHantOverrides)
