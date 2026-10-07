import { defineLocale } from './define-locale'
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
import { zhHantSettings } from './zh-hant_settings'

export const zhHant = defineLocale({
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
    consentTitle: '協助改進 Hermes？',
    consentBody:
      '共享指標只包含有上限的計數，絕不包含提示詞、檔案、路徑或錯誤文字。收集僅在本機進行；傳送給 Nous 需要另行同意。',
    whatIsCollected: '收集哪些內容',
    collectedIntro: '僅限有上限的計數：',
    collectedActivity:
      '活動、工作階段長度、結果和錯誤類別，包括記憶寫入或上下文壓縮被拒絕、失敗或略過時的原因（來自固定清單）',
    collectedModels: '模型路由和 token 總量',
    collectedNames: '內建工具、指令和目錄項名稱',
    collectedMilestones: '分組的設定計數',
    collectedReliability:
      '更新和安裝的結果與耗時（失敗時包括來自固定清單的原因和所在階段；全新安裝記錄在本機，僅在你同意後計入）、當機、啟動與回覆速度、訊息平台狀態',
    collectedUsage:
      'Hermes 的使用方式：代理的準確度與效率（編輯是否成功、迴圈、錯誤後的恢復、每個任務的 token 與工具呼叫數、快取中斷），各介面與 Desktop 模式的活躍時間，哪些應用程式區域、操作與設定被使用、很快關閉或被關閉，以及供應商設定的結果',
    collectedMachine:
      '概略的機器資訊：記憶體範圍、GPU 類型、Hermes 版本新舊與發行通道、落後的更新數、是否使用本機模型伺服器',
    installId:
      '傳送會把每日資料包上傳到 Nous 遙測服務。資料包帶有此設定檔的安裝 ID：一個不含個人資訊的固定隨機 UUID，刪除共享指標目錄即可重設。',
    consentWindow:
      '只有整個收集期間都落在已記錄同意時段內的資料包才會被傳送。除全新安裝記錄（記錄在本機，僅在你同意後計入）外，你同意之前的資料，或傳送關閉期間的資料，都會留在本機。你可以隨時再次關閉傳送。',
    readDocs: '查看完整說明',
    share: '收集並傳送給 Nous',
    local: '僅在本機收集',
    off: '不用了',
    changeLater: '你可以隨時在 設定 → 安全性 中變更。',
    saveFailed: '無法儲存你的選擇',
    collectLabel: '收集使用統計',
    collectDesc: '在此裝置上保存有上限的計數。絕不包含提示詞、檔案、路徑或錯誤文字。',
    sendLabel: '向 Nous 傳送使用統計',
    sendDesc: '將每日資料包上傳到 Nous 遙測服務。只傳送同意時段內的資料。需要先開啟收集。',
    unavailable: '請更新 Hermes 後端以變更此設定。',
    stripBody: '僅限有界計數器，絕不包含提示詞或檔案。',
    stripReaskBody: '再次詢問：舊版本可能在你看到此問題之前就已儲存了「不用了」。',
    stripChoices: { share: '傳送給 Nous', local: '僅限本機', off: '不用了' },
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
  guidedGreeting: zhHantBoot.guidedGreeting,
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
  ui: zhHantCommon.ui
})
