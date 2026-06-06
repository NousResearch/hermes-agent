import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale } from './define-locale'

export const zhHant = defineLocale({
  common: {
    save: '儲存',
    saving: '儲存中…',
    cancel: '取消',
    close: '關閉',
    confirm: '確認',
    delete: '刪除',
    refresh: '重新整理',
    retry: '重試',
    on: '開啟',
    off: '關閉'
  },

  language: {
    label: '語言',
    description: '選擇桌面介面的語言。',
    saving: '正在儲存語言…',
    saveError: '語言更新失敗',
    switchTo: '切換語言',
    searchPlaceholder: '搜尋語言…',
    noResults: '找不到語言'
  },

  settings: {
    closeSettings: '關閉設定',
    exportConfig: '匯出設定',
    importConfig: '匯入設定',
    resetToDefaults: '恢復預設值',
    resetConfirm: '要將所有設定恢復為 Hermes 預設值嗎？',
    exportFailed: '匯出失敗',
    resetFailed: '重設失敗',
    nav: {
      providers: '提供方',
      providerAccounts: '帳號',
      providerApiKeys: 'API 金鑰',
      gateway: '閘道',
      apiKeys: '工具與金鑰',
      keysTools: '工具',
      keysSettings: '設定',
      mcp: 'MCP',
      archivedChats: '已封存聊天',
      about: '關於'
    },
    sections: {
      model: '模型',
      chat: '聊天',
      appearance: '外觀',
      workspace: '工作區',
      safety: '安全性',
      memory: '記憶與上下文',
      voice: '語音',
      advanced: '進階'
    },
    searchPlaceholder: {
      about: '關於 Hermes Desktop',
      config: '搜尋設定…',
      gateway: '閘道連線…',
      keys: '搜尋 API 金鑰…',
      mcp: '搜尋 MCP 伺服器…',
      sessions: '搜尋已封存工作階段…'
    },
    modeOptions: {
      light: { label: '明亮', description: '明亮的桌面介面' },
      dark: { label: '深色', description: '降低眩光的工作區' },
      system: { label: '跟隨系統', description: '跟隨作業系統外觀' }
    },
    appearance: {
      title: '外觀',
      intro: '這些是僅限桌面端的顯示偏好。模式控制亮度；主題控制強調色與聊天介面樣式。',
      colorMode: '色彩模式',
      colorModeDesc: '選擇固定模式，或讓 Hermes 跟隨系統設定。',
      toolViewTitle: '工具呼叫顯示',
      toolViewDesc: '產品模式會隱藏原始工具 payload；技術模式會顯示完整輸入/輸出。',
      product: '產品',
      productDesc: '易讀的工具活動與精簡摘要。',
      technical: '技術',
      technicalDesc: '包含原始工具參數、結果與底層細節。',
      themeTitle: '主題',
      themeDesc: '僅限桌面端的調色盤。所選模式會套用在其上。'
    },
    fieldLabels: defineFieldCopy({
      model: '預設模型',
      model_context_length: '上下文視窗',
      fallback_providers: '備用模型',
      toolsets: '已啟用工具集',
      timezone: '時區',
      display: {
        personality: '人格',
        show_reasoning: '推理區塊'
      },
      agent: {
        max_turns: '最大代理步數',
        image_input_mode: '圖片附件',
        api_max_retries: 'API 重試次數',
        service_tier: '服務層級',
        tool_use_enforcement: '工具使用強制'
      },
      terminal: {
        cwd: '工作目錄',
        backend: '執行後端',
        timeout: '指令逾時',
        persistent_shell: '持久化 Shell',
        env_passthrough: '環境變數傳遞'
      },
      file_read_max_chars: '檔案讀取上限',
      tool_output: {
        max_bytes: '終端機輸出上限',
        max_lines: '檔案頁面上限',
        max_line_length: '行長上限'
      },
      code_execution: {
        mode: '程式碼執行模式'
      },
      approvals: {
        mode: '批准模式',
        timeout: '批准逾時',
        mcp_reload_confirm: '確認 MCP 重新載入'
      },
      command_allowlist: '指令允許清單',
      security: {
        redact_secrets: '遮蔽密鑰',
        allow_private_urls: '允許私有 URL'
      },
      browser: {
        allow_private_urls: '瀏覽器私有 URL',
        auto_local_for_private_urls: '私有 URL 使用本機瀏覽器'
      },
      checkpoints: {
        enabled: '檔案檢查點',
        max_snapshots: '檢查點上限'
      },
      voice: {
        record_key: '語音快捷鍵',
        max_recording_seconds: '最長錄音時間',
        auto_tts: '朗讀回覆'
      },
      stt: {
        enabled: '語音轉文字',
        provider: '語音轉文字提供方',
        local: {
          model: '本機轉寫模型',
          language: '轉寫語言'
        },
        elevenlabs: {
          model_id: 'ElevenLabs STT 模型',
          language_code: 'ElevenLabs 語言',
          tag_audio_events: '標記音訊事件',
          diarize: '說話者分離'
        }
      },
      tts: {
        provider: '文字轉語音提供方',
        edge: {
          voice: 'Edge 語音'
        },
        openai: {
          model: 'OpenAI TTS 模型',
          voice: 'OpenAI 語音'
        },
        elevenlabs: {
          voice_id: 'ElevenLabs 語音',
          model_id: 'ElevenLabs 模型'
        }
      },
      memory: {
        memory_enabled: '持久記憶',
        user_profile_enabled: '使用者設定檔',
        memory_char_limit: '記憶預算',
        user_char_limit: '設定檔預算',
        provider: '記憶提供方'
      },
      context: {
        engine: '上下文引擎'
      },
      compression: {
        enabled: '自動壓縮',
        threshold: '壓縮閾值',
        target_ratio: '壓縮目標',
        protect_last_n: '保護最近訊息'
      },
      delegation: {
        model: '子代理模型',
        provider: '子代理提供方',
        max_iterations: '子代理輪次上限',
        max_concurrent_children: '平行子代理',
        child_timeout_seconds: '子代理逾時',
        reasoning_effort: '子代理推理強度'
      },
      updates: {
        non_interactive_local_changes: '應用程式內更新的本機變更'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: '除非你在輸入框選擇其他模型，否則新聊天會使用此模型。',
      model_context_length: '保留 0 會使用所選模型偵測到的上下文視窗。',
      fallback_providers: '預設模型失敗時要嘗試的備用 provider:model 項目。',
      display: {
        personality: '新工作階段的預設助手風格。',
        show_reasoning: '後端提供推理內容時顯示該區塊。'
      },
      timezone: 'Hermes 需要本機時間上下文時使用。留空則使用系統時區。',
      agent: {
        image_input_mode: '控制圖片附件如何傳送給模型。',
        max_turns: 'Hermes 停止一次執行前的工具呼叫輪次上限。'
      },
      terminal: {
        cwd: '工具與終端機操作的預設專案資料夾。',
        persistent_shell: '後端支援時，在指令之間保留 Shell 狀態。',
        env_passthrough: '傳入工具執行的環境變數。'
      },
      code_execution: {
        mode: '程式碼執行被限制在目前專案中的嚴格程度。'
      },
      file_read_max_chars: 'Hermes 單次檔案讀取可讀取的最大字元數。',
      approvals: {
        mode: 'Hermes 如何處理需要明確批准的指令。',
        timeout: '批准提示逾時前等待的時間。'
      },
      security: {
        redact_secrets: '盡可能從模型可見內容中隱藏偵測到的密鑰。'
      },
      checkpoints: {
        enabled: '在檔案編輯前建立可回復的快照。'
      },
      memory: {
        memory_enabled: '儲存有助於未來工作階段的持久記憶。',
        user_profile_enabled: '維護一份精簡的使用者偏好設定檔。'
      },
      context: {
        engine: '長對話接近上下文上限時的管理策略。'
      },
      compression: {
        enabled: '對話變大時摘要較早的上下文。'
      },
      voice: {
        auto_tts: '自動朗讀助手回覆。'
      },
      stt: {
        enabled: '啟用本機或提供方支援的語音轉寫。',
        elevenlabs: {
          language_code: '可選的 ISO-639-3 語言代碼。留空讓 ElevenLabs 自動偵測。'
        }
      },
      updates: {
        non_interactive_local_changes:
          'Hermes 從應用程式內更新自身時，保留本機原始碼變更（stash）或丟棄（discard）。終端機更新一律會詢問。'
      }
    }),
    about: {
      heading: 'Hermes Desktop',
      version: value => `版本 ${value}`,
      versionUnavailable: '版本不可用',
      updates: '更新',
      checkNow: '立即檢查',
      checking: '檢查中…',
      seeWhatsNew: '查看新增內容',
      releaseNotes: '發行說明',
      onLatest: '你已是最新版本。',
      installing: '正在安裝更新。',
      cantUpdate: '此版本無法從應用程式內自行更新。',
      cantReach: '無法連線到更新伺服器。',
      tapCheck: '點選「立即檢查」以尋找更新。',
      updateReady: count => `新更新已就緒（包含 ${count} 項變更）。`,
      lastChecked: age => `上次檢查：${age}`,
      justNowSuffix: ' · 剛剛',
      automaticUpdates: '自動更新',
      automaticUpdatesDesc: 'Hermes 會在背景自動檢查更新，並在有可用更新時通知你。',
      branchCommit: (branch, commit) => `分支 ${branch} · 提交 ${commit}`,
      never: '從未',
      justNow: '剛剛',
      minAgo: count => `${count} 分鐘前`,
      hoursAgo: count => `${count} 小時前`,
      daysAgo: count => `${count} 天前`
    }
  },

  skills: {
    all: '全部',
    noDescription: '無可用描述。'
  },

  profiles: {
    newProfile: '新增設定檔',
    noProfiles: '找不到設定檔。',
    skills: count => `${count} 個技能`,
    defaultBadge: '預設',
    rename: '重新命名',
    saveSoul: '儲存 SOUL',
    cloneFromDefault: '從預設設定檔複製設定',
    invalidName: hint => `設定檔名稱無效。${hint}`,
    nameRequired: '名稱為必填',
    created: '已建立',
    renamed: '已重新命名',
    deleted: '已刪除',
    soulSaved: 'SOUL.md 已儲存'
  },

  cron: {
    last: '上次',
    next: '下次',
    resume: '繼續',
    pause: '暫停',
    triggerNow: '立即觸發',
    namePlaceholder: '例如：每日摘要',
    promptPlaceholder: '代理每次執行時應做什麼？'
  }
})
