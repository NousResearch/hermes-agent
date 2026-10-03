import { defineFieldCopy } from '@/app/settings/field-copy'

import type { TranslationOverrides } from './define-locale'

export const zhHantSettings = {
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
    subpages: {
      appearanceTheme: '主題',
      appearanceTypography: '字體與縮放',
      appearanceWindowLayout: '視窗與佈局',
      appearanceChatDisplay: '聊天顯示',
      appearancePet: '寵物',
      appearanceGeneral: '一般',
      modelMain: '主要模型',
      modelAuxiliary: '輔助模型',
      modelMoa: '多代理協作',
      modelFallbacks: '備用模型',
      chatBehavior: '行為',
      chatAttachments: '附件',
      workspaceProjects: '專案與探索',
      workspaceShell: 'Shell 環境',
      workspaceFiles: '檔案與執行',
      safetyApprovals: '核准',
      safetyPrivacy: '隱私與網路',
      safetyCheckpoints: '檢查點',
      browserProfile: '瀏覽器設定檔',
      browserNetwork: '本機與私人網址',
      memoryPersistent: '持久記憶',
      memoryContext: '上下文與壓縮',
      voiceConversation: '語音對話',
      voiceTranscription: '語音轉文字',
      voiceSpeech: '文字轉語音',
      advancedRuntime: '代理限制',
      advancedTools: '工具存取',
      advancedTerminal: '終端後端',
      advancedOutput: '輸出限制',
      advancedDelegation: '子代理',
      advancedDesktop: '桌面與啟動',
      gatewayConnection: '目前視窗',
      gatewayDevices: '已儲存的連線',
      gatewayManagedUpdates: '遠端更新',
      gatewayManagedUpdatesUnavailable: '遠端更新需要支援受管理 SSH 更新的桌面版本。',
      gatewayManagedUpdatesEmpty: '請在已儲存的連線中新增 SSH 連線，即可在此管理更新。',
      keyboardShortcuts: '按鍵繫結',
      hudGesture: 'HUD 手勢',
      screenCapture: '螢幕擷取',
      notificationAlerts: '桌面通知',
      notificationSounds: '聲音',
      archivedSessions: '封存與保留',
      defaultDirectory: '預設專案資料夾',
      vaultCredentials: '已儲存的憑證',
      vaultSources: '密碼管理員',
      appUpdates: '版本與更新',
      uninstall: '解除安裝',
      billingOverview: '概覽',
      billingPlans: '方案'
    },
    plugins: {
      openFolder: '開啟桌面外掛資料夾',
      installModal: {
        installUncertain:
          'Hermes 已停止等待安裝結果，但外掛可能仍在安裝中。請關閉此視窗，重新掃描外掛清單後再嘗試安裝。',
        installFromGit: '從 Git 安裝',
        reviewRepository: '檢查儲存庫',
        repoPlaceholder: 'https://github.com/owner/repo',
        toolsConnected: n => `已連線 ${n} 個工具`,
        skillsReady: names => (names.length === 1 ? `技能 ${names[0]} 已就緒` : `${names.length} 個技能已就緒`),
        nextChat: '更多工具將在下一次聊天中可用',
        serverNotConnected: (server, reason) => `MCP 伺服器 ${server} 未連線${reason ? `：${reason}` : '。'}`
      }
    },
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
      providerCustomEndpoints: '自訂端點',
      providerLocalModels: '本地模型',
      gateway: '閘道',
      apiKeys: '工具與金鑰',
      keybinds: '鍵盤快捷鍵',
      keysTools: '工具',
      keysSettings: '設定',
      mcp: 'MCP',
      archivedChats: '已封存聊天',
      sessions: '工作階段',
      about: '關於',
      billing: '帳單',
      notifications: '通知',
      vault: '密碼與登入'
    },
    vault: {
      title: '密碼與登入',
      blurb:
        '說一句「登入 GitHub」，代理就會代你登入。第一次遇到登入頁時它會當場向你索取登入資訊，之後就自動完成。密碼在本機加密儲存並直接填入頁面——模型永遠看不到。',
      count: n => `已儲存 ${n} 項`,
      loadFailed: '無法載入保險庫項目',
      empty: '尚未儲存任何內容',
      emptyDesc: '這裡不必手動新增。讓代理登入某個網站時，它會當場向你詢問一次登入資訊。若想提前輸入，可按「新增」。',
      add: '新增',
      addTitle: '新增登入資訊、信用卡或地址',
      addDescription: '加密儲存在此裝置上。代理永遠不會看到密碼。',
      added: '已儲存。',
      adding: '儲存中…',
      addConfirm: '儲存',
      kindField: '類型',
      kinds: { login: '登入', payment: '支付卡', address: '地址' },
      labelField: '標籤',
      labelPlaceholder: '例如：GitHub 工作帳號',
      labelRequired: '標籤為必填。',
      originField: '網站來源',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: '請輸入有效的 URL，例如 https://example.com。',
      identifierTypeField: '識別碼類型',
      identifierTypes: { email: '電子郵件', phone: '電話', username: '使用者名稱' },
      identifierField: '識別碼',
      identifierShown: identifier => identifier,
      passwordField: '密碼',
      loginFieldsRequired: '識別碼與密碼為必填。',
      cardNumberField: '卡號',
      cardNameField: '持卡人姓名',
      expMonthField: '到期月份',
      expYearField: '到期年份',
      cvcField: 'CVC',
      postalField: '郵遞區號',
      addressLine1Field: '地址第 1 行',
      addressLine2Field: '地址第 2 行',
      cityField: '城市',
      stateField: '州 / 地區',
      countryField: '國家/地區',
      optional: '（選填）',
      createdOn: date => `新增於 ${date}`,
      deleteAction: '移除已儲存項目',
      otpField: '驗證器金鑰',
      otpPlaceholder: 'Base32 金鑰或 otpauth:// 連結',
      otpHint: '啟用兩步驟驗證時網站顯示的「設定金鑰」。儲存後 Hermes 會自動產生驗證碼。',
      twoFactorBadge: '自動 2FA',
      deleteTitle: '刪除此項目？',
      deleteDescription: label => `「${label}」將從加密保險庫中移除。此操作無法復原。`,
      deleteConfirm: '刪除',
      sources: {
        title: '密碼管理器',
        blurb:
          '已安裝的密碼管理器會被自動偵測。代理第一次需要其中的登入資訊時會請你解鎖（每個工作階段一次）；記憶體中只保留工作階段權杖，代理永遠看不到你的主密碼或任何登入資訊。',
        toggleFailed: '無法更新密碼管理器',
        notInstalled: name => `未偵測到。安裝 ${name} 命令列工具並登入後，Hermes 會自動偵測。`,
        disabledDesc: '已偵測到，但已為 Hermes 關閉。',
        lockedDesc: '已偵測到。代理需要登入資訊時會請你解鎖，也可立即解鎖。',
        unlockedDesc: '本工作階段已解鎖。閒置 30 分鐘或關閉 Hermes 後會自動鎖定。',
        statusLocked: '已鎖定',
        statusNotDetected: '未偵測到',
        statusOff: '已關閉',
        statusUnlocked: '已解鎖',
        unlock: '解鎖',
        unlocking: '解鎖中…',
        lock: '鎖定',
        unlocked: name => `${name} 已在本工作階段解鎖。`,
        unlockTitle: name => `解鎖 ${name}`,
        unlockDescription: '輸入主密碼。它會交給本機的密碼管理器後立即捨棄，不會被儲存、記錄或顯示給代理。',
        masterPasswordPlaceholder: '主密碼'
      }
    },
    notifications: {
      title: '通知',
      intro: '原生桌面通知，與應用程式內提示不同。設定會依裝置保存，每台電腦各自獨立。',
      enableAll: '啟用通知',
      enableAllDesc: '關閉後靜音下方所有通知。',
      focusedHint: '完成提醒僅在 Hermes 位於背景時觸發。',
      kinds: {
        approval: {
          label: '需要核准',
          description: '有指令正在等待你核准或拒絕。'
        },
        input: {
          label: '需要輸入',
          description: 'Hermes 提出了問題，或需要密碼或密鑰。'
        },
        turnDone: {
          label: '回覆就緒',
          description: 'Hermes 在背景時完成了一輪對話。'
        },
        turnError: {
          label: '本輪失敗',
          description: '背景回合錯誤。'
        },
        backgroundDone: {
          label: '背景工作完成',
          description: '背景終端機指令已完成。'
        },
        credits: {
          label: '額度提醒',
          description: '額度存取被暫停或恢復。'
        },
        plugin: {
          label: '外掛通知',
          description: 'Hermes 在背景時，桌面外掛傳送了通知。'
        }
      },
      test: '傳送測試通知',
      testTitle: 'Hermes',
      testBody: '通知運作正常。',
      testSent: '測試已傳送。若沒有出現，請檢查系統通知權限與專注模式／勿擾模式。',
      testUnsupported: '此系統不支援原生通知。',
      completionSoundTitle: '完成提示音',
      completionSoundDesc: '代理回合結束時播放。可在此選擇預設並預覽。',
      completionSoundPreview: '預覽'
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
      chatTextScaleTitle: '聊天文字大小',
      chatTextScaleDesc: '相對於介面縮放調整對話文字和訊息輸入框的字級。側邊欄與控制項大小保持不變。',
      title: '外觀',
      intro: '這些是僅限桌面端的顯示偏好。模式控制亮度；主題控制強調色與聊天介面樣式。',
      colorMode: '色彩模式',
      colorModeDesc: '選擇固定模式，或讓 Hermes 跟隨系統設定。',
      toolViewTitle: '工具呼叫顯示',
      toolViewDesc: '產品模式會隱藏原始工具 payload；技術模式會顯示完整輸入/輸出。',
      hideCodeDiffsTitle: '隱藏程式碼差異',
      hideCodeDiffsDesc: '將檔案編輯顯示為附有新增和刪除行數的行內工具列，不顯示程式碼。',
      hideThreadTimelineTitle: '隱藏對話時間軸列',
      hideThreadTimelineDesc: '隱藏每個對話右側邊緣的導覽列。',
      reasoningCollapsedTitle: '預設摺疊推理過程',
      reasoningCollapsedDesc: '保留串流推理內容，但在您開啟前維持摺疊。',
      uiScaleTitle: '介面縮放',
      uiScaleDesc: (percent: number) =>
        `縮放整個應用程式的文字與介面。也可使用 Cmd/Ctrl 加 +、- 或 0 調整。目前：${percent}%`,
      sessionDensityTitle: '工作階段列表密度',
      sessionDensityDesc: '選擇側邊欄工作階段標題下方顯示的資訊量。',
      sessionDensityCompact: '緊湊',
      sessionDensityComfortable: '舒適',
      sessionDensityDetailed: '詳細',
      tabStripTitle: '分頁列',
      tabStripDesc: '在分區上方顯示分頁。自動模式會在分區只有一個面板時隱藏分頁，除非還開著其他聊天或磚塊分區。',
      tabStripAuto: '自動',
      tabStripAlways: '一律',
      tabStripNever: '永不',
      appActionsTitle: '應用操作',
      appActionsDesc: '設定、版面與 HUD 放在標題列左側或右側。選右側可把左側留給分頁。',
      appActionsLeft: '左側',
      appActionsRight: '右側',
      terminalFontTitle: '終端機字型',
      terminalFontDesc:
        '選擇已安裝的字型用於桌面端終端機。Nerd Font 可正確顯示 Powerlevel10k 與 Shell 圖示；留空則使用內建的 JetBrains Mono。',
      terminalFontPlaceholder: 'MesloLGS NF 或 CSS 字型堆疊',
      terminalFontPreview: '字形預覽',
      terminalFontReset: '使用預設字型',
      chatFontTitle: '聊天字型',
      chatFontDesc: '為聊天與應用程式介面選擇已安裝的字型，適合 OpenDyslexic 等易讀字型；留空則使用主題字型。',
      chatFontPlaceholder: 'OpenDyslexic 或 CSS 字型堆疊',
      chatFontPreview: '預覽',
      chatFontSample: '敏捷的棕色狐狸跳過懶狗。0123456789',
      chatFontReset: '使用主題字型',
      translucencyTitle: '視窗透明',
      translucencyDesc: '讓整個視窗（包括文字）透出桌面。',
      translucencyGlassDesc: '霧面玻璃：桌面以柔和模糊透出，文字保持清晰。',
      translucencyModeClear: '透明',
      translucencyModeGlass: '玻璃',
      translucencyTintTitle: '色調',
      translucencyFadeTitle: '淡出',
      translucencyFrostTitle: '磨砂質感',
      translucencyFrost: {
        'under-window': '深邃',
        popover: '柔和',
        titlebar: '明亮',
        header: '透亮'
      },
      translucencyScopeTitle: '套用範圍',
      translucencyScope: {
        window: '整個視窗',
        sidebar: '僅側邊欄'
      },
      backdropTitle: '聊天背景',
      backdropDesc: '對話後方那張淡淡的雕像圖片。',
      userBubbleTitle: '訊息氣泡',
      userBubbleDesc: '你自己的訊息有多透明。0 為不透明，100 時只保留邊框。',
      textDirectionTitle: '文字方向',
      textDirectionDesc:
        '設定聊天訊息和輸入框的文字方向。「自動」依每段的第一個字母判斷；混合文字排列不對時，可手動選擇方向。程式碼一律從左到右顯示。',
      textDirection: { auto: '自動', rtl: '從右到左', ltr: '從左到右' },
      introSplashTitle: '開場標識',
      introSplashDesc: '空白對話中顯示的字標和提示語。',
      modelPricingTitle: '模型價格',
      modelPricingDesc: '在模型選擇器中顯示每百萬 token 的輸入、輸出和快取讀取價格。',
      reactionsTitle: '訊息回應',
      reactionsDesc: 'iMessage 風格的表情回應 — 你可以對訊息做出回應，Hermes 也能回應你的訊息。',
      tipsTitle: '應用程式內提示',
      tipsDesc: '偶爾顯示來自應用程式和 Hermes 的提示，每則提示只出現一次。開始使用滿30天後自動關閉，你可以重新開啟。',
      tipsReset: (count: number) => `再次顯示 ${count} 則提示`,
      toursTitle: '導覽',
      toursDesc: '讓 Hermes 逐步標示每個位置，帶你認識應用程式。開始使用滿30天後自動關閉，你可以重新開啟。',
      composerPopoutTitle: '懸浮輸入框',
      composerPopoutDesc: '允許將輸入框拖出底部停靠區。關閉時，輸入框停靠在底部。',
      fileBrowserTitle: '檔案瀏覽器',
      fileBrowserDesc: '開啟工作區時，在聊天旁顯示檔案瀏覽器。標題列的切換按鈕也會變更此設定。',
      vibeHeartsTitle: '心情愛心',
      vibeHeartsDesc: '當你說謝謝、愛你、good bot 或送出愛心時飄出的愛心。與上方的訊息回應是兩回事。',
      embedsTitle: '內嵌預覽',
      embedsDesc:
        '豐富預覽會從第三方網站（YouTube、X 等）載入。詢問會在你允許前顯示佔位符；一律會自動載入；關閉則保留純連結。',
      embedsAsk: '詢問',
      embedsAlways: '一律',
      embedsOff: '關閉',
      embedsReset: (count: number) => `重設 ${count} 個已允許的服務`,
      resumeLastSessionTitle: '啟動時恢復上次會話',
      resumeLastSessionDesc: '開啟後，應用冷啟動時重新打開最近的聊天。關閉則始終從空白新會話開始。',
      product: '產品',
      productDesc: '易讀的工具活動與精簡摘要。',
      technical: '技術',
      technicalDesc: '包含原始工具參數、結果與底層細節。',
      themeTitle: '主題',
      themeDesc: '僅限桌面端的調色盤。所選模式會套用在其上。',
      themeSearchPlaceholder: '搜尋本機主題或 VS Code Marketplace…',
      themeProfileNote: profile => `已為「${profile}」設定檔儲存——每個設定檔保留各自的主題。`,
      installTitle: '從 VS Code 安裝',
      installDesc: '貼上 Marketplace 擴充功能 ID（例如 dracula-theme.theme-dracula），將其配色主題轉換為桌面調色盤。',
      installPlaceholder: 'publisher.extension',
      installButton: '安裝',
      installing: '安裝中…',
      installError: '無法安裝該主題。',
      installed: name => `已安裝「${name}」。`,
      removeTheme: '移除主題',
      importedBadge: '已匯入',
      pet: {
        title: '寵物',
        intro:
          '領養一隻懸浮在應用上的 petdex 動畫寵物，它會根據 Hermes 的狀態做出反應——工具執行時奔跑、成功時歡呼、出錯時沮喪。',
        restartHint: '寵物功能需要重新啟動——目前執行的應用在此功能加入前啟動。請結束並重新開啟 Hermes，然後回到此處。',
        scaleTitle: '大小',
        scaleDesc: '調整懸浮寵物的大小，所有介面即時生效。',
        roamTitle: '漫遊',
        roamDesc: '閒置時讓寵物自己在視窗內四處走動。',
        chooseTitle: '選擇寵物',
        chooseDesc: '選擇後會自動安裝（如需）並設為目前寵物。',
        searchPlaceholder: '搜尋寵物…',
        unreachable: '無法連線至 petdex 畫廊。請檢查網路連線並重新開啟此頁面。',
        noMatch: query => `沒有符合「${query}」的寵物。`,
        installedTag: '已安裝',
        generatedTag: '生成',
        countCapped: (cap, total) => `顯示 ${total} 個中的 ${cap} 個——輸入關鍵字以縮小範圍。`,
        count: n => `${n} 個寵物。`,
        uninstall: name => `解除安裝 ${name}`,
        delete: name => `刪除 ${name}`,
        deleteTitle: name => `刪除 ${name}？`,
        deleteBody: '此操作會永久刪除寵物，且無法重新安裝。',
        deleteConfirm: '刪除',
        rename: name => `重新命名 ${name}`,
        renameTitle: '重新命名寵物',
        renamePlaceholder: '為寵物取個名字',
        renameSave: '儲存',
        exportPet: name => `匯出 ${name}`,
        adoptFailed: slug => `無法領養 ${slug}`,
        uninstallFailed: slug => `無法解除安裝 ${slug}`,
        renameFailed: slug => `無法重新命名 ${slug}`,
        exportFailed: slug => `無法匯出 ${slug}`,
        noneAvailable: '目前沒有可開啟的寵物。',
        turnOnFailed: '無法開啟寵物。',
        turnOffFailed: '無法關閉寵物。'
      }
    },
    fieldLabels: defineFieldCopy({
      model: '預設模型',
      modelContextLength:
        '僅覆寫主聊天模型偵測到的上下文視窗（以 token 計）。保留 0 會使用所選模型偵測到的值。不影響輔助模型/MoA 模型。',
      fallbackProviders: '備用模型',
      toolsets: '已啟用工具集',
      timezone: '時區',
      display: {
        personality: '人格',
        showReasoning: '推理區塊'
      },
      desktop: {
        repoScanEnabled: '自動探索程式碼儲存庫',
        repoScanRoots: '程式碼儲存庫掃描根目錄',
        repoScanExcludePaths: '排除的程式碼儲存庫路徑'
      },
      agent: {
        maxTurns: '最大代理步數',
        imageInputMode: '圖片附件',
        apiMaxRetries: 'API 重試次數',
        serviceTier: '服務層級',
        toolUseEnforcement: '工具使用強制'
      },
      terminal: {
        cwd: '工作目錄',
        backend: '執行後端',
        timeout: '指令逾時',
        persistentShell: '持久化 Shell',
        envPassthrough: '環境變數傳遞',
        dockerImage: 'Docker 映像',
        singularityImage: 'Singularity 映像',
        modalImage: 'Modal 映像',
        daytonaImage: 'Daytona 映像'
      },
      fileReadMaxChars: '檔案讀取上限',
      toolOutput: {
        maxBytes: '終端機輸出上限',
        maxLines: '檔案頁面上限',
        maxLineLength: '行長上限'
      },
      codeExecution: {
        mode: '程式碼執行模式'
      },
      approvals: {
        mode: '批准模式',
        timeout: '批准逾時',
        mcpReloadConfirm: '確認 MCP 重新載入'
      },
      commandAllowlist: '指令允許清單',
      security: {
        redactSecrets: '遮蔽密鑰',
        allowPrivateUrls: '允許私有 URL'
      },
      browser: {
        allowPrivateUrls: '瀏覽器私有 URL',
        autoLocalForPrivateUrls: '私有 URL 使用本機瀏覽器',
        useRealProfile: '使用我的真實瀏覽器設定檔'
      },
      checkpoints: {
        enabled: '檔案檢查點',
        maxSnapshots: '檢查點上限'
      },
      voice: {
        maxRecordingSeconds: '最長錄音時間',
        autoTts: '朗讀回覆',
        voiceChatMode: '語音聊天模式',
        gptLive: {
          voice: 'GPT-Live 音色',
          instructions: 'GPT-Live 人設'
        }
      },
      stt: {
        enabled: '語音轉文字',
        provider: '語音轉文字提供方',
        echoTranscripts: '回傳轉寫文字',
        local: {
          model: '本機轉寫模型',
          language: '轉寫語言'
        },
        openai: {
          model: 'OpenAI STT 模型'
        },
        groq: {
          model: 'Groq STT 模型'
        },
        mistral: {
          model: 'Mistral STT 模型'
        },
        elevenlabs: {
          modelId: 'ElevenLabs STT 模型',
          languageCode: 'ElevenLabs 語言',
          tagAudioEvents: '標記音訊事件',
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
          voiceId: 'ElevenLabs 語音',
          modelId: 'ElevenLabs 模型'
        },
        deepinfra: {
          model: 'DeepInfra TTS 模型',
          voice: 'DeepInfra 語音'
        },
        xai: {
          voiceId: 'xAI (Grok) 語音',
          language: 'xAI 語言',
          speed: '播放速度',
          autoSpeechTags: '自動語音標籤',
          optimizeStreamingLatency: '串流延遲最佳化',
          sampleRate: '取樣率',
          bitRate: '位元率'
        },
        minimax: {
          model: 'MiniMax TTS 模型',
          voiceId: 'MiniMax 語音'
        },
        mistral: {
          model: 'Mistral TTS 模型',
          voiceId: 'Mistral 語音'
        },
        gemini: {
          model: 'Gemini TTS 模型',
          voice: 'Gemini 語音'
        },
        neutts: {
          model: 'NeuTTS 模型',
          device: 'NeuTTS 裝置'
        },
        kittentts: {
          model: 'KittenTTS 模型',
          voice: 'KittenTTS 語音'
        },
        piper: {
          voice: 'Piper 語音'
        }
      },
      memory: {
        memoryEnabled: '持久記憶',
        userProfileEnabled: '使用者設定檔',
        memoryCharLimit: '記憶預算',
        userCharLimit: '設定檔預算',
        provider: '記憶提供方'
      },
      context: {
        engine: '上下文引擎'
      },
      compression: {
        enabled: '自動壓縮',
        threshold: '壓縮閾值',
        codexGpt55Autoraise: 'Codex 壓縮自動提高',
        targetRatio: '壓縮目標',
        protectLastN: '保護最近訊息'
      },
      auxiliary: {
        compression: {
          timeout: '壓縮模型逾時（秒）'
        }
      },
      delegation: {
        model: '子代理模型',
        provider: '子代理提供方',
        maxIterations: '子代理輪次上限',
        maxConcurrentChildren: '平行子代理',
        childTimeoutSeconds: '子代理逾時',
        reasoningEffort: '子代理推理強度'
      },
      updates: {
        nonInteractiveLocalChanges: '應用程式內更新的本機變更'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: '除非你在輸入框選擇其他模型，否則新聊天會使用此模型。',
      modelContextLength: '保留 0 會使用所選模型偵測到的上下文視窗。',
      fallbackProviders: '預設模型失敗時要嘗試的備用 provider:model 項目。',
      display: {
        personality: '新工作階段的預設助手風格。',
        showReasoning: '後端提供推理內容時顯示該區塊。'
      },
      desktop: {
        repoScanEnabled: '掃描本機資料夾，並在「專案」中顯示 Git 程式碼儲存庫。',
        repoScanRoots: '要掃描的資料夾。留空時掃描主目錄。',
        repoScanExcludePaths: '探索程式碼儲存庫時略過這些資料夾及其子目錄。'
      },
      timezone: 'Hermes 需要本機時間上下文時使用。留空則使用系統時區。',
      agent: {
        imageInputMode: '控制圖片附件如何傳送給模型。',
        maxTurns: 'Hermes 停止一次執行前的工具呼叫輪次上限。'
      },
      terminal: {
        cwd: '工具與終端機操作的預設專案資料夾。',
        persistentShell: '後端支援時，在指令之間保留 Shell 狀態。',
        envPassthrough: '傳入工具執行的環境變數。',
        dockerImage: '執行後端為 Docker 時使用的容器映像。',
        singularityImage: '執行後端為 Singularity 時使用的映像。',
        modalImage: '執行後端為 Modal 時使用的映像。',
        daytonaImage: '執行後端為 Daytona 時使用的映像。'
      },
      codeExecution: {
        mode: '程式碼執行被限制在目前專案中的嚴格程度。'
      },
      fileReadMaxChars: 'Hermes 單次檔案讀取可讀取的最大字元數。',
      approvals: {
        mode: 'Hermes 如何處理需要明確批准的指令。',
        timeout: '訊息平台上的批准提示逾時前等待的時間。App 與終端機會一直等到你回覆。'
      },
      security: {
        redactSecrets: '盡可能從模型可見內容中隱藏偵測到的密鑰。'
      },
      checkpoints: {
        enabled: '在檔案編輯前建立可回復的快照。'
      },
      memory: {
        memoryEnabled: '儲存有助於未來工作階段的持久記憶。',
        userProfileEnabled: '維護一份精簡的使用者偏好設定檔。'
      },
      context: {
        engine: '長對話接近上下文上限時的管理策略。'
      },
      compression: {
        enabled: '對話變大時摘要較早的上下文。',
        codexGpt55Autoraise: '為支援的 ChatGPT Codex OAuth 模型將壓縮閾值提高到 85%。'
      },
      auxiliary: {
        compression: {
          timeout: '每次呼叫輔助壓縮模型的等待秒數（預設 120）。本機模型較慢時請調高。'
        }
      },
      browser: {
        useRealProfile:
          '本機瀏覽會使用你的真實登入狀態。Hermes 會將預設瀏覽器的設定（Cookie、登入資訊與偏好）複製成受管理的快照，再以內建的 Chromium 驅動它——不會直接開啟你正在使用的設定檔，且每次執行都會從目前的設定檔重新整理副本。設定雲端瀏覽器後端時，也允許代理視需要開啟本機真實設定檔工作階段。僅支援 Chromium 系瀏覽器（Chrome、Edge、Brave、Brave Origin、Chromium）；若預設瀏覽器並非 Chromium 系，會顯示明確錯誤。預設關閉。'
      },
      voice: {
        autoTts: '自動朗讀助手回覆。',
        voiceChatMode:
          'chained：語音轉文字 → Hermes → 文字轉語音，使用下方的提供方。gpt-live：由全雙工 OpenAI 語音模型（gpt-live-1）負責聆聽與說話，並將每個實際請求交給 Hermes——由你選擇的任意模型使用完整工具集作答。需要 OpenAI API 金鑰；語音層每分鐘收費 $0.05。',
        gptLive: {
          voice: 'GPT-Live 模式使用的音色，可填入自訂音色 ID。',
          instructions: '附加至即時語音人設的句子（語氣、語速、語言）。Hermes 會保留自己的系統提示詞。'
        }
      },
      stt: {
        enabled: '啟用本機或提供方支援的語音轉寫。',
        echoTranscripts: '將語音訊息的原始 🎙️ 轉寫文字傳回聊天。',
        elevenlabs: {
          languageCode: '可選的 ISO-639-3 語言代碼。留空讓 ElevenLabs 自動偵測。'
        }
      },
      tts: {
        xai: {
          voiceId: 'xAI 音色 ID（例如 eve）或自訂音色 ID。',
          language: '口語語言代碼（例如 en、pt-BR），或填入 "auto" 自動偵測。',
          speed: '播放速度。0.7 = 較慢，1.0 = 正常，1.5 = 較快。',
          autoSpeechTags: '合成前讓 LLM 在文稿中插入富有表現力的音訊標籤（例如 [laughing]、[sighs]）。',
          optimizeStreamingLatency: '延遲與品質的權衡。0 = 最佳品質，2 = 最低延遲。',
          sampleRate: '音訊取樣率（Hz）。越高音質越好、檔案越大。',
          bitRate: 'MP3 位元率（bps）。僅在編碼為 mp3 時生效。'
        },
        neutts: {
          device: 'NeuTTS 的本機推論裝置。'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Hermes 從應用程式內更新自身時，保留本機原始碼變更（stash）或丟棄（discard）。終端機更新一律會詢問。'
      }
    }),
    uninstallSection: {
      dangerZone: '危險操作',
      checkingInstalled: '正在檢查已安裝內容…',
      uninstallHermes: '解除安裝 Hermes',
      chooseHowMuch: '選擇要移除的內容。應用程式會關閉以完成作業；隨時重新開啟安裝程式即可返回。',
      confirmUninstall: '確認解除安裝',
      confirmBody: what => `這將移除${what}。此操作無法復原。`,
      appLabel: '應用程式：',
      couldNotStart: '無法開始解除安裝。',
      uninstalling: '正在解除安裝…',
      yesUninstall: '是，解除安裝',
      options: {
        gui: {
          title: '僅解除安裝聊天 GUI',
          description: '移除此桌面應用程式。Hermes 代理、你的設定和聊天記錄都會保留。',
          consequence: '桌面聊天 GUI（此應用程式及其資料）'
        },
        lite: {
          title: '解除安裝 GUI 與代理，保留資料',
          description: '移除應用程式和 Hermes 代理，但保留設定、聊天記錄和機密，以便日後重新安裝。',
          consequence: '聊天 GUI 和 Hermes 代理（設定、聊天記錄和機密會保留）'
        },
        full: {
          title: '解除安裝全部',
          description: '移除應用程式、代理和所有使用者資料——設定、聊天記錄、排程工作、機密和日誌。',
          consequence: '全部內容——聊天 GUI、Hermes 代理，以及你的所有設定、聊天記錄、機密和日誌'
        }
      }
    },
    poolLimits: {
      warmBotBackendsAria: '預熱機器人後端',
      warmBotBackendsTitle: '預熱機器人後端',
      backendIdleTimeoutAria: '後端閒置逾時（毫秒）',
      backendIdleTimeoutTitle: '後端閒置逾時（毫秒）'
    },
    customEndpoints: {
      active: '已啟用',
      apiKeySet: '已設定 API 金鑰',
      use: '使用',
      editTitle: '編輯端點',
      addTitle: '新增端點',
      fields: {
        name: '名稱',
        providerId: '供應商 ID',
        endpointUrl: '端點 URL',
        defaultModel: '預設模型',
        context: '上下文',
        apiKey: 'API 金鑰',
        apiKeyNewPlaceholder: '留空以保留目前金鑰',
        apiKeyPlaceholder: '選填',
        useNewChats: '用於新對話',
        discoverModels: '探索模型'
      },
      test: '測試',
      save: '儲存',
      newEndpoint: '新增端點',
      apiMode: 'API 模式',
      autoDetect: '自動偵測',
      couldNotLoad: '無法載入自訂端點',
      endpointSaved: '自訂端點已儲存。',
      saveFailed: '儲存失敗',
      endpointReachable: '端點可連線。',
      endpointReachableTransport: transport => `端點可連線（${transport} 路由已回應）。`,
      endpointReachableModels: (reachable, count) => `${reachable} 找到 ${count} 個模型。`,
      endpointValidationFailed: '端點驗證失敗。',
      validationFailed: '驗證失敗',
      activationFailed: '啟用失敗',
      deleteConfirm: name => `刪除 ${name}？`,
      deleteFailed: '刪除失敗',
      title: '自訂端點',
      deleteEndpoint: '刪除端點',
      emptyDescription: '在下方新增 OpenAI 相容端點。',
      emptyTitle: '尚無自訂端點',
      namePlaceholder: '我的代理',
      contextPlaceholder: '自動'
    },
    computerUse: {
      accessibility: '輔助使用',
      screenRecording: '螢幕錄製',
      driverHealth: '驅動程式健康狀態'
    },
    about: {
      updates: '更新'
    },
    config: {
      minimizeToTrayTitle: '最小化至系統匣',
      minimizeToTrayDesc:
        '最小化視窗或關閉主視窗時，將其隱藏至系統匣（macOS 上為選單列），讓 Hermes 繼續執行。透過系統匣選單中的「結束 Hermes」或 Cmd+Q 結束。預設關閉，僅適用於此裝置。',
      minimizeToTrayUnavailable: '系統匣無法使用。視窗將正常最小化和關閉。關閉此選項後重新開啟即可重試。',
      none: '無',
      noneParen: '(無)',
      builtinOnly: '僅內建',
      notSet: '未設定',
      commaSeparated: '逗號分隔的值',
      searchPlaceholder: '搜尋…',
      noResults: '找不到結果',
      systemDefault: '系統預設',
      loading: '正在載入 Hermes 設定...',
      emptyTitle: '無可設定項目',
      emptyDesc: '此區段沒有可調整的設定。',
      failedLoad: '設定載入失敗',
      autosaveFailed: '自動儲存失敗',
      imported: '設定已匯入',
      invalidJson: '設定 JSON 無效',
      keepAwakeTitle: '保持電腦喚醒',
      keepAwakeDesc: '阻止本機睡眠，讓長時間或整夜執行持續進行。螢幕仍可變暗。',
      voiceShortcutHintTitle: '語音錄製快捷鍵',
      voiceShortcutHintDesc:
        '請在「設定 → 鍵盤快捷鍵」中設定語音錄製快捷鍵（「Start / stop voice conversation」）。voice.record_key 設定僅適用於 CLI 和 TUI。',
      showOptions: '顯示選項'
    },
    hudModifier: {
      title: '輕按叫出 HUD',
      description:
        '在 Mac 上按下並放開 ⌘ + Option，在 Windows/Linux 上按下並放開 Ctrl + Alt，即可從任何應用程式將 HUD 帶到前景。預設關閉，僅適用於此裝置。',
      permission: '請在系統設定 → 隱私權與安全性 → 輸入監控中允許 Hermes，然後重試。此手勢不會記錄按鍵或擷取畫面。',
      unavailable: 'HUD 手勢輔助程式無法啟動或意外停止。請重試或重新啟動 Hermes。Hermes 內原有的 HUD 快速鍵仍可使用。',
      missingHelper: '此 Hermes 安裝缺少 HUD 手勢輔助程式。請更新或重新安裝 Hermes，然後重試。',
      unsupportedSession: '此桌面工作階段不支援全域修飾鍵輕按事件。Linux 需要 X11；不支援 Wayland。'
    },
    screenshot: {
      enabledTitle: '螢幕截圖快速鍵',
      enabledDesc:
        '在任何應用程式中同時按下左右兩個 Command 鍵，即可擷取最前方的視窗並附加到目前的 Hermes 草稿。絕不會自動傳送。預設關閉，僅適用於這台 Mac。視窗可能包含敏感內容，請在傳送前檢查附件。',
      statusTitle: '螢幕截圖快速鍵狀態',
      checking: '正在檢查螢幕截圖快速鍵…',
      disabled: '螢幕截圖快速鍵已關閉。',
      starting: '正在啟動快速鍵偵測，尚未就緒。',
      ready: '快速鍵已就緒。截圖會附加到目前的草稿，不會傳送。',
      inputPermission:
        '輸入監控權限可讓 Hermes 在其他應用程式使用中時偵測兩個 Command 鍵。請在系統設定 → 隱私權與安全性 → 輸入監控中允許 Hermes，然後返回此處重試。',
      screenPermission:
        '螢幕錄製權限可讓 Hermes 在你使用此快速鍵時擷取最前方的應用程式視窗。請在系統設定 → 隱私權與安全性 → 螢幕錄製中允許 Hermes，然後返回此處重試。如果 macOS 提示，請重新啟動 Hermes。',
      openSettings: '開啟系統設定',
      retry: '重試',
      unavailable: '螢幕截圖快速鍵無法使用。請重試或將其關閉。',
      errorTitle: '螢幕截圖快速鍵錯誤',
      loadFailed: '無法讀取快速鍵狀態。請重試以檢查目前的設定。',
      saveFailed: '無法確認快速鍵變更。請重試以檢查目前的設定。',
      permissionFailed: '無法開啟系統設定。請手動開啟「隱私權與安全性」，然後重試。',
      captureFailed: '無法擷取最前方的視窗。未附加或傳送任何內容。',
      contextChanged: '截圖期間目前的草稿已變更。截圖未附加或傳送。'
    },
    quickEntry: {
      enabledTitle: '快速輸入',
      enabledDesc: '用全域快速鍵在任何地方喚出一個小輸入框，無需開啟 Hermes 即可送出提示。',
      shortcutTitle: '快速輸入快速鍵',
      shortcutDesc: '至少需要一個修飾鍵，例如 CommandOrControl+Shift+Space。',
      active: '快速鍵已生效。',
      takenBy: '此快速鍵已被其他應用程式占用，請換一個。',
      invalidShortcut: '不是有效的快速鍵。請至少包含一個修飾鍵。'
    },
    credentials: {
      pasteKey: '貼上金鑰',
      pasteLabelKey: label => `貼上 ${label} 金鑰`,
      optional: '選填',
      enterValueFirst: '請先輸入一個值。',
      couldNotSave: '無法儲存憑證。',
      remove: '移除',
      getKey: '取得金鑰',
      saving: '儲存中'
    },
    envActions: {
      actions: '動作',
      manageInKeys: '在 API 金鑰中管理',
      docs: '文件',
      hideValue: '隱藏值',
      revealValue: '顯示值',
      replace: '取代',
      set: '設定',
      clear: '清除'
    },
    gateway: {
      loading: '正在載入閘道設定...',
      unavailableTitle: '閘道設定不可用',
      unavailableDesc: '桌面 IPC 橋接器未公開閘道設定。',
      title: '閘道連線',
      envOverride: '環境變數覆寫',
      intro:
        'Hermes Desktop 預設會啟動自己的本機閘道。如果您希望此應用程式控制另一台機器或可信代理後面已執行的 Hermes 後端，請使用遠端閘道。閘道連線屬於本機層級設定；設定檔是從已連線的閘道中探索出來的。',
      envOverrideTitle: '環境變數正在控制此桌面工作階段。',
      envOverrideDesc: '取消設定 HERMES_DESKTOP_REMOTE_URL 和 HERMES_DESKTOP_REMOTE_TOKEN 後才會使用下方儲存的設定。',
      localTitle: '本機閘道',
      localDesc: '在 localhost 啟動私有 Hermes 後端。這是預設方式，可離線使用。',
      remoteTitle: '遠端閘道',
      remoteDesc:
        '將此桌面殼層連線至遠端 Hermes 後端。託管閘道使用 OAuth 或帳號密碼；自託管閘道也可使用工作階段 Token。',
      remoteUrlTitle: '遠端 URL',
      remoteUrlDesc: '遠端儀表板後端的基礎 URL。支援路徑前綴，例如 /hermes。',
      probing: '正在檢查此閘道的驗證方式…',
      probeError: '暫時無法連線此閘道。請檢查 URL；閘道回應後將顯示驗證方式。',
      signedIn: '已登入',
      signIn: '登入',
      signOut: '登出',
      signInWith: provider => `使用 ${provider} 登入`,
      authTitle: '驗證',
      authSignedInPassword: '此閘道使用帳號和密碼。您已登入，工作階段會自動重新整理。',
      authSignedInOauth: '此閘道使用 OAuth。您已登入，工作階段會自動重新整理。',
      authNeedsPassword: '此閘道使用帳號和密碼。請登入以授權此桌面應用程式。',
      authNeedsOauth: provider => `此閘道使用 OAuth。請使用 ${provider} 登入以授權此桌面應用程式。`,
      tokenTitle: '工作階段 Token',
      tokenDesc: '用於 REST 和 WebSocket 存取的儀表板工作階段 Token。留空則保留已儲存的 Token。',
      existingToken: value => `現有 Token ${value}`,
      savedToken: '已儲存',
      pasteSessionToken: '貼上工作階段 Token',
      plainTextConfirmTitle: '以純文字儲存閘道 Token？',
      plainTextConfirmDesc:
        '在此裝置上找不到作業系統的金鑰環服務，因此 Token 將以未加密的純文字儲存在應用程式的連線設定檔中，以該使用者身分執行的任何處理程序皆可讀取。請安裝或啟用系統鑰匙圈（Linux 上為 GNOME Keyring 或 KWallet）以進行加密儲存。',
      plainTextConfirmAction: '以純文字儲存',
      plainTextStoredTitle: 'Token 以純文字儲存',
      plainTextStoredDesc:
        '安全儲存無法使用，因此已儲存的 Token 以未加密方式儲存在此裝置上應用程式的連線設定檔中。請安裝或啟用系統鑰匙圈（Linux 上為 GNOME Keyring 或 KWallet）以將其加密。',
      keychainEncryptionTitle: '使用系統鑰匙圈加密已儲存的機密',
      keychainEncryptionDesc:
        '預設關閉。開啟後，閘道 Token 與登入憑證將使用系統鑰匙圈（Keychain Access、GNOME Keyring 或 Windows DPAPI）加密——系統可能會要求授權或密碼。關閉時，它們以僅目前使用者可讀的一般檔案形式儲存。',
      keychainEncryptionFailed: '無法變更機密加密設定',
      testRemote: '測試遠端',
      saveForRestart: '儲存至下次重新啟動',
      saveAndReconnect: '儲存並重新連線',
      diagnostics: '診斷',
      diagnosticsDesc: '在檔案管理員中顯示 desktop.log，閘道啟動失敗時很有用。',
      openLogs: '開啟記錄',
      incompleteTitle: '遠端閘道設定不完整',
      incompleteSignIn: '切換至遠端前，請輸入遠端 URL 並完成登入。',
      incompleteToken: '切換至遠端前，請輸入遠端 URL 和工作階段 Token。',
      incompleteSignInTest: '測試前，請輸入遠端 URL 並完成登入。',
      incompleteTokenTest: '測試前，請輸入遠端 URL 和工作階段 Token。',
      enterUrlFirst: '請先輸入遠端 URL。',
      restartingTitle: '閘道連線正在重新啟動',
      savedTitle: '閘道設定已儲存',
      restartingMessage: 'Hermes Desktop 將使用已儲存的設定重新連線。',
      savedMessage: '已儲存，下次重新啟動後生效。',
      connectedTo: (baseUrl, version) => `已連線至 ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: '遠端閘道可連線',
      signedOutTitle: '已登出',
      signedOutMessage: '已清除遠端閘道工作階段。',
      failedLoad: '閘道設定載入失敗',
      signInFailed: '登入失敗',
      signOutFailed: '登出失敗',
      testFailed: '遠端閘道測試失敗',
      applyFailed: '無法套用閘道設定',
      saveFailed: '無法儲存閘道設定',
      sshTitle: '透過 SSH 連線',
      sshDesc:
        'Hermes 會透過 SSH 在遠端啟動並以通道連線到本應用程式——無需自行啟動或公開任何服務。前提：已具備到該主機的金鑰 SSH 存取。',
      sshTrustHint: '首次提供的主機金鑰會被信任並固定；後續變更將被拒絕。',
      sshHostTitle: '主機',
      sshHostDesc: 'user@host，或 ~/.ssh/config 中的 Host 別名。',
      sshHostPick: '選擇主機…',
      sshHostPickTitle: '主機',
      sshHostPickDesc: '~/.ssh/config 中的 Host 別名，或選擇「自訂」手動輸入。',
      sshHostCustom: '自訂（手動輸入）…',
      sshUserTitle: '使用者',
      sshUserDesc: '留空 = ~/.ssh/config 或目前使用者。',
      sshUserPlaceholder: '來自 ~/.ssh/config',
      sshPortTitle: '連接埠',
      sshPortDesc: '留空 = 22 或 ~/.ssh/config 中的連接埠。',
      sshKeyTitle: '金鑰檔案',
      sshKeyDesc: '私密金鑰路徑。留空 = ssh-agent 或 ~/.ssh/config。',
      sshHermesPathTitle: 'Hermes 路徑（選用）',
      sshHermesPathDesc: '遠端 hermes 執行檔的完整路徑。留空 = 自動偵測。',
      sshHermesPathPlaceholder: '自動偵測',
      sshTestConnection: '測試 SSH',
      sshConnect: '連線',
      sshButtonsHint: '「儲存」會在下次啟動時生效，「連線」則立即重新連線。',
      sshReachable: (host, platform) => `可連線：${host}（${platform}）——已找到 Hermes`,
      sshIncompleteHost: '連線前請輸入 SSH 主機。',
      sshErrUnreachable: '無法透過 SSH 連線到該主機。請檢查主機、連接埠和網路。',
      sshErrAuth:
        'SSH 驗證失敗。請將金鑰載入 ssh-agent（ssh-add），或在 ~/.ssh/config 中設定 IdentityFile——Hermes 以非互動方式執行 ssh。',
      sshErrHostKey: '自上次連線以來主機金鑰已變更。請確認這是預期的，然後執行 ssh-keygen -R <host> 並重新連線。',
      sshErrNotInstalled:
        '遠端主機上未安裝 Hermes。請在遠端安裝（curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh）或設定 Hermes 路徑。',
      sshErrPlatform: '不支援的遠端平台。Hermes Desktop 的 SSH 模式支援 Linux、macOS 和 Windows 遠端主機。',
      sshErrTimeout: 'SSH 連線逾時。主機可能無法存取或處於睡眠狀態。',
      sshErrUpdateRequired: '使用 Desktop SSH 連線前，請更新遠端主機上的 Hermes。',
      sshErrUnknown: 'SSH 連線失敗。'
    },
    keys: {
      loading: '正在載入 API 金鑰和憑證...',
      failedLoad: 'API 金鑰載入失敗',
      empty: '此類別尚未有任何設定。'
    },
    search: {
      placeholder: '搜尋所有設定...',
      pill: '搜尋'
    },
    profileScope: {
      appliesTo: '套用至',
      editsProfile: profile => `此頁面的變更將套用至「${profile}」設定檔。`
    },
    mcp: {
      loading: '正在載入 MCP 伺服器...',
      invalidJson: 'MCP JSON 無效',
      saveFailed: '儲存失敗',
      removeFailed: '移除失敗',
      reloadFailed: 'MCP 重新載入失敗',
      savedTitle: 'MCP 伺服器已儲存',
      savedMessage: name => `${name} 會在 MCP 重新載入後生效。`,
      disabled: '已停用',
      name: '名稱',
      serverJson: '伺服器 JSON',
      remove: '移除',
      capabilitySummary: (tools, prompts, resources) =>
        `已啟用 ${[`${tools} 個工具`, ...(prompts ? [`${prompts} 個提示`] : []), ...(resources ? [`${resources} 個資源`] : [])].join('、')}`,
      costTokens: tokens => `每次呼叫約 ${tokens} token`,
      usage30d: uses => `30 天內 ${uses} 次呼叫`,
      statusConnecting: '連線中…',
      statusNeedsAuth: '需要驗證',
      statusError: '錯誤',
      statusOff: '關閉',
      allServers: '所有伺服器',
      authenticatedTitle: '已驗證',
      authenticatedMessage: (server, count) => `${server}：${count} 個工具`,
      authenticate: '驗證',
      noOutput: '尚無輸出。',
      deepLinkTitle: '新增 MCP 伺服器？',
      deepLinkDescription: '一個連結要求將此 MCP 伺服器加入 Hermes。請檢查下方的完整設定——它來自該連結，而非 Hermes。',
      deepLinkStdioWarning: '此伺服器會使用下方所示指令在你的電腦上執行本機程序。僅在信任其來源時繼續。',
      deepLinkConfirm: '新增伺服器',
      deepLinkNameInvalid: '名稱須為 1-64 個字母、數字、點、連字號或底線。',
      deepLinkNameConflict: name => `已存在名為 ${name} 的伺服器——請改用其他名稱或取消。`,
      deepLinkErrorTitle: '已拒絕 MCP 安裝連結',
      deepLinkErrorName: '連結中的伺服器名稱缺失或無效。',
      deepLinkErrorConfig: '連結中的設定不是有效的 base64 編碼 JSON。',
      deepLinkErrorShape: '設定必須是包含字串 `url` 或 `command` 欄位的 JSON 物件。',
      deepLinkErrorUrl: '僅允許 http:// 和 https:// 伺服器網址。',
      deepLinkErrorTooLarge: '設定內容超過 32KB 上限。'
    },
    model: {
      setupProviderFallback: '提供方',
      setUpProvider: name => `設定 ${name}`,
      staleAuxBefore: (count, names) => `${count} 個輔助任務（${names}）仍由 `,
      staleAuxAfter: ' 執行，而非主要模型。',
      staleAuxOtherProviders: '其他提供方',
      moaEnabled: '啟用',
      moaSetDefault: '設為預設',
      moaNewPresetPlaceholder: '新預設',
      moaAddPreset: '新增預設',
      customModel: '自訂模型…',
      customModelPlaceholder: '模型 ID',
      chooseFromList: '從清單中選擇',
      moaDefault: '預設：',
      moaReferenceToggle: (enabled, index) => `${enabled ? '停用' : '啟用'}參考 ${index}`,
      moaReferenceTitle: index => `參考 ${index}`,
      moaAddReference: '新增參考模型',
      loading: '正在載入模型設定...',
      appliesDesc: '套用至新工作階段。可在輸入框的模型選擇器中臨時切換目前對話。',
      provider: '提供方',
      model: '模型',
      applying: '套用中...',
      loadFailed: '無法載入模型',
      restartRequired: '更新後此後端仍在執行舊程式碼。請重新啟動以載入新程式碼。',
      restartBackend: '重新啟動後端',
      restartingBackend: '正在重新啟動後端...',
      restartFailed: '無法重新啟動後端',
      auxiliaryTitle: '輔助模型',
      resetAllToMain: '全部重設為主要模型',
      staleAuxDismiss: '不再顯示',
      auxiliaryDesc: '輔助任務預設使用主要模型。您可以為任何任務指定專用模型。',
      setToMain: '設為主要模型',
      change: '變更',
      autoUseMain: '自動 · 使用主要模型',
      inheritMainEffort: '繼承 · 主要模型推理強度',
      providerDefault: '(提供方預設)',
      moaTitle: '混合代理（Mixture of Agents）',
      moaPreset: '預設',
      moaDescription:
        '設定以「混合代理」提供者下模型形式出現的命名預設。聚合模型是執行模型——它執行工具迴圈的每一步，整個運行幾乎全部費用都計入其提供者。參考模型預設每輪使用者訊息僅提供一次建議。',
      moaAggregator: '聚合模型',
      moaAggregatorBilled: '執行模型 · 整個運行在此計費',
      moaReferenceHint: '默認每輪僅建議一次',
      tasks: {
        vision: { label: '視覺', hint: '圖片分析' },
        compression: { label: '壓縮', hint: '上下文壓縮' },
        skills_hub: { label: '技能中心', hint: '技能搜尋' },
        approval: { label: '核准', hint: '智慧自動核准' },
        mcp: { label: 'MCP', hint: 'MCP 工具路由' },
        title_generation: { label: '標題生成', hint: '工作階段標題' },
        review: { label: '評審', hint: '/review 評審子代理' },
        triage_specifier: { label: '分類指定', hint: '看板任務規格補全' },
        kanban_decomposer: { label: '看板分解', hint: '任務拆解' },
        profile_describer: { label: '設定檔描述', hint: '自動生成設定檔描述' },
        curator: { label: '策展器', hint: '技能使用審查' }
      }
    },
    localModels: {
      connectionChanged: '本地模型連線已變更',
      title: '本地模型',
      runtimeTitle: '本地執行環境',
      runtimeReady: backend => `就緒 · ${backend}`,
      serverRunning: '執行中',
      runtimeInstalled: '已安裝 llama.cpp 執行環境',
      runtimeInstalledDetail: (tag, backend) => `組建 ${tag}，${backend} 後端。Hermes 會為您啟動並管理伺服器。`,
      installTitle: '安裝本地執行環境',
      installDetail: '下載 llama.cpp 推理引擎（數百 MB）。下載的模型完全在本機執行——無需帳號，資料不會離開您的電腦。',
      installAction: '安裝執行環境',
      installing: '正在安裝執行環境…',
      installFailed: '執行環境安裝失敗',
      hardwareTitle: '本機配置',
      hardwareLoading: '正在檢測硬體…',
      vram: label => `${label} 顯示記憶體`,
      ram: label => `${label} 記憶體`,
      unifiedMemory: '統一記憶體',
      modelsTitle: '模型',
      recommended: '推薦',
      recommendedReason: {
        'best-quality-resident': '在完全駐留 GPU 且保持全速的模型中品質最高。推薦會在品質與該硬體的預計速度之間權衡。',
        'speed-gated-quality':
          '有更高品質的模型可以裝入這台機器，但受記憶體頻寬限制回應會太慢——這是保持流暢的最佳模型。',
        'fastest-resident': '沒有模型能在該硬體上達到全速；這是完全駐留 GPU 記憶體中最快的一個。'
      } as Record<string, string>,
      noRecommendationTitle: '此裝置暫無自動推薦模型',
      noRecommendationDetail:
        '自動設定需要一個可完全放入 GPU 記憶體或統一記憶體的精選模型。你仍可在下方自行選擇，或瀏覽更多模型。',
      noRecommendationAction: '瀏覽模型',
      quickstartConfigure: '讓我選擇',
      downloaded: '已下載',
      downloadAction: size => `下載 · ${size}`,
      downloadProgress: (done, total) => `${done} / ${total}`,
      downloadStatusRunning: '下載中',
      downloadSpeed: rate => `${rate}`,
      downloadEta: time => `剩餘約 ${time}`,
      downloadPausedLabel: '已暫停',
      downloadPauseAction: '暫停',
      downloadResumeAction: '繼續',
      downloadDoneToast: model => `${model} 已就緒。`,
      installDoneToast: '本地執行環境已安裝就緒。',
      useAction: '使用',
      activePill: '預設',
      updateTitle: '引擎有可用更新',
      updateDetail: (next, current) =>
        `新的 llama.cpp 組建（${next}）可以安裝——目前為 ${current}。下載期間模型仍可正常使用。`,
      updateAction: '更新引擎',
      updating: '正在更新引擎…',
      upToDateTitle: '引擎已是最新',
      upToDateDetail: (tag, backend) => `正在執行 llama.cpp ${tag}（${backend})。`,
      activeDetail: '新對話使用此模型——傳送首條訊息時載入',
      activeNotLoaded: '首條訊息時載入',
      loadedPill: '已載入',
      placementResident: '全部在 GPU',
      placementSpilled: '部分在記憶體',
      placementResidentTip: '完全在 GPU 記憶體中以此上下文視窗執行——全速。',
      placementSpilledTip:
        '模型的一部分從系統記憶體執行——可用但較慢。更緊湊的版本或更小的上下文可以完全放入顯示記憶體。',
      loadingPill: '載入中…',
      ejectTip: '釋放顯示記憶體（需要時重新載入）',
      ejected: '模型已卸載——顯示記憶體已釋放。',
      ejectFailed: '無法卸載模型',
      stopServer: '關閉',
      startServer: '開啟',
      runtimeRunningDetail: '本地伺服器執行中。關閉後將釋放全部顯示記憶體，新對話將不再使用本地模型，直到您重新開啟。',
      serverStopped: '本地伺服器已停止——顯示記憶體已釋放。',
      serverStarted: '本地伺服器執行中。',
      serverStopFailed: '無法停止本地伺服器',
      serverStartFailed: '無法啟動本地伺服器',
      activating: '啟動中…',
      activateFailed: model => `無法切換到 ${model}`,
      activateDoneToast: model => `新對話將使用 ${model}。`,
      downloadFailed: model => `${model} 下載失敗`,
      pillFitsGpu: '完全在 GPU 上執行',
      pillUsesRam: '使用系統記憶體',
      pillTooBig: '超出本機記憶體',
      browseTitle: '發現更多模型',
      browseHint: '搜尋整個 Hugging Face。在這裡下載的模型會自動適配你的機器，但未經我們測試。',
      browsePlaceholder: '按名稱或作者搜尋模型…',
      browseSearching: '正在搜尋 Hugging Face',
      browseListing: '正在讀取模型檔案',
      browseShowFiles: '查看檔案',
      browseRefresh: '重新整理',
      browseDownloads: '次下載',
      browseLikes: '個讚',
      browseGated: '需要登入 Hugging Face',
      browseNoGguf: '未找到相容的模型檔案。',
      browseFitUnknown: '適配情況未知',
      browseAlreadyDownloaded: '已下載。',
      addedByYou: '由你新增',
      browseDownloadStarted: '正在下載 {name}',
      browseDownloadAria: '下載 {name}',
      sideloadButton: '新增模型檔案',
      sideloadTitle: '選擇 GGUF 模型檔案',
      sideloadDone: '已新增 {name}。',
      sideloadAlreadyPresent: '已在你的庫中。',
      pillFullContext: max => `完整 ${max} 上下文`,
      pillFullContextTip: '從一開始就以模型的完整上下文視窗執行',
      pillUpTo: max => `最高 ${max} 上下文`,
      pillGrowsTip: '隨著對話需要更多空間自動增長',
      pillVision: '識圖',
      deleteAction: '刪除模型',
      deleteConfirm: model => `從磁碟刪除 ${model}？`,
      deleted: model => `已刪除 ${model}。`,
      deleteFailed: '刪除失敗'
    },
    billing: {
      perMonth: amount => `${amount}/月`,
      creditsPerMonth: amount => `${amount} 額度/月`,
      usageLabel: label => `${label}用量`,
      freeTier: {
        signIn: '登入',
        title: '你正在使用 Nous 免費服務',
        message: '登入 Nous 帳戶以解鎖更多模型和工具。',
        caption: '使用 nous/welcome，包含連接器。登入後會保留連接器，並增加需要帳戶的工具和其他所有模型。',
        name: 'Nous · 免費服務',
        footnote: '免費服務沒有餘額，無需付款。登入 Nous 帳戶後才會顯示支付與用量。',
        plan: '免費服務',
        model: '模型',
        connectors: '連接器',
        included: '已包含'
      },
      amountValidation: {
        reloadTo: '儲值金額',
        greaterThanThreshold: '儲值金額必須大於門檻。',
        decimal: label => `${label}：請輸入最多含兩位小數的美元金額。`,
        positive: label => `${label}：金額必須大於 $0。`,
        minimum: (label, amount) => `${label}：最低金額為 ${amount}。`,
        maximum: (label, amount) => `${label}：最高金額為 ${amount}。`
      },
      stepUp: {
        openVerification: '開啟驗證頁面',
        dismiss: '關閉',
        waiting: '正在等待驗證連結…',
        verify: '驗證以繼續',
        deniedTitle: '驗證未獲核准',
        deniedBody: '驗證已結束，但未允許此終端進行遠端支出。',
        successTitle: '驗證完成',
        successBody: '此終端已獲准進行遠端支出。'
      },
      charge: {
        added: amount => (amount ? `已加入 $${amount}。` : '已加入額度。'),
        failedTitle: '扣款失敗',
        unconfirmedTitle: '扣款結果尚未確認',
        unconfirmedBody: message => `${message} 上次扣款結果尚未確認，請在重試前檢查餘額和歷史記錄。`,
        checkTitle: '無法檢查扣款',
        checkBody: '無法檢查扣款。',
        untrackedTitle: '無法追蹤扣款',
        untrackedBody: '帳單服務已接受請求，但未傳回扣款識別碼。',
        timeoutTitle: '5 分鐘後仍在處理',
        timeoutBody: '扣款仍可能結算，請在重試前檢查入口網站。',
        authenticationRequired: '銀行要求驗證（3DS）。請在入口網站完成驗證以完成本次購買。',
        expired: '銀行卡已到期。請在入口網站中更新。',
        declined: '銀行卡遭拒。請在入口網站中嘗試另一張卡。',
        failedBody: reason => `扣款未成功（${reason}）。`
      },
      title: '帳單',
      preview: '預覽',
      summary: { balance: '餘額', plan: '方案', autoRefill: '自動儲值' },
      sections: {
        invoices: '發票',
        plan: '方案',
        paymentAndCredits: '支付與額度',
        usage: '用量'
      },
      usage: { title: '用量' },
      buyCredits: {
        customAmount: '自訂儲值金額',
        title: '立即購買額度',
        buyButton: '購買',
        processing: '處理中…正在確認結算',
        added: amount => `已加入 ${amount}，正在重新整理餘額。`,
        retry: '重試',
        openPortal: '開啟入口網站'
      },
      plan: {
        title: '方案',
        changePlan: '變更方案',
        viewPlans: '查看方案',
        backAria: '返回帳單',
        current: '目前方案',
        scheduled: '已排定',
        empty: '目前沒有可切換的方案。',
        undo: '復原',
        undoing: '正在復原…',
        downgrade: '降級',
        confirmDowngrade: '確認降級',
        tryAgain: '重試',
        checkingChange: '正在檢查此變更…',
        cannotChange: '無法在此進行該變更。',
        alreadyOn: name => `你已使用 ${name}，無需變更。`,
        notScheduleable: '無法在此排定該變更。',
        scheduling: '正在排定…',
        cancel: '取消',
        effectScheduled: (targetName, effectiveAt, creditsDelta) =>
          `變更為 ${targetName}，於 ${effectiveAt} 生效。現在不扣款；在此之前保留目前方案。${creditsDelta ? `每月額度變化：${creditsDelta}。` : ''}`
      },
      autoReload: {
        threshold: '門檻',
        thresholdAria: '自動儲值門檻',
        reloadTo: '儲值金額',
        reloadToAria: '自動儲值金額',
        turnOffConfirm: '關閉自動儲值？',
        turnOff: '關閉',
        disable: '停用',
        updated: '自動儲值已更新。',
        turnedOff: '自動儲值已關閉。',
        manage: '管理',
        save: '儲存',
        saving: '正在儲存…',
        cancel: '取消'
      },
      state: {
        notice: {
          loggedOut: {
            title: '連接你的 Nous 帳戶',
            message: '登入你的 Nous 帳戶，即可在此查看餘額、方案和用量。',
            action: '登入'
          },
          openPortal: '開啟入口網站 ↗',
          noCard: {
            title: '尚未新增付款方式',
            message: '新增銀行卡後才能購買額度和使用自動儲值。請在入口網站中新增。',
            action: '新增銀行卡 ↗'
          }
        },
        paymentMethod: {
          title: '付款方式',
          description: '管理用於儲值和訂閱續費的銀行卡。',
          addAction: '新增付款方式',
          updateAction: '更新',
          provenance: {
            autoRefill: '自動儲值卡',
            customerDefault: '帳戶預設卡',
            subPin: '訂閱卡',
            suffix: label => ` - ${label}`
          }
        },
        buyCredits: { description: '從銀行卡一次性扣款，今天即可計入餘額。' },
        autoRefill: {
          title: '餘額不足時儲值',
          genericDescription: '餘額低於門檻時自動補充額度。',
          offPill: '已關閉',
          enabledPill: '已啟用',
          notAvailablePill: '—',
          manageCaption: '在入口網站中管理自動儲值。',
          turnOnCaption: '在入口網站中開啟自動儲值',
          chargesDescription: (reloadTo, threshold) => `餘額低於 ${threshold} 時自動扣款 ${reloadTo}。`,
          distinctCardCaption: cardLabel => `自動儲值使用 ${cardLabel} 扣款，請在入口網站中核對`,
          distinctCardFallback: '另一張銀行卡',
          reconcileAction: '核對 ↗'
        },
        usage: {
          subscriptionCredits: {
            title: '訂閱額度',
            barLabel: '剩餘訂閱額度',
            captionResets: date => `於 ${date} 重設`,
            valueOf: (remaining, monthly) => `${monthly} 中剩餘 ${remaining}`,
            valueOver: (remaining, monthly, over) => `${monthly} 中剩餘 ${remaining} · 超出 ${over}`
          },
          topupCredits: { title: '儲值額度', caption: '不會到期' },
          monthlyCap: {
            title: '每月支出上限',
            barLabel: '已用每月支出限額',
            captionDefault: '預設上限',
            captionSpending: '每月遠端支出',
            valueUsed: (spent, limit) => `${limit} 中已用 ${spent}`
          }
        },
        planCard: {
          freeTier: '免費',
          chooseAction: '選擇 ↗',
          adjustPlanAction: '調整方案 ↗',
          unavailableCaption: '訂閱詳細資料暫不可用，仍可開啟入口網站。',
          downgradeCaption: (tierName, when) => `於 ${when} 變更為 ${tierName}。`,
          cancellationCaption: when => `於 ${when} 取消。`,
          renewsCaption: date => `於 ${date} 續費`,
          noSubscriptionCaption: '沒有有效訂閱，付費模型會扣除儲值額度。'
        }
      },
      errors: {
        consentRequired: { title: '需要確認銀行卡', message: '請在入口網站中確認此卡可用於終端扣款' },
        insufficientScope: {
          title: '需要核准遠端支出',
          message: '此操作需要遠端支出權限。請發起一次儲值以授權，然後重試。'
        },
        remoteSpendingRevoked: {
          title: '遠端支出已停止',
          messageByAdmin: '管理員已停止此終端的遠端支出。',
          messageBySelf: '你已停止此終端的遠端支出。'
        },
        remoteSpendingReconnect: who => `${who} 請從「設定 → 閘道」重新連線以重新授權此裝置。`,
        sessionRevoked: { title: '工作階段已登出', message: '你的工作階段已登出。請從「設定 → 閘道」重新登入。' },
        cliBillingDisabled: {
          title: '遠端支出已關閉',
          message: '此帳戶的遠端支出已關閉，帳單管理員可在入口網站的 Hermes Agent 頁面開啟。'
        },
        roleRequired: {
          title: '需要管理員權限',
          message: '新增資金需要組織管理員或擁有者權限。請聯絡管理員，或在入口網站中管理。'
        },
        idempotencyConflict: { title: '請發起新的儲值', message: '🔴 此扣款識別碼已用於另一金額。請發起新的儲值。' },
        noPaymentMethod: {
          title: '沒有已儲存的銀行卡',
          message: '💳 尚未儲存用於終端扣款的銀行卡。請在入口網站中設定（一次性購買額度不會儲存可重複使用的卡）。'
        },
        orgAccessDenied: { title: '組織存取遭拒', message: '此權杖未綁定到你可管理的組織' },
        monthlyCapExceeded: {
          title: '已達每月支出上限',
          messageReached: '🔴 已達每月支出上限。',
          messageHeadroom: remaining => `🔴 已達每月支出上限，剩餘額度為 $${remaining}。`
        },
        rateLimited: {
          title: '目前扣款請求過多',
          message: mins => `🟡 目前扣款請求過多${mins > 0 ? `（請約 ${mins} 分鐘後重試）` : ''}。這不是付款失敗。`
        },
        stripeUnavailable: {
          title: 'Stripe 遇到問題',
          message: mins => (mins > 0 ? `Stripe 遇到問題，請約 ${mins} 分鐘後重試` : 'Stripe 遇到問題，請稍後重試')
        },
        upgradeCapExceeded: { title: '已達每日方案變更次數上限', message: '已達每日方案變更次數上限，請明天重試' },
        endpointUnavailable: {
          title: '帳單端點不可用',
          message: '帳單端點傳回了非 JSON 回應（此部署可能不支援該端點）。'
        },
        timeout: { title: '帳單請求逾時', message: '帳單請求逾時。' },
        transport: { title: '帳單連線失敗', message: '帳單請求在到達閘道前失敗。' },
        default: { title: '帳單請求失敗', message: '帳單請求失敗。' }
      }
    },
    providers: {
      connectAccount: '連結帳號',
      haveApiKey: '改用 API 金鑰？',
      intro: '使用訂閱登入，無需複製 API 金鑰。Hermes 會在應用程式中為您完成瀏覽器登入。',
      connected: '已連線',
      collapse: '收合',
      connectAnother: '連結其他提供方',
      otherProviders: '其他提供方',
      removeConfirm: provider => `移除 ${provider}？`,
      removeKeyManaged: provider => `${provider} 由 API 金鑰設定。請從 API Keys 中移除。`,
      removedTitle: '帳號已移除',
      removedMessage: provider => `${provider} 已移除。`,
      failedRemove: provider => `無法移除 ${provider}`,
      noProviderKeys: '沒有可用的提供方 API 金鑰。',
      searchKeys: '搜尋提供方…',
      noKeysMatch: '沒有符合的提供方。',
      localEndpoint: {
        title: '本地 / 自訂端點',
        description: '將 Hermes 指向任意 OpenAI 相容端點（Zyphra、vLLM、llama.cpp、Ollama 等）。'
      },
      loading: '正在載入提供方...'
    },
    sessions: {
      loading: '正在載入已封存工作階段…',
      archivedTitle: '已封存工作階段',
      archivedIntro: '已封存的聊天會從側邊欄隱藏，但保留全部訊息。在側邊欄 Alt/⌥+Shift 點擊聊天即可封存。',
      emptyArchivedTitle: '暫無封存',
      emptyArchivedDesc: '封存一個聊天後會顯示在這裡。',
      unarchive: '取消封存',
      deletePermanently: '永久刪除',
      messages: count => `${count} 則訊息`,
      restored: '已還原',
      deleteConfirm: title => `永久刪除「${title}」？此操作無法復原。`,
      autoArchiveTitle: '自動封存閒置對話',
      autoArchiveDesc:
        '自動封存你一段時間未使用的對話。已釘選的對話永遠不會被封存，也不會刪除任何內容——封存的對話會移到這裡。',
      autoArchiveDaysLabel: '封存前',
      autoArchiveDaysUnit: '天無活動',
      autoArchiveFailed: '無法更新自動封存設定',
      defaultDirTitle: '預設專案目錄',
      defaultDirDesc: '新工作階段預設從此資料夾開始，除非您選擇其他目錄。留空則使用您的家目錄。',
      defaultDirUpdated: '預設專案目錄已更新',
      defaultsTo: label => `預設使用 ${label}。`,
      change: '變更',
      choose: '選擇',
      clear: '清除',
      notSet: '未設定',
      failedLoad: '無法載入已封存工作階段',
      unarchiveFailed: '取消封存失敗',
      deleteFailed: '刪除失敗',
      updateDirFailed: '無法更新預設目錄',
      clearDirFailed: '無法清除預設目錄'
    },
    toolsets: {
      loadingConfig: '正在載入設定',
      savedTitle: '憑證已儲存',
      savedMessage: key => `${key} 已更新。`,
      removedTitle: '憑證已移除',
      removedMessage: key => `${key} 已移除。`,
      failedSave: key => `儲存 ${key} 失敗`,
      failedRemove: key => `移除 ${key} 失敗`,
      failedReveal: key => `顯示 ${key} 失敗`,
      removeConfirm: key => `從 .env 中移除 ${key}？`,
      set: '已設定',
      notSet: '未設定',
      selectedTitle: '已選擇提供方',
      selectedMessage: provider => `${provider} 現在處於作用中狀態。`,
      failedSelect: provider => `選擇 ${provider} 失敗`,
      failedLoad: '工具設定載入失敗',
      noProviderOptions: '此工具集沒有提供方選項；啟用後即可使用目前設定。',
      noProviders: '此工具集目前沒有可用提供方。',
      ready: '就緒',
      needsSignIn: '需要登入',
      needsSetup: '需要安裝',
      activeBackend: '目前後端',
      activeBackendHint: '這是你目前使用的後端',
      useBackend: '使用此後端',
      nousIncluded: '包含在 Nous 訂閱中；登入 Nous Portal 即可啟用。',
      nousAuthNeededTitle: '登入 Nous Portal',
      nousAuthNeededMessage: provider => `已儲存 ${provider}，但在登入 Nous Portal 之前不會啟用。`,
      nousAuthSignIn: '登入',
      nousAuthDoneTitle: '已連接 Nous Portal',
      nousAuthDoneMessage: '訂閱後端現已啟用。',
      nousAuthFailed: 'Nous Portal 登入未完成',
      noApiKeyRequired: '不需要 API 金鑰。',
      postSetupHint: step => `此後端需要一次性安裝 (${step})。將在此機器上執行，可能需要幾分鐘。`,
      postSetupInstalledHint: '已安裝。僅在出現問題時才需要重新執行安裝。',
      postSetupRun: '執行設定',
      postSetupRerun: '重新執行設定',
      postSetupInstalled: '已安裝',
      postSetupRunning: '安裝中…',
      postSetupStarting: '啟動中…',
      postSetupCompleteTitle: '設定完成',
      postSetupCompleteMessage: step => `已安裝 ${step}。`,
      postSetupErrorTitle: '設定完成但有錯誤',
      postSetupErrorMessage: step => `請檢查 ${step} 日誌。`,
      postSetupFailed: step => `執行 ${step} 設定失敗`,
      webSearchActive: backend => `搜尋：${backend}`,
      webExtractActive: backend => `擷取：${backend}`,
      webCapabilityUnset: '未設定',
      webUseForSearch: '用於搜尋',
      webUseForExtract: '用於擷取',
      webUsedForSearch: '搜尋後端',
      webUsedForExtract: '擷取後端',
      webCapabilitySelectedMessage: (provider, capability) =>
        `${provider} 現在負責網頁${capability === 'search' ? '搜尋' : '擷取'}。`,
      failedSelectCapability: provider => `無法設定 ${provider}`,
      terminalBackend: {
        sectionTitle: '執行後端',
        loading: '正在檢查執行後端…',
        failedLoad: '無法載入終端後端',
        ready: '就緒',
        needsSetup: '需要設定',
        unavailable: '不可用',
        inUse: '使用中',
        selectedTitle: '已選擇後端',
        selectedMessage: backend => `終端命令現在透過 ${backend} 執行。將套用於新工作階段。`,
        failedSelect: backend => `選擇 ${backend} 失敗`,
        needsSetupHint: '此後端已選取但尚未完成設定——在設定完成前命令將會失敗。',
        needsSetupConfirmTitle: backend => `仍要選擇 ${backend} 嗎？`,
        needsSetupConfirmDescription: detail =>
          `${detail} 此變更生效後啟動的工作階段在設定完成前將沒有終端或檔案工具。`,
        needsSetupConfirmDescriptionGeneric:
          '此後端尚未完成設定。此變更生效後啟動的工作階段在設定完成前將沒有終端或檔案工具。',
        needsSetupConfirmAction: '仍然選擇'
      },
      browserRealProfile: {
        label: '使用我的真實瀏覽器設定檔',
        description:
          '將預設瀏覽器的登入資訊與 Cookie 複製到受管理的快照中，代理使用該快照進行瀏覽。絕不會直接開啟你的真實設定檔。將套用於新工作階段。',
        enabledTitle: '真實設定檔瀏覽：已開啟',
        enabledMessage: '新工作階段將使用預設瀏覽器設定檔的快照進行瀏覽。',
        disabledTitle: '真實設定檔瀏覽：已關閉',
        disabledMessage: '設定檔快照將被刪除；新工作階段使用乾淨的瀏覽器。',
        failedSave: '無法儲存真實設定檔設定',
        prompt: {
          title: '讓網站保持登入狀態',
          body: '讓 Hermes 使用預設瀏覽器設定檔的快照進行瀏覽，網站開啟時即已登入。',
          bulletSnapshot: 'Cookie 與登入資訊會複製到受管理的快照中。',
          bulletLiveProfile: '絕不會直接開啟你的真實瀏覽器設定檔。',
          bulletLocal: '所有資料都不會離開這台電腦。',
          dontShowAgain: '不再顯示',
          notNow: '暫不',
          enable: '使用我的設定檔'
        }
      }
    }
  },

  modelPicker: {
    title: '切換模型',
    current: '目前：',
    unknown: '（未知）',
    search: '篩選提供方和模型...',
    noModels: '找不到模型。',
    addProvider: '新增提供方',
    loadFailed: '無法載入模型',
    downloading: '下載中',
    localDownloadsHeading: '本地',
    noAuthenticatedProviders: '沒有已驗證的提供方。',
    pro: 'Pro',
    proNeedsSubscription: 'Pro 模型需要付費 Nous 訂閱。',
    free: '免費',
    freeTier: '免費層',
    priceTitle: '每百萬 Token 的輸入/輸出價格',
    wasPrice: '原價',
    customModel: '自訂模型',
    addCustomModelAction: '新增自訂模型…',
    customModelPlaceholder: '輸入模型 ID，例如 openai/gpt-5'
  },

  modelVisibility: {
    title: '模型',
    search: '搜尋模型',
    noAuthenticatedProviders: '沒有已驗證的提供方。',
    addProvider: '新增提供方…',
    addCustomModel: '新增自訂模型',
    removeCustomModel: '移除自訂模型',
    resetToDefaults: '恢復預設',
    resetConfirm: '將模型可見性恢復為預設？',
    resetDescription: '你對模型顯示與隱藏的選擇將被清除，每個提供方都會恢復預設清單。你新增的自訂模型會保留並顯示。',
    resetAction: '恢復'
  }
} satisfies Pick<TranslationOverrides, 'language' | 'settings' | 'modelPicker' | 'modelVisibility'>
