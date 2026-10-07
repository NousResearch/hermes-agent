import type { TranslationOverrides } from './define-locale'

export const zhHantDiagnostics = {
  notifications: {
    sharedProfileWarning:
      '另一個 Rabbit 安裝實例正在使用此設定檔。兩個實例共用此設定檔的設定和資料，因此變更可能發生衝突。你可以繼續使用，也可以在變更前關閉另一個實例。',
    region: '通知',
    hide: '隱藏',
    show: '顯示',
    more: count => `另外 ${count} 則通知`,
    clearAll: '全部清除',
    dismiss: '關閉通知',
    details: '詳細資訊',
    copyDetail: '複製詳情',
    copyDetailFailed: '無法複製通知詳情',
    backendOutOfDateTitle: '後端版本過舊',
    backendOutOfDateMessage: '您的 Rabbit 後端早於目前的桌面版本，可能無法正常運作。請更新以保持一致。',
    desktopOutOfDateTitle: '應用程式版本過舊',
    desktopOutOfDateMessage: '此 Rabbit 應用程式早於所連接的後端，可能無法正常運作。請更新應用程式以保持一致。',
    updateDesktopApp: '更新應用程式',
    installMethodUnsupportedTitle: '不受支援的安裝方式',
    updateRabbit: '更新 Rabbit',
    updateReadyTitle: '有可用更新',
    updateReadyMessage: count => `有 ${count} 項新變更可用。`,
    updateReadyMessageUnknown: '有新更新可用。',
    seeWhatsNew: '查看新增內容',
    mcp: {
      needsAuthTitle: 'MCP 伺服器需要重新驗證',
      needsAuthMessage: name => `${name} MCP 需要重新驗證。`,
      errorTitle: 'MCP 伺服器無法連線',
      errorMessage: name => `${name} MCP 健康檢查失敗。`,
      signIn: '登入',
      view: '檢視',
      disable: '停用',
      disabledMessage: name => `已停用 ${name} MCP。可隨時在「功能 → MCP」重新啟用。`,
      disableFailed: name => `無法停用 ${name} MCP。`
    },
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT 需要 ELEVENLABS_API_KEY。',
      elevenLabsRejectedKey: 'ElevenLabs 拒絕了該 API 金鑰 (401)。',
      diskFull: '磁碟已滿 — 請騰出一些空間後再試。',
      gatewayAuthFailed: '閘道認證失敗 — 請檢查你的 API_SERVER_KEY。',
      methodNotAllowed: '桌面後端拒絕了該請求 (405 Method Not Allowed)。請嘗試重新啟動 Rabbit Desktop。',
      microphonePermission: '麥克風權限已被拒絕。',
      openaiRejectedApiKey: 'OpenAI 拒絕了該 API 金鑰。',
      openaiTtsNeedsKey: 'OpenAI TTS 需要 VOICE_TOOLS_OPENAI_KEY 或 OPENAI_API_KEY。',
      codeSkewRestartRequired: '更新後此後端仍在執行舊程式碼。請重新啟動以載入新程式碼。'
    },
    voice: {
      configureSpeechToText: '設定語音轉文字後即可使用語音模式。',
      couldNotStartSession: '無法啟動語音工作階段',
      microphoneAccessDenied: '麥克風存取被拒絕。',
      microphoneConstraintsUnsupported: '此裝置不支援目前的麥克風限制條件。',
      microphoneFailed: '麥克風發生錯誤',
      microphoneInUse: '麥克風正被其他應用程式使用中。',
      microphonePermissionDenied: '麥克風權限被拒絕。',
      microphoneStartFailed: '無法開始麥克風錄音。',
      microphoneUnsupported: '目前執行環境不支援麥克風錄音。',
      noMicrophone: '找不到麥克風。',
      noSpeechDetected: '未偵測到語音',
      playbackFailed: '語音播放失敗',
      recordingFailed: '語音錄製失敗',
      sayStopToEnd: phrase => `說「${phrase}」即可結束語音對話。`,
      transcriptionFailed: '語音轉寫失敗',
      transcriptionUnavailable: '語音轉寫暫不可用。',
      tryRecordingAgain: '請再錄製一次。',
      unavailable: '語音不可用'
    },
    native: {
      approvalTitle: '需要核准',
      approvalTitleNamed: session => `需要核准 — ${session}`,
      approveAction: '核准',
      rejectAction: '拒絕',
      inputTitle: '需要輸入',
      inputTitleNamed: session => `需要輸入 — ${session}`,
      inputBody: 'Rabbit 正在等待你的回應。',
      turnDoneTitle: 'Rabbit 已完成',
      turnDoneBody: '',
      turnErrorTitle: '本輪失敗',
      backgroundDoneTitle: '背景工作已完成',
      backgroundFailedTitle: '背景工作失敗',
      creditsTitle: '額度'
    }
  },


  errors: {
    genericFailure: '發生錯誤',
    boundaryTitle: '介面出現問題',
    boundaryDesc: '此檢視遇到意外錯誤。您的聊天和設定是安全的。',
    reloadWindow: '重新載入視窗',
    openLogs: '開啟記錄'
  }
} satisfies Pick<TranslationOverrides, 'notifications' | 'errors'>
