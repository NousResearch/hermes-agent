import type { TranslationOverrides } from './define-locale'

export const zhHantDiagnostics = {
  notifications: {
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
    backendOutOfDateMessage: '您的 Hermes 後端早於目前的桌面版本，可能無法正常運作。請更新以保持一致。',
    installMethodUnsupportedTitle: '不受支援的安裝方式',
    updateHermes: '更新 Hermes',
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
      methodNotAllowed: '桌面後端拒絕了該請求 (405 Method Not Allowed)。請嘗試重新啟動 Hermes Desktop。',
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
      inputBody: 'Hermes 正在等待你的回應。',
      turnDoneTitle: 'Hermes 已完成',
      turnDoneBody: '',
      turnErrorTitle: '本輪失敗',
      backgroundDoneTitle: '背景工作已完成',
      backgroundFailedTitle: '背景工作失敗',
      creditsTitle: '額度'
    }
  },

  sendDiagnostics: {
    title: '向 Nous 傳送診斷資訊',
    privacyNotice:
      '這會將偵錯套件上傳到 Nous 內部儲存空間（並非公開貼上板）。內容包括系統資訊（作業系統、版本、服務商、已設定的 API 金鑰種類 — 絕不包含金鑰本身）以及完整的 agent、gateway 與桌面端日誌（每個最多 512 KB，很可能包含對話內容、工具輸出與檔案路徑）。上傳前會先遮罩機密資訊。僅 Nous 員工與獲准的 Discord 版主可檢視，14 天後自動刪除。',
    upload: '上傳',
    uploading: '上傳中…',
    cancel: '取消',
    close: '關閉',
    copyLink: '複製連結',
    uploadIdFallback: id => `未回傳檢視連結 — 請向支援人員提供上傳 ID ${id}`,
    doneTitle: '診斷資訊已傳送',
    doneDescription: '偵錯套件已私密上傳。在您的支援討論串中分享以下連結，團隊即可檢視您的日誌。',
    failedTitle: '上傳失敗',
    failedHint:
      '您也可以在終端機執行 `hermes debug share --nous`，或執行 `hermes debug share --local` 在不上傳的情況下檢視報告。',
    handoffLead: '在以下位置繼續討論:',
    links: {
      github: 'GitHub Issues',
      portal: 'Nous Portal 支援',
      discord: 'Discord'
    }
  },

  errors: {
    genericFailure: '發生錯誤',
    boundaryTitle: '介面出現問題',
    boundaryDesc: '此檢視遇到意外錯誤。您的聊天和設定是安全的。',
    reloadWindow: '重新載入視窗',
    openLogs: '開啟記錄'
  },
} satisfies Pick<TranslationOverrides, 'notifications' | 'sendDiagnostics' | 'errors'>
