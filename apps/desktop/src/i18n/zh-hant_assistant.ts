import type { TranslationOverrides } from './define-locale'

export const zhHantAssistant = {
  assistant: {
    catalogInstall: {
      preparing: '正在準備安裝…',
      install: '安裝',
      advanced: '進階',
      skip: '略過',
      installing: '正在安裝…',
      installed: '已安裝',
      notInstalled: '未安裝',
      failed: '失敗',
      showNames: '顯示名稱',
      hideNames: '隱藏名稱',
      skill: name => `技能 ${name}`,
      kind: { plugin: '外掛', skill: '技能' },
      tier: { official: '官方', community: '社群' },
      targetProfile: profile => `安裝到你的 ${profile} 設定檔`,
      sendFailed: '無法傳送你的回覆，請再試一次。',
      commitLabel: '提交',
      subdirLabel: '資料夾',
      securityHeading: '安全性',
      scan: { passed: '掃描通過', warnings: '掃描發現警告', failed: '掃描未通過' },
      requirementsLabel: '需求',
      credentialsHeading: '憑證'
    },
    thread: {
      loadingSession: '正在載入工作階段',
      showEarlier: '顯示較早的訊息',
      loadingResponse: 'Hermes 正在載入回覆',
      resumeWhenBackgroundDone: count =>
        count === 1 ? '背景工作完成後將自動繼續' : `${count} 個背景工作完成後將自動繼續`,
      thinking: '思考中',
      thought: '已思考',
      thoughtBriefly: '思考了片刻',
      thoughtFor: duration => `思考了 ${duration}`,
      turnDuration: duration => `本輪耗時 ${duration}`,
      turnStats: '本輪統計',
      turnStatsInput: '輸入',
      turnStatsOutput: '輸出',
      turnStatsReasoning: '推理',
      turnStatsCached: '快取',
      turnStatsHit: n => `命中 ${n}`,
      turnStatsTime: '耗時',
      turnStatsCalls: '呼叫',
      turnStatsCost: '費用',
      turnStatsEstimate: '估算',
      today: time => `今天，${time}`,
      yesterday: time => `昨天，${time}`,
      copy: '複製',
      refresh: '重新整理',
      moreActions: '更多動作',
      branchNewChat: '在新聊天中分支',
      react: '回應',
      dismissError: '关闭错误',
      errorGenericProvider: 'AI 服務',
      errorLayerBodies: {
        generic: 'Hermes 回覆時發生問題。請重試；若問題持續，請複製錯誤詳細資訊。',
        provider: 'AI 服務無法完成此請求。請稍後重試或切換服務商。',
        endpoint: 'Hermes 無法連線至你的自訂模型伺服器。請確認它正在執行，然後重新傳送訊息。',
        streaming: '回覆完成前連線已中斷。請重試以重新傳送。'
      },
      errorCodes: {
        provider_policy_blocked: {
          title: '帳戶設定封鎖了此模型',
          body: provider => `${provider} 無法依你帳戶的資料或隱私設定路由此請求。請選擇其他模型或切換服務商。`
        },
        content_policy_blocked: {
          title: 'AI 服務拒絕回答此請求',
          body: provider => `${provider} 拒絕回答這則訊息。請修改後重新傳送。`
        },
        format_error: {
          title: 'AI 服務拒絕了請求格式',
          body: provider => `${provider} 不接受此請求的建構方式。請切換服務商，或傳送診斷資訊以便我們排查。`
        },
        invalid_response: {
          title: 'AI 服務傳回了無法讀取的回覆',
          body: provider => `${provider} 傳回了 Hermes 無法讀取的內容。請稍後重試。`
        },
        empty_response: {
          title: 'AI 服務傳回了空回覆',
          body: provider => `${provider} 沒有為此訊息傳回內容。請稍後重試。`
        },
        rate_limit: {
          title: 'AI 服務忙碌中',
          body: provider => `${provider} 正在限制請求數量。請稍等片刻後重試。`
        },
        upstream_rate_limit: {
          title: 'AI 服務忙碌中',
          body: provider => `${provider} 正在限制請求數量。請稍等片刻後重試。`
        },
        overloaded: {
          title: 'AI 服務負載過高',
          body: provider => `${provider} 目前遇到問題。請稍後重試或切換服務商。`
        },
        server_error: {
          title: 'AI 服務發生錯誤',
          body: provider => `${provider} 傳回了伺服器錯誤。請稍後重試或切換服務商。`
        },
        timeout: {
          title: '無法連線到 AI 服務',
          body: provider => `無法連線到 ${provider}，或其未及時回應。請檢查網路連線後重試。`
        },
        ssl_cert_verification: {
          title: '安全連線失敗',
          body: provider => `Hermes 無法驗證與 ${provider} 的安全連線。請檢查網路或代理設定，或切換服務商後重新傳送。`
        }
      },
      errorLayers: {
        auth: '認證錯誤',
        billing: '額度不足',
        disk: '磁碟已滿',
        endpoint: '自訂端點錯誤',
        gateway: '閘道錯誤',
        generic: '本輪失敗',
        provider: '模型服務商錯誤',
        runtime: '本機執行環境錯誤',
        streaming: '串流連線錯誤'
      },
      errorRetry: '重試',
      errorLimitResets: time => `限額將於 ${time} 重設`,
      errorRetryAtReset: time => `限額重設後重試（${time}）`,
      errorRetryScheduled: (time, wait) => `將於 ${time} 重試 — 還剩 ${wait}`,
      errorRetryScheduledCancel: '取消',
      errorStartNewSession: '開始新工作階段',
      errorSwitchProvider: '切換服務商',
      errorSignInAgain: provider => `重新登入 ${provider}`,
      errorOauthExpired: provider => `您的 ${provider} 登入已過期或被撤銷。請重新登入以繼續對話。`,
      errorOpenLogs: '開啟日誌',
      errorOpenLogsFailed: '無法開啟日誌資料夾',
      errorOpenDesktopLogs: '開啟桌面端日誌',
      errorCopyDiagnostics: '複製錯誤詳細資訊',
      errorSendDiagnostics: '傳送診斷資訊',
      filesChanged: count => `${count} 個檔案已變更`,
      reviewChanges: '檢視',
      readAloudFailed: '朗讀失敗',
      preparingAudio: '正在準備音訊...',
      stopReading: '停止朗讀',
      readAloud: '朗讀',
      copyFullResponse: '複製完整回覆',
      readAloudFullResponseHint: '按住 Shift 點擊：朗讀完整回覆',
      editMessage: '編輯訊息',
      stop: '停止',
      restorePrevious: '還原至上一個檢查點',
      restoreCheckpoint: '還原檢查點',
      restoreFromHere: '還原檢查點 — 從此提示重新執行',
      restoreTitle: '還原至此檢查點？',
      restoreBody: '此提示之後的所有訊息將從對話中移除，並從此處重新執行該提示。',
      restoreConfirm: '還原並重新執行',
      restoreNext: '還原至下一個檢查點',
      goForward: '前進',
      sendEdited: '傳送編輯後的訊息',
      attachingFile: '正在附加…'
    },
    approval: {
      gatewayDisconnected: 'Hermes 閘道未連線',
      sendFailed: '無法傳送核准回應',
      run: '執行',
      command: '指令',
      moreOptions: '更多核准選項',
      allowSession: '允許本工作階段',
      alwaysAllowMenu: '一律允許…',
      jumpToApproval: '需要核准',
      reject: '拒絕',
      alwaysTitle: '一律允許此指令？',
      alwaysDescription: pattern =>
        `這會將「${pattern}」模式加入永久允許清單（~/.hermes/config.yaml）。Hermes 對類似指令將不再詢問，包括目前工作階段和未來工作階段。`,
      alwaysAllow: '一律允許'
    },
    clarify: {
      notReady: '澄清請求尚未就緒',
      gatewayDisconnected: 'Hermes 閘道未連線',
      sendFailed: '無法傳送澄清回應',
      loadingQuestion: '正在載入問題…',
      other: '其他（輸入您的答案）',
      placeholder: '輸入您的答案…',
      skip: '略過',
      skipped: '已略過',
      noAnswer: '未回答',
      confirmAndContinueLabel: '確認並繼續',
      singleSelectHint: '選一個',
      multiSelectHint: '可多選',
      questionProgress: (answered, total) => `已回答 ${answered}/${total}`,
      notDelivered: '此問題未送達應用程式，無法在此回答。請按停止結束本輪，然後在聊天中回覆。'
    },
    tool: {
      copyCode: '複製程式碼',
      renderingImage: '正在渲染圖片',
      copyOutput: '複製輸出',
      copyCommand: '複製指令',
      copyContent: '複製內容',
      copyUrl: '複製 URL',
      copyResults: '複製結果',
      copyQuery: '複製查詢',
      copyFile: '複製檔案',
      copyPath: '複製路徑',
      failedCalls: (count: number) => `${count} 次工具呼叫失敗`,
      skillActivity: {
        loading: '正在載入技能',
        loaded: '已載入技能',
        loadFailed: '技能載入失敗',
        readingResource: '正在讀取技能資源',
        readResource: '已讀取技能資源',
        resourceFailed: '技能資源讀取失敗',
        listing: '正在列出技能',
        listed: '已列出技能',
        listFailed: '技能清單取得失敗',
        unavailable: '技能結果無法使用'
      },
      outputAlt: '工具輸出',
      rawResponse: '原始回應',
      copyActivity: '複製活動',
      recoveredOne: '在 1 個失敗步驟後已復原',
      recoveredMany: count => `在 ${count} 個失敗步驟後已復原`,
      failedOne: '1 個步驟失敗',
      failedMany: count => `${count} 個步驟失敗`,
      statusRunning: '執行中',
      statusError: '錯誤',
      statusRecovered: '已復原',
      statusDone: '完成',
      resultUnavailable: '結果無法使用',
      resultInterrupted: '已中斷',
      memoryWriteNoted: '已記下記憶寫入',
      actions: {
        read: '已讀取',
        reading: '正在讀取',
        opened: '已開啟',
        opening: '正在開啟',
        failedToOpen: '開啟失敗',
        searched: '已搜尋',
        searching: '正在搜尋',
        ran: '已執行',
        running: '正在執行',
        ranCode: '已執行程式碼',
        runningCode: '正在撰寫腳本'
      },
      prefixes: {
        browser: '瀏覽器',
        web: '網頁'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action}「${value}」`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix}${action}`,
        runningPrefixedTool: (prefix, action) => `正在執行${prefix}${action}`,
        runningTool: action => `正在執行 ${action}`
      },
      titles: {
        browser_click: { done: '已點擊頁面元素', pending: '正在點擊頁面元素', pendingAction: '正在點擊' },
        browser_fill: { done: '已填寫表單欄位', pending: '正在填寫表單欄位', pendingAction: '正在填寫' },
        browser_navigate: { done: '已開啟頁面', pending: '正在開啟頁面', pendingAction: '正在開啟' },
        browser_snapshot: { done: '已擷取頁面快照', pending: '正在擷取頁面快照', pendingAction: '正在擷取' },
        browser_take_screenshot: { done: '已擷取截圖', pending: '正在擷取截圖', pendingAction: '正在擷取' },
        browser_type: { done: '已在頁面輸入', pending: '正在頁面輸入', pendingAction: '正在輸入' },
        clarify: { done: '已提問', pending: '正在提問', pendingAction: '正在提問' },
        cronjob: { done: 'Cron 工作', pending: '正在安排 Cron 工作', pendingAction: '正在安排' },
        edit_file: { done: '已編輯檔案', pending: '正在編輯檔案', pendingAction: '正在編輯' },
        execute_code: { done: '已執行程式碼', pending: '正在撰寫腳本', pendingAction: '正在撰寫腳本' },
        image_generate: { done: '已生成圖片', pending: '正在生成圖片', pendingAction: '正在生成' },
        list_files: { done: '已列出檔案', pending: '正在列出檔案', pendingAction: '正在列出' },
        memory: { done: '已儲存至記憶', pending: '正在儲存至記憶', pendingAction: '正在儲存' },
        patch: { done: '已修補檔案', pending: '正在修補檔案', pendingAction: '正在修補' },
        read_file: { done: '已讀取檔案', pending: '正在讀取檔案', pendingAction: '正在讀取' },
        search_files: { done: '已搜尋檔案', pending: '正在搜尋檔案', pendingAction: '正在搜尋' },
        session_search_recall: {
          done: '已搜尋工作階段歷史',
          pending: '正在搜尋工作階段歷史',
          pendingAction: '正在搜尋'
        },
        terminal: { done: '已執行指令', pending: '正在執行指令', pendingAction: '正在執行' },
        todo: { done: '已更新待辦', pending: '正在更新待辦', pendingAction: '正在更新' },
        vision_analyze: { done: '已分析圖片', pending: '正在分析圖片', pendingAction: '正在分析' },
        web_extract: { done: '已讀取網頁', pending: '正在讀取網頁', pendingAction: '正在讀取' },
        web_search: { done: '已搜尋網頁', pending: '正在搜尋網頁', pendingAction: '正在搜尋' },
        write_file: { done: '已編輯檔案', pending: '正在編輯檔案', pendingAction: '正在編輯' }
      }
    }
  }
} satisfies Pick<TranslationOverrides, 'assistant'>
