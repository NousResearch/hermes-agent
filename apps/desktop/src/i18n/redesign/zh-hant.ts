import type { RedesignOverlay } from './types'

// Traditional Chinese wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const zhHantRedesign: RedesignOverlay = {
  sharedMetrics: {
    collectedMilestones: '設定：完成了哪些步驟、提供方連線、技能、外掛和例行任務的數量',
    collectedUsage: '智慧體品質：未成功的編輯、損壞的工具呼叫、卡住的迴圈、每個任務的成本',
    sending:
      '如果你選擇分享，Hermes 每天傳送一次統計資料給 Nous，並附帶此智慧體的隨機 ID。除了一則 Hermes 已安裝的一次性記錄之外，你同意之前的統計資料永遠不會被傳送。你可以隨時在設定中變更。',
    sendDesc: '僅限計數，每天傳送一次，並附帶此智慧體的隨機 ID。絕不包含你的訊息、檔案、路徑或錯誤文字。',
    consentBody:
      'Hermes 可以統計你的使用方式：對話長度、執行了哪些模型和工具，以及何時發生失敗。它絕不記錄你的訊息、檔案、路徑或錯誤文字。',
    collectedActivity: '對話：長度、結果、錯誤類型、每日活躍時間'
  },
  common: {
    bots: '智慧體'
  },
  boot: {
    steps: {
      connectingGateway: '正在連線機器',
      loadingSessions: '正在載入最近對話'
    },
    errors: {
      gatewayConnectionLost: '與機器的連線已中斷',
      gatewayConnectionLostDetail: '仍在背景重試。你可以繼續閱讀和撰寫草稿；如果問題持續，請開啟機器設定。',
      gatewaySignInRequired: '需要機器登入'
    },
    failure: {
      description: 'Hermes 的背景服務未啟動。請嘗試下面的復原步驟。這裡的操作不會刪除您的對話或設定。',
      remoteTitle: '需要重新登入遠端機器',
      remoteDescription: '您的遠端機器工作階段已過期。請重新登入以重新連線。這裡的操作不會刪除您的對話或設定。',
      useLocalGateway: '使用本機',
      gatewaySettings: '機器設定',
      remoteSignInHint: signInLabel =>
        `先登出已儲存的遠端瀏覽器工作階段，然後開啟${signInLabel}。使用本機可切換至內建後端。`,
      remoteFailureHint: '在「機器設定」中檢查機器 URL 與登入，或切換至本機。',
      cloudDownTitle: 'Nous Cloud 智慧體已停機',
      cloudDownDescription:
        '此機器連線的 Nous 託管雲端智慧體正在回傳伺服器錯誤。無法在此處重新啟動——請檢查其狀態、切換至本機，或取得支援。',
      signedInMessage: '正在重新連線至遠端機器…',
      signInToRemoteGateway: '登入遠端機器'
    }
  },
  notifications: {
    sharedProfileWarning:
      '另一個 Hermes 安裝實例正在使用此智慧體。兩個實例共用此智慧體的設定和資料，因此變更可能發生衝突。你可以繼續使用，也可以在變更前關閉另一個實例。',
    errors: {
      gatewayAuthFailed:
        '此 Hermes 不再接受你儲存的登入資訊。請開啟「機器」並重新登入（或貼上新的存取權杖），然後重試。'
    },
    voice: {
      couldNotStartSession: '無法啟動語音對話'
    },
    native: {
      inputBody: '智慧體正在等待你的回應。',
      turnDoneTitle: '智慧體已完成'
    }
  },
  promptNotices: {
    desktopCommands: {
      agentsEmpty: '此對話中沒有作用中的子智慧體或背景程序。',
      subagentsHeader: (count: string) => `子智慧體（${count}）：`,
      subagentFallback: '子智慧體',
      configFooter: '請在「設定」中變更，或在連線的機器上用 `hermes config edit` 編輯 config.yaml。',
      saveLegacyRefused:
        '未儲存：此機器無法匯出遮蔽、指定檔名或非 JSON 的副本。請更新 Hermes，或不帶選項執行 /save，在連線的機器上寫入原始 JSON。',
      savedOnHost: (path: string) => `已儲存到連線的機器：${path}`,
      exportRemote:
        '/export 會從連線的機器打包智慧體，而連線的機器不是這台電腦。請在該機器上執行 `hermes profile export`。',
      importRemote:
        '/import 會在連線的機器上安裝智慧體，而連線的機器不是這台電腦。請在該機器上執行 `hermes profile import <archive>`。',
      historyHermes: '智慧體',
      checkpointsDisabled: '此對話未啟用檢查點（在 config.yaml 中設定 checkpoints.enabled）。'
    },
    sessionCommands: {
      busy: (command: string) =>
        `對話忙碌中 — 智慧體仍在回覆。請先停止目前回覆（Stop 按鈕，或在終端機中按 Ctrl+C），然後再執行 /${command}。`,
      busyCompressing: (command: string) => `對話忙碌中 — 正在執行 /compress；完成後再執行 /${command}。`,
      titleSet: (title, queued) => `已設定對話標題：${title}${queued ? '（對話開始後生效）' : ''}`,
      titleCleared: '已清除對話標題。'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      '這會將偵錯套件上傳到 Nous 內部儲存空間（並非公開貼上板）。內容包括系統資訊（作業系統、版本、服務商、已設定的 API 金鑰種類 — 絕不包含金鑰本身）以及完整的智慧體、Hermes 服務與桌面端日誌（每個最多 512 KB，很可能包含對話內容、工具輸出與檔案路徑）。上傳前會先遮罩機密資訊。僅 Nous 員工與獲准的 Discord 版主可檢視，14 天後自動刪除。'
  },
  settings: {
    subpages: {
      modelMoa: '多智慧體協作',
      advancedRuntime: '智慧體限制',
      advancedDelegation: '子智慧體',
      gatewayConnection: '主要機器',
      appearanceChatDisplay: '對話顯示'
    },
    pluginPages: {
      agentSettings: '智慧體設定'
    },
    nav: {
      gateway: '機器',
      archivedChats: '已封存對話',
      sessions: '對話'
    },
    plugins: {
      installModal: {
        partialInstall: sha =>
          sha
            ? `外掛只安裝了一部分：智慧體部分已安裝在提交 ${sha}，但由於儲存庫在你審閱後有變更，桌面部分未能隨之安裝。在「進階」中開啟強制重新安裝，可用顯示的版本取代兩部分；或取消，只保留智慧體部分。`
            : '外掛只安裝了一部分：智慧體部分已安裝，但由於儲存庫在你審閱後有變更，桌面部分未能隨之安裝。在「進階」中開啟強制重新安裝，可用顯示的版本取代兩部分；或取消，只保留智慧體部分。',
        nextChat: '更多工具將在下一次對話中可用'
      }
    },
    vault: {
      blurb:
        '說一句「登入 GitHub」，智慧體就會代你登入。第一次遇到登入頁時它會當場向你索取登入資訊，之後就自動完成。密碼在本機加密儲存並直接填入頁面——模型永遠看不到。',
      emptyDesc: '這裡不必手動新增。讓智慧體登入某個網站時，它會當場向你詢問一次登入資訊。若想提前輸入，可按「新增」。',
      addDescription: '加密儲存在此裝置上。智慧體永遠不會看到密碼。',
      sources: {
        blurb:
          '已安裝的密碼管理器會被自動偵測。智慧體第一次需要其中的登入資訊時會請你解鎖（每個工作階段一次）；記憶體中只保留工作階段權杖，智慧體永遠看不到你的主密碼或任何登入資訊。',
        lockedDesc: '已偵測到。智慧體需要登入資訊時會請你解鎖，也可立即解鎖。',
        unlockDescription: '輸入主密碼。它會交給本機的密碼管理器後立即捨棄，不會被儲存、記錄或顯示給智慧體。'
      },
      otpHint: '啟用兩步驟驗證時網站顯示的「設定金鑰」。儲存後智慧體會自動產生驗證碼。'
    },
    notifications: {
      completionSoundDesc: '智慧體回合結束時播放。可在此選擇預設並預覽。',
      kinds: {
        input: {
          description: '智慧體提出了問題，或需要密碼或密鑰。'
        }
      }
    },
    searchPlaceholder: {
      gateway: '機器連線…',
      sessions: '搜尋已封存對話…'
    },
    appearance: {
      themeProfileNote: profile => `已為智慧體「${profile}」儲存——每個智慧體保留各自的主題。`,
      chatTextScaleTitle: '對話文字大小',
      intro: '這些是僅限桌面端的顯示偏好。模式控制亮度；主題控制強調色與對話介面樣式。',
      sessionDensityTitle: '對話列表密度',
      sessionDensityDesc: '選擇側邊欄對話標題下方顯示的資訊量。',
      tabStripDesc: '在分區上方顯示分頁。自動模式會在分區只有一個面板時隱藏分頁，除非還開著其他對話或磚塊分區。',
      chatFontTitle: '對話字型',
      chatFontDesc: '為對話與應用程式介面選擇已安裝的字型，適合 OpenDyslexic 等易讀字型；留空則使用主題字型。',
      backdropTitle: '對話背景',
      textDirectionDesc:
        '設定對話訊息和輸入框的文字方向。「自動」依每段的第一個字母判斷；混合文字排列不對時，可手動選擇方向。程式碼一律從左到右顯示。',
      reactionsDesc: 'iMessage 風格的表情回應 — 你可以對訊息做出回應，智慧體也能回應你的訊息。',
      tipsDesc:
        '偶爾顯示來自應用程式和你的智慧體的提示，每則提示只出現一次。開始使用滿30天後自動關閉，你可以重新開啟。',
      toursDesc: '讓智慧體逐步標示每個位置，帶你認識應用程式。開始使用滿30天後自動關閉，你可以重新開啟。',
      fileBrowserDesc: '開啟工作區時，在對話旁顯示檔案瀏覽器。標題列的切換按鈕也會變更此設定。',
      resumeLastSessionDesc: '開啟後，應用冷啟動時重新打開最近的對話。關閉則始終從空白新對話開始。',
      pet: {
        intro:
          '領養一隻懸浮在應用上的 petdex 動畫寵物，它會根據智慧體的狀態做出反應——工具執行時奔跑、成功時歡呼、出錯時沮喪。'
      }
    },
    uninstallSection: {
      options: {
        gui: {
          description: '移除此桌面應用程式。Hermes 智慧體、你的設定和對話都會保留。',
          title: '僅解除安裝對話 GUI',
          consequence: '桌面對話 GUI（此應用程式及其資料）'
        },
        lite: {
          title: '解除安裝 GUI 與智慧體，保留資料',
          description: '移除應用程式和 Hermes 智慧體，但保留設定、對話和機密，以便日後重新安裝。',
          consequence: '對話 GUI 和 Hermes 智慧體（設定、對話和機密會保留）'
        },
        full: {
          description: '移除應用程式、智慧體和所有使用者資料——設定、對話、例行任務、機密和日誌。',
          consequence: '全部內容——對話 GUI、Hermes 智慧體，以及你的所有設定、對話、機密和日誌'
        }
      }
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `此版本比穩定版 ${v} 更新，且你的對話已在使用 ${v} 無法開啟的 Hermes 執行環境。在更新的穩定版發布前，Hermes 會維持目前版本。`
      }
    },
    gateway: {
      loading: '正在載入機器設定...',
      unavailableTitle: '機器設定不可用',
      unavailableDesc: '桌面 IPC 橋接器未公開機器設定。',
      webappHostDesc:
        'Webapp 一律在提供它的機器上執行。下方已儲存的連線是這台機器的清單，與 Hermes Desktop 應用程式共用；其他機器透過這台機器連線。',
      webappOauthPaste: {
        description: '如果登入分頁停在無法開啟的位址（此瀏覽器不在連線的機器上），請將該位址貼到這裡。'
      },
      webAccess: {
        pendingOn: '已儲存。Hermes 重新啟動後將提供網頁應用程式。',
        pendingOff: '已關閉，但 Hermes 在重新啟動前仍會繼續提供網頁應用程式。',
        unavailableDetail: reason => `Hermes 無法提供網頁應用程式：${reason}`,
        restartBusy: '已儲存。有對話正在進行，因此 Hermes 沒有重新啟動。對話結束後請選擇「重新啟動 Hermes」。',
        restartFailed: '已儲存，但 Hermes 無法重新啟動。請選擇「重新啟動 Hermes」重試。',
        restartFailedBecause: reason => `Hermes 無法重新啟動：${reason}`,
        startTimedOut: 'Hermes 未能在 90 秒內提供網頁應用程式。',
        stopTimedOut: 'Hermes 未能在 90 秒內停止提供網頁應用程式。',
        startStages: {
          restarting: '正在重新啟動 Hermes 以提供網頁應用程式。'
        },
        restartGateway: '重新啟動 Hermes'
      },
      title: '機器連線',
      intro: '預設使用本機。若此應用程式應在其他地方執行 Hermes，請使用遠端。連線一台機器即可探索其智慧體。',
      localTitle: '本機',
      remoteTitle: '遠端機器',
      remoteDesc: '將此桌面殼層連線至遠端 Hermes 後端。',
      probing: '正在檢查此機器的驗證方式…',
      probeError: '暫時無法連線此機器。請檢查 URL；機器回應後將顯示驗證方式。',
      authSignedInPassword: '此機器使用帳號和密碼。您已登入，工作階段會自動重新整理。',
      authSignedInOauth: '此機器使用 OAuth。您已登入，工作階段會自動重新整理。',
      authNeedsPassword: '此機器使用帳號和密碼。請登入以授權此桌面應用程式。',
      authNeedsOauth: provider => `此機器使用 OAuth。請使用 ${provider} 登入以授權此桌面應用程式。`,
      plainTextConfirmTitle: '以純文字儲存存取 Token？',
      keychainEncryptionDesc:
        '預設關閉。開啟後，存取 Token 與登入憑證將使用系統鑰匙圈（Keychain Access、GNOME Keyring 或 Windows DPAPI）加密——系統可能會要求授權或密碼。關閉時，它們以僅目前使用者可讀的一般檔案形式儲存。',
      diagnosticsDesc: '在檔案管理員中顯示 desktop.log，Hermes 啟動失敗時很有用。',
      incompleteTitle: '遠端機器設定不完整',
      restartingTitle: '正在重新連線機器',
      savedTitle: '機器設定已儲存',
      reachableTitle: '遠端機器可連線',
      signedOutMessage: '已清除遠端機器工作階段。',
      failedLoad: '機器設定載入失敗',
      testFailed: '遠端機器測試失敗',
      applyFailed: '無法套用機器設定',
      saveFailed: '無法儲存機器設定',
      webappHostTitle: 'Hermes 機器',
      envOverrideTitle: '此連線由 Hermes 的啟動方式固定。',
      plainTextConfirmDesc:
        '在此機器上找不到作業系統的金鑰環服務，因此 Token 將以未加密的純文字儲存在應用程式的連線設定檔中，以該使用者身分執行的任何處理程序皆可讀取。請安裝或啟用系統鑰匙圈（Linux 上為 GNOME Keyring 或 KWallet）以進行加密儲存。',
      plainTextStoredDesc:
        '安全儲存無法使用，因此已儲存的 Token 以未加密方式儲存在此機器上應用程式的連線設定檔中。請安裝或啟用系統鑰匙圈（Linux 上為 GNOME Keyring 或 KWallet）以將其加密。'
    },
    profileScope: {
      editsProfile: profile => `此頁面的變更將套用至智慧體「${profile}」。`
    },
    model: {
      moaTitle: '混合智慧體（Mixture of Agents）',
      moaDescription:
        '設定以「混合智慧體」提供者下模型形式出現的命名預設。聚合模型是執行模型——它執行工具迴圈的每一步，整個運行幾乎全部費用都計入其提供者。參考模型預設每輪使用者訊息僅提供一次建議。',
      tasks: {
        review: {
          hint: '/review 評審子智慧體'
        },
        profile_describer: {
          label: '智慧體描述',
          hint: '自動生成智慧體描述'
        },
        title_generation: {
          hint: '對話標題'
        },
        voice_chat: {
          label: '語音對話'
        }
      },
      subagents: {
        inherit: '繼承主智慧體設定',
        modelOnly: '自訂模型（沿用主智慧體憑證）',
        providerLabel: '子智慧體提供方',
        modelLabel: '子智慧體模型',
        customProviderLabel: '自訂子智慧體提供方',
        parentModel: '使用主智慧體模型',
        parentModelWarning:
          '所選提供方必須支援主智慧體模型。模型不相容時，僅指定提供方的路由可能失敗。若該提供方設定了自己的預設模型，子智慧體會改用該模型。',
        routing: {
          settingsTitle: '子智慧體預設值',
          intro: '決定每個子智慧體使用哪個模型。',
          modes: {
            default: {
              description: '所有子智慧體都使用下方的子智慧體模型。'
            },
            model_map: {
              description: '為幾個模型命名，智慧體會為每個子智慧體挑選一個。'
            },
            profile_routes: {
              label: '智慧體路由',
              description: '使用你現有的智慧體來執行子任務，沿用它自己的模型、技能和金鑰。'
            }
          },
          pluginMissing: '模型路由和智慧體路由來自 Advanced Delegation 外掛。',
          pluginDisabled: 'Advanced Delegation 已安裝，但此智慧體尚未啟用它。',
          modeFailed: '無法變更子智慧體路由',
          loadFailed: '無法載入子智慧體路由',
          blockedDetail: '此模式已儲存，因此 Hermes 會拒絕每個子智慧體，而不是悄悄改用預設。',
          blockedMissing: '安裝 Advanced Delegation 之前，子智慧體不會啟動。',
          blockedDisabled: '為此智慧體開啟 Advanced Delegation 之前，子智慧體不會啟動。',
          blockedNotLoaded: 'Hermes 載入 Advanced Delegation 之前，子智慧體不會啟動。重新啟動 Hermes 以完成。',
          blockedUnavailable: label => `${label} 能在此處執行之前，子智慧體不會啟動。其外掛回報無法執行。`,
          blockedUnregistered: name => `安裝並開啟提供「${name}」的外掛之前，子智慧體不會啟動。`,
          modelRoutesEmpty: '尚無路由。新增路由之前，子智慧體會使用主智慧體的模型。',
          deleteConfirmBody: '子智慧體將無法再選擇它。',
          dialogDesc: '智慧體會讀取名稱和描述，決定子智慧體走哪條路由。',
          nameHint: '簡短，不含空格。智慧體依此名稱挑選路由。',
          descriptionHint: '智慧體在選擇前讀取的一行說明。',
          effortInherit: '子智慧體預設值',
          profileRoutesTitle: '可用來執行子任務的智慧體',
          profileRoutesHint: '每個勾選的智慧體都會使用自己的模型、技能、指令和金鑰。',
          profileEmpty: '尚無其他智慧體。可從智慧體列建立。',
          profileMissing: name => `${name}（智慧體已不存在）`,
          inactiveModelMap: route => `模型路由會為每個子智慧體挑選模型。未選擇路由的子智慧體使用「${route}」。`,
          inactiveProfiles: profile => `智慧體路由會挑選現有的智慧體來執行子任務。未指定時使用「${profile}」。`,
          inactiveModelMapNoDefault: '模型路由會為每個子智慧體挑選模型。未選擇路由的子智慧體使用主智慧體的模型。',
          inactiveProfilesNoDefault: '智慧體路由會挑選現有的智慧體來執行子任務。未指定時使用主智慧體的模型。',
          inactivePlugin: label => `${label} 決定每個子智慧體的模型，因此此設定不適用。`
        }
      },
      appliesDesc: '套用至新對話。可在輸入框的模型選擇器中臨時切換目前對話。'
    },
    billing: {
      errors: {
        remoteSpendingReconnect: who => `${who} 請從「設定 → 機器」重新連線以重新授權此裝置。`,
        sessionRevoked: {
          message: '你的工作階段已登出。請從「設定 → 機器」重新登入。'
        },
        transport: {
          message: '帳單請求在到達 Hermes 前失敗。'
        }
      }
    },
    toolsets: {
      browserRealProfile: {
        description:
          '將預設瀏覽器的登入資訊與 Cookie 複製到受管理的快照中，智慧體使用該快照進行瀏覽，應用程式內瀏覽器也會用它們登入。絕不會直接開啟你的真實設定檔。將套用於新對話。',
        prompt: {
          body: '讓智慧體使用預設瀏覽器設定檔的快照進行瀏覽，這樣網站開啟時即已登入，智慧體和應用程式內瀏覽器分頁皆然。'
        },
        enabledMessage: '新對話將使用預設瀏覽器設定檔的快照進行瀏覽。',
        disabledMessage: '設定檔快照將被刪除，應用程式內瀏覽器也會登出這些登入；新對話使用乾淨的瀏覽器。'
      },
      terminalBackend: {
        selectedMessage: backend => `終端命令現在透過 ${backend} 執行。將套用於新對話。`,
        needsSetupConfirmDescription: detail => `${detail} 此變更生效後啟動的對話在設定完成前將沒有終端或檔案工具。`,
        needsSetupConfirmDescriptionGeneric:
          '此後端尚未完成設定。此變更生效後啟動的對話在設定完成前將沒有終端或檔案工具。'
      }
    },
    fieldLabels: {
      'agent.maxTurns': '智慧體步數上限',
      'delegation.model': '子智慧體模型',
      'delegation.provider': '子智慧體提供方',
      'delegation.maxIterations': '子智慧體輪次上限',
      'delegation.maxConcurrentChildren': '平行子智慧體',
      'delegation.childTimeoutSeconds': '子智慧體逾時',
      'delegation.reasoningEffort': '子智慧體推理強度',
      'voice.voiceChatMode': '語音對話模式'
    },
    fieldDescriptions: {
      'browser.useRealProfile':
        '本機瀏覽會使用你的真實登入狀態。Hermes 會將預設瀏覽器的設定（Cookie、登入資訊與偏好）複製成受管理的快照，再以內建的 Chromium 驅動它——不會直接開啟你正在使用的設定檔，且每次執行都會從目前的設定檔重新整理副本。設定雲端瀏覽器後端時，也允許智慧體視需要開啟本機真實設定檔工作階段。僅支援 Chromium 系瀏覽器（Chrome、Edge、Brave、Brave Origin、Chromium）；若預設瀏覽器並非 Chromium 系，會顯示明確錯誤。預設關閉。',
      model: '除非你在輸入框選擇其他模型，否則新對話會使用此模型。',
      'display.personality': '新對話的預設助手風格。',
      'memory.memoryEnabled': '儲存有助於未來對話的持久記憶。',
      'stt.echoTranscripts': '將語音訊息的原始 🎙️ 轉寫文字傳回對話。'
    },
    sections: {
      chat: '對話'
    },
    memoryLayers: {
      selectedDesc: current => `尚未生效：在儲存此選擇之前，對話仍使用 ${current}。`,
      noProviderDesc: '智慧體只使用下方的內建層。如需新增，請在 設定 › 記憶與脈絡 中選擇外部提供者。',
      providerDesc: name => `${name} 會在內建層之外，為每個對話加入自己的回憶。`,
      builtinDesc:
        'MEMORY.md 和 USER.md 是智慧體自己的記憶，獨立於任何外部提供者。兩者預設都會載入對話；可在 設定 › 記憶與脈絡 中以記憶和使用者檔案開關分別關閉。提供者是對這一層的補充，絕不會取代它。'
    },
    sessions: {
      loading: '正在載入已封存對話…',
      archivedTitle: '已封存對話',
      archivedIntro: '已封存的對話會從側邊欄隱藏，但保留全部訊息。在側邊欄 Alt/⌥+Shift 點擊對話即可封存。',
      emptyArchivedDesc: '封存一個對話後會顯示在這裡。',
      defaultDirDesc: '新對話預設從此資料夾開始，除非您選擇其他目錄。留空則使用您的家目錄。',
      failedLoad: '無法載入已封存對話'
    }
  },
  skills: {
    lockHint: '鎖定期間，智慧體無法變更此技能。你仍可自行編輯。',
    unlockedHint: '鎖定後，智慧體將無法變更此技能。你仍可自行編輯。',
    plugins: {
      pageBlurb: '外掛可以擴充本應用程式、智慧體，或兩者 — 每一部分都有自己的開關。',
      autoUpdateTip: (hours: number) =>
        `Hermes 檢查時（每 ${hours} 小時）自動套用外掛更新。更新後的程式碼會在新對話中載入。`
    },
    remotePluginHint: '在此智慧體所屬的機器上開啟它以管理外掛。外掛安裝使用已連線的機器。',
    openAgentSettings: '智慧體設定',
    appliesToNewSessions: name => `${name} 將套用至新對話。`,
    bulkUpdated: count => `已為新對話更新 ${count} 項。`,
    changesApplyNewSessions: '變更將套用至新對話。',
    openToolLibrary: '瀏覽工具目錄'
  },
  gatewayGroupOpen: {
    lookupFailed: '無法在機器上查找此群組，請重試。'
  },
  agents: {
    close: '關閉智慧體',
    subtitle: '目前回合的子智慧體即時活動。',
    emptyTitle: '暫無活躍子智慧體',
    emptyDesc: '當某個回合派發任務時，子智慧體會在此即時顯示進度。',
    moreAgents: count => `還有 ${count} 個子智慧體`,
    steerPlaceholder: '此子智慧體的指令',
    requestRejected: '子智慧體未接受請求',
    agentsCount: count => `${count} 個智慧體`
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: '機器狀態、記錄、重新啟動/更新'
      },
      sessions: {
        title: '對話面板',
        detail: '搜尋、釘選和管理對話'
      }
    },
    gatewayRunning: '訊息服務執行中',
    gatewayStopped: '訊息服務已停止',
    restartGateway: '重新啟動 Hermes',
    gatewayRestartFailed: 'Hermes 重新啟動失敗。',
    sharedGatewayRestartTitle: '為所有智慧體重新啟動 Hermes？',
    sharedGatewayRestartDescription: bots => `此裝置上的所有智慧體都會重新連線：${bots}`,
    sharedGatewayRestarted: count => `共用 Hermes 已重新啟動（${count} 個智慧體）`,
    searchPlaceholder: '搜尋對話、檢視和動作',
    goToSession: '前往對話',
    archivedChats: '已封存對話',
    sections: {
      sessions: '對話'
    },
    nav: {
      newChat: {
        title: '新對話',
        detail: '開始新的對話'
      },
      artifacts: {
        title: '資源庫'
      }
    },
    providerSessions: '對話',
    pinSession: '釘選對話',
    unpinSession: '取消釘選對話',
    exportSession: '匯出對話',
    deleteSession: '刪除對話',
    noSessions: '尚無對話。',
    hermesActiveSessions: (version, count) => `Hermes ${version} · 活躍對話 ${count}`,
    statSessions: '對話'
  },
  messaging: {
    states: {
      gateway_stopped: '訊息服務已停止'
    },
    hintPendingRestart: '在狀態列重新啟動 Hermes 以套用此變更。',
    sharedListenerUrl: '透過共享 Hermes 監聽器提供，位址為',
    hintGatewayStopped: '在狀態列啟動 Hermes 以建立連線。',
    gatewayStopped: '訊息服務已停止',
    restartToApply: '此變更將在 Hermes 重新啟動後生效。',
    restartToReconnect: '新憑證將在 Hermes 重新啟動後生效。',
    appliedLive: '已套用到執行中的 Hermes。',
    connectingLive: '執行中的 Hermes 正在使用新憑證連線。',
    restartNeeded: '已儲存。請重新啟動 Hermes 以套用新設定。',
    restartFailedManual: 'Hermes 重新啟動失敗 — 請手動重新啟動並檢查 Hermes 日誌。',
    relayLink: {
      intro: (platform: string) =>
        `${platform} 透過共用的 Hermes 機器人連到這個智慧體。連結你的 ${platform} 帳號，再傳訊息給機器人就能和智慧體對話。`,
      linked: (platform: string) => `${platform} 帳號已連結。傳訊息給機器人就能和智慧體對話。`,
      managedNote: (platform: string) =>
        `此執行個體的訊息由 Team Gateway 承載，因此 Hermes 每次啟動都會關閉 ${platform} 的直接連線。若要與中繼同時使用，請在 Hermes 環境中設定 GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true（直接連線會使執行個體無法縮減到零，並可能重複傳遞訊息）。`,
      managedIntro: '此智慧體的帳號在 Nous Portal 中連結。'
    },
    telegramQr: {
      savedRestarting: 'Telegram 已儲存；Hermes 正在重新啟動…',
      savedRestartFailed: detail => `Telegram 已儲存；Hermes 重新啟動失敗${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: '機器人主動傳送訊息的頻道（例行任務輸出、提醒等）。'
      },
      QQBOT_HOME_CHANNEL: {
        help: '例行任務傳遞的預設頻道或群組。'
      }
    }
  },
  profiles: {
    close: '關閉智慧體',
    title: '智慧體',
    count: count => `${count} 個智慧體`,
    search: '搜尋智慧體…',
    loading: '正在載入智慧體…',
    newProfile: '新增智慧體',
    importProfile: '匯入智慧體…',
    exportProfile: '匯出智慧體…',
    imported: '智慧體已匯入',
    exported: '智慧體已匯出',
    failedImport: '匯入智慧體失敗',
    failedExport: '匯出智慧體失敗',
    exportNeedsHost: name => `瀏覽器無法儲存智慧體封存檔。請在執行 Hermes 的機器上執行：hermes profile export ${name}`,
    allProfiles: '全部智慧體',
    showAllProfiles: '顯示全部智慧體',
    manageProfiles: '管理智慧體…',
    remoteOverride: {
      description: '此智慧體的對話將在你指定的遠端 Hermes 上執行，而不是這台電腦。',
      collisionWarning: (label: string) => `設定中已存在名為「${label}」的機器。此智慧體的連線是獨立的，不會變更它。`,
      confirmTitle: '將此智慧體連線至遠端機器？',
      savedTitle: '智慧體已連線',
      menuItem: '連線至遠端機器…',
      title: (profile: string) => `將 ${profile} 連線至遠端機器`,
      confirmNote: (profile: string, host: string) =>
        `${profile} 中的新對話將在 ${host} 上執行。指令執行與檔案讀取都會發生在那台電腦上，而不是這台。請只連線你信任的機器。`,
      authFailedTitle: '遠端機器拒絕了已儲存的權杖'
    },
    defaultProfile: '預設智慧體',
    defaultDescription: '用於 Hermes 啟動和新建對話。現有對話仍保留在各自的智慧體中。',
    failedSetDefault: '無法設定預設智慧體',
    noProfiles: '尚無智慧體。',
    selectPrompt: '選擇一個智慧體以檢視其詳細資訊。',
    refresh: '重新整理智慧體',
    refreshing: '正在重新整理智慧體',
    soulDesc: '此智慧體的指令與角色設定。',
    soulMissing: '此智慧體尚無 SOUL.md 檔案。在下方輸入指令並儲存即可建立。config.yaml 中的人格預設需另外管理。',
    soulPlaceholder: mode => `此智慧體的指令與角色設定。\n留空則保留${mode}預設值。`,
    deleteTitle: '刪除智慧體？',
    createDesc: '每個智慧體都有自己的指令、設定、技能和對話。',
    cloneFromDesc: '從選取的智慧體複製設定、技能和 SOUL.md。',
    cloneFromDefault: '從預設智慧體複製設定',
    cloneFromDefaultDesc: '從您的預設智慧體複製設定、技能和 SOUL.md。',
    invalidName: hint => `智慧體名稱無效。${hint}`,
    createAction: '建立智慧體',
    renameTitle: '重新命名智慧體',
    renameDescPrefix: '重新命名會更新智慧體的目錄以及 ',
    failedLoad: '載入智慧體失敗',
    failedDelete: '刪除智慧體失敗',
    failedCreate: '建立智慧體失敗',
    failedRename: '重新命名智慧體失敗',
    fleet: {
      localDevice: '此裝置（本機後端——若未安裝 Hermes 則會安裝，否則開啟新的對話）',
      installDeviceDesc: '這會在本機安裝 Hermes，然後在這台電腦上開啟新的對話。確認之前不會開始安裝。'
    },
    created: '智慧體已建立',
    renamed: '智慧體已重新命名',
    deleted: '智慧體已刪除'
  },
  cron: {
    close: '關閉例行任務',
    title: '例行任務',
    count: count => `${count} 個例行任務`,
    search: '搜尋例行任務…',
    loading: '正在載入例行任務…',
    newCron: '新增例行任務',
    emptyDescNew: '在你選擇的時間自動執行提示詞，並將結果傳送至你偏好的目的地。',
    emptyTitleNew: '暫無例行任務',
    actionsTitle: '例行任務動作',
    edit: '編輯例行任務',
    deleteTitle: '刪除例行任務？',
    resumed: '例行任務已恢復',
    paused: '例行任務已暫停',
    triggered: '例行任務已觸發',
    deleted: '例行任務已刪除',
    created: '例行任務已建立',
    updated: '例行任務已更新',
    failedLoad: '載入例行任務失敗',
    failedUpdate: '更新例行任務失敗',
    failedTrigger: '觸發例行任務失敗',
    failedDelete: '刪除例行任務失敗',
    failedSave: '儲存例行任務失敗',
    editTitle: '編輯例行任務',
    createTitle: '新增例行任務',
    createDesc: '設定要自動執行的提示詞與時間。可使用 cron 語法或「每 15 分鐘」等自然語言。',
    promptPlaceholder: '智慧體每次執行時應做什麼？',
    scriptOnlyEditHint: '此例行任務僅執行指令碼（無 AI 提示詞）。例行任務 ID：',
    createAction: '建立例行任務',
    tabs: {
      jobs: '例行任務'
    },
    blueprints: {
      dialogDesc: '填寫詳細資訊並設定執行時間。',
      scheduled: '已從藍圖建立例行任務'
    },
    resume: '繼續例行任務',
    pause: '暫停例行任務'
  },
  sidebar: {
    filter: {
      profile: '智慧體'
    },
    gatewayGroups: {
      grouping: '機器與智慧體',
      aliasHint: '僅變更顯示名稱；機器和智慧體名稱維持不變。'
    },
    profileRail: '智慧體列',
    nav: {
      cron: '例行任務',
      'new-session': '新對話',
      artifacts: '資源庫'
    },
    cronJobs: '例行任務',
    projects: {
      createdInPreviousContext: '專案已在先前的連線或智慧體中建立。請切換回去；IDEA.md 尚未寫入。',
      showAllSessions: '顯示所有對話',
      showAllCount: count => `顯示全部 ${count} 個對話`
    },
    row: {
      ownedByProfile: profile => `智慧體：${profile}`,
      copyIdFailed: '無法複製對話 ID',
      sessionActions: '對話動作',
      sessionRunning: '對話執行中',
      renameTitle: '重新命名對話',
      untitledPlaceholder: '未命名對話',
      deleteTitle: '刪除對話？',
      deleted: '對話已刪除',
      untitledChat: id => `對話 ${id}`
    },
    searchAria: '搜尋對話',
    searchPlaceholder: '搜尋對話…',
    noMatch: query => `沒有對話符合「${query}」。`,
    sessions: '對話',
    groupAriaGrouped: '以單一清單顯示對話',
    groupAriaUngrouped: '依工作區分組對話',
    showSessions: '顯示對話',
    groupTitleGrouped: '取消對話分組',
    allPinned: '這裡的全部已釘選。取消釘選某個對話即可在最近中顯示。',
    shiftClickHint: 'Shift + 點擊對話以釘選',
    projectEmpty: '尚無對話',
    projectLoadFailed: '無法載入對話',
    noSessions: '尚無對話',
    noFilterMatches: '沒有對話符合這些篩選條件',
    newSessionIn: label => `在 ${label} 中新建對話`
  },
  composer: {
    commandDescs: {
      '/retry': '重試最後一則訊息（重新傳送給智慧體）',
      '/export': '將智慧體（設定、技能、主題）匯出為可分享的封存檔',
      '/import': '將分享的智慧體封存檔匯入為新的智慧體',
      '/agents': '顯示作用中的智慧體與執行中的任務',
      '/review': '產生獨立子智慧體，審查剛才討論的工作（PR、程式碼、文件）',
      '/profile': '切換作用中的智慧體',
      '/browser': '管理智慧體瀏覽器 [connect|disconnect|status]',
      '/platform': '暫停、恢復或列出故障的訊息平台',
      '/clear': '開始新對話',
      '/resume': '繼續已儲存的對話',
      '/new': '開始新的桌面對話',
      '/topic': '啟用或查看 Telegram 私訊話題對話',
      '/title': '重新命名目前對話',
      '/handoff': '將此對話交接至訊息平台',
      '/branch': '將最新訊息分支至新的對話',
      '/bg': '在獨立的背景對話中執行提示詞',
      '/goal': '設定持續目標，讓智慧體跨回合工作直到完成',
      '/heartbeat': '設定週期性提示詞，在閒置時重新進入此對話',
      '/loop': '在此對話以固定間隔重新執行提示詞',
      '/status': '顯示目前對話狀態',
      '/learn': '從你描述的任何內容（目錄、URL、此對話、筆記）學習可重用的技能',
      '/model': '切換此對話的模型'
    },
    queueLostDiscardTip: 'Hermes 在該回合進行中重新啟動，無法完成。捨棄後其後的排隊回合將繼續執行。',
    newSessionPlaceholders: [
      '我們要建立什麼？',
      '給智慧體一個任務',
      '您在想什麼？',
      '描述您需要什麼',
      '我們該處理什麼？',
      '盡管問',
      '從一個目標開始'
    ],
    attachUrlDesc: '智慧體將擷取該頁面並作為此回合的脈絡。',
    queueDroppedBody: '該背景佇列項目因其對話多次嘗試後仍無法恢復而遭捨棄。佇列中的其他內容不受影響。',
    dropSession: '拖曳以連結此對話',
    browseAllSessions: '瀏覽所有對話…'
  },
  statusStack: {
    agents: '智慧體',
    subagents: count => `${count} 個子智慧體`,
    control: {
      waitSession: target => `等待對話 ${target}`,
      loopDeferredNotice: '目前由進行中的目標控制對話。',
      continuationBusy: '目標已恢復 — 對話忙碌中，請先停止目前回覆（Stop 按鈕或 Esc）以繼續',
      controlUnavailable: msg => `對話控制無法使用: ${msg}`
    },
    coding: {
      agentShip: '讓智慧體提交並開 PR'
    }
  },
  updates: {
    bundleOutOfSyncDesc:
      'Hermes 執行環境已更新,但桌面應用程式本身仍是舊建置——在應用程式更新之前,新的介面功能(如智慧體模式)不會顯示。請執行下方的更新以重新建置應用程式。如果此警告仍未消除,請從最新的桌面安裝程式重新安裝。',
    bundleSwapPendingDesc: '更新後的應用程式已安裝完成，只需重新啟動 Hermes 即可載入新版本。對話和設定不會受到影響。',
    applyStatus: {
      noReturn: '後端未恢復連線。更新可能未完成——請檢查執行 Hermes 的機器。'
    }
  },
  install: {
    setupChoiceDesc: '將此應用程式連線至已執行 Hermes 的機器，或在這台電腦上本機安裝 Hermes。',
    remoteSetupDesc: '輸入機器 URL。Hermes Desktop 會偵測需要權杖還是瀏覽器登入。',
    remoteUrlTitle: '機器 URL',
    remoteUrlDesc: '使用 Hermes 機器的基礎 URL；遠端位址請包含 https://。',
    probing: '正在偵測機器驗證方式...',
    probeError: '無法連線到該 Hermes 機器。',
    authNeedsOauth: provider => `測試此機器前請先使用 ${provider} 登入。`,
    enterUrlFirst: '請先輸入機器 URL。',
    tokenDesc: '貼上遠端機器 .env 檔案中的工作階段權杖。',
    incompleteSignInTest: '測試此受 OAuth 保護的機器前請先登入。',
    incompleteTokenTest: '測試此機器前請輸入工作階段權杖。',
    connectExistingDesc: '使用工作階段權杖或瀏覽器登入連線遠端機器。不會啟動本機安裝。'
  },
  shell: {
    gatewayMenu: {
      gateway: '機器',
      reconnectGateway: '重新連線機器',
      machineSettings: '機器',
      restartTitle: '在這台機器上重新啟動 Hermes？'
    },
    statusbar: {
      gateway: '機器',
      messagingStopped: '訊息服務已停止',
      gatewayTitle: '機器',
      agents: '智慧體',
      closeAgents: '關閉智慧體',
      openAgents: '開啟智慧體',
      subagents: count => `${count} 個子智慧體`,
      cron: '例行任務',
      openCron: '開啟例行任務',
      contextUsagePanel: {
        categories: {
          subagent_definitions: '子智慧體定義'
        }
      }
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: '已移除的機器',
    terminalGatewayRemovedBody: '此終端機所在的機器已被移除，無法重新開啟。請關閉此分頁。',
    terminalGatewayNoShellBody: '此機器不提供 Shell 存取，此終端機無法重新開啟。請關閉此分頁。',
    addToChat: '新增至對話'
  },
  preview: {
    saveScopeChanged: '請切換回原本的連線和智慧體以儲存此草稿。',
    web: {
      remoteLoopback:
        '這個位址指向執行智慧體的那台機器，而不是本機。瀏覽器窗格會在本機載入頁面，因此遠端開發伺服器需要連接埠轉送或可連線的主機名稱。',
      restarting: '智慧體正在重新啟動伺服器...',
      askRestart: '請智慧體重新啟動伺服器',
      lookingRestart: taskId => `智慧體正在尋找要重新啟動的預覽伺服器 (${taskId})`,
      restartingMessage: '智慧體正在背景執行。可在預覽主控台查看進度。',
      blankPageBody: '在上方輸入網址開始瀏覽，或請智慧體開啟頁面。',
      finishedRestarting: message => `智慧體已完成預覽伺服器重新啟動${message ? `：${message}` : ''}`,
      restartFailedMessage: '智慧體無法重新啟動伺服器。',
      stillWorking: '智慧體仍在執行，但尚未收到重新啟動結果。伺服器指令可能正在前台執行。'
    },
    console: {
      sendEntry: '將此項目傳送至對話',
      sendToChat: '傳送至對話',
      sentTitle: '已傳送至對話'
    }
  },
  assistant: {
    thread: {
      errorCodes: {
        cron_session_read_only: {
          title: '此對話屬於一個例行任務',
          body: '它只會自動執行該例行任務，因此你的訊息未被執行。請開始新的對話後再傳送。'
        },
        no_reply: {
          body: '智慧體已完成本輪，但回覆沒有送達此視窗。請重試以重新傳送。'
        },
        context_overflow: {
          title: '此對話過長'
        },
        format_error: {
          body: provider => `${provider} 無法接受此對話的結構。請切換服務商；若問題持續，請開始新的對話。`
        },
        loop_error: {
          title: '智慧體無法完成此步驟'
        },
        session_read_only: {
          title: '此對話為唯讀',
          body: '它是 Hermes 只能顯示、無法繼續的已儲存歷史記錄，因此你的訊息未被執行。請開始新的對話後再傳送。'
        },
        workspace_unavailable: {
          title: '此對話的資料夾已不存在',
          body: '此對話使用的資料夾已被移動或刪除，因此你的訊息未被執行。請重新建立該資料夾，或使用「Move to project」為對話指定其他資料夾，然後重試。'
        },
        launch_credentials_unavailable: {
          title: '此對話的 API 金鑰已變更',
          body: '此對話開始時使用的 API 金鑰後來已被變更或刪除，因此你的訊息未送出。開啟中的對話不會自行切換到其他金鑰。請開始新的對話以使用目前的金鑰。'
        },
        agent_init_failed: {
          title: '無法開始此對話',
          body: provider => `Hermes 無法為此對話連線到 ${provider}。請重試；若持續失敗，請在設定中檢查服務商。`
        },
        SESSION_NOT_OWNED: {
          title: '對話已在別處開啟',
          body: '此對話已在另一個 Hermes 視窗或終端機中開啟。請在這裡開始新的對話，或先在那裡關閉它。'
        },
        free_tier_disabled: {
          title: '登入以繼續對話',
          body: '使用 Nous 帳戶登入即可繼續對話，完全免費。'
        },
        free_tier_refused: {
          title: '登入以繼續對話'
        }
      },
      loadingSession: '正在載入對話',
      loadingResponse: '智慧體正在載入回覆',
      branchNewChat: '在新對話中分支',
      errorLayers: {
        generic: '智慧體無法完成此回覆',
        runtime: '智慧體遇到問題'
      },
      errorLayerBodies: {
        generic: '智慧體回覆時發生問題。請重試；若問題持續，請複製錯誤詳細資訊。'
      },
      errorStartNewChatLink: '開始新的對話',
      errorStartNewSession: '開始新對話',
      errorOauthExpired: provider => `你的 ${provider} 登入已過期或被撤銷。請重新登入以繼續對話。`
    },
    approval: {
      gatewayDisconnected: '機器未連線',
      allowSession: '允許本對話',
      alwaysDescription: pattern =>
        `這會將「${pattern}」模式加入永久允許清單（~/.hermes/config.yaml）。智慧體對類似指令將不再詢問，包括目前對話和未來對話。`
    },
    clarify: {
      gatewayDisconnected: '機器未連線',
      notDelivered: '此問題未送達應用程式，無法在此回答。請按停止結束本輪，然後在對話中回覆。'
    },
    catalogInstall: {
      targetProfile: profile => `安裝到你的智慧體 ${profile}`
    },
    tool: {
      titles: {
        cronjob: {
          done: '已設定例行任務',
          pending: '正在設定例行任務',
          pendingAction: '正在設定'
        },
        session_search_recall: {
          done: '已搜尋對話歷史',
          pending: '正在搜尋對話歷史'
        }
      }
    },
    themePicker: {
      basicDetail: '用來與你的智慧體對話。'
    }
  },
  prompts: {
    gatewayDisconnected: '機器未連線',
    sudoDesc: '輸入 sudo 密碼前，請先確認指令。密碼會傳送給執行指令的智慧體，並在本次對話中快取。',
    sudoCommandUnavailable: '此智慧體未提供指令。如果無法在對話中確認，請取消。',
    sudoInstallDesc:
      '智慧體需要您的 sudo 密碼，以在其所在的機器上安裝智慧體畫面套件（TigerVNC + Xfce）。它只會傳送到該機器。',
    vaultUnlockDesc: name =>
      `智慧體想使用儲存在 ${name} 中的登入資訊登入網站。輸入主密碼以在本工作階段解鎖——它會直接交給本機的 ${name}，不會被儲存或顯示給智慧體。`,
    secretDesc: '智慧體需要一個憑證才能繼續。',
    vaultSaveDesc: origin =>
      `智慧體到達了 ${origin} 的登入頁，但沒有為它儲存的登入資訊。在此輸入一次；它會在本機加密儲存並直接填入頁面，模型永遠看不到密碼。`,
    vaultCodeDesc: site =>
      `${site} 要求輸入一次性驗證碼（簡訊、電子郵件或驗證器應用程式）。在此輸入，智慧體會將其填入頁面；模型永遠看不到它。`,
    vaultCodeFootnote: '提示：在「設定 → 密碼與登入」中為此登入儲存驗證器金鑰後，智慧體會自動填寫驗證碼。'
  },
  desktop: {
    profileStatus: current => `智慧體：${current}。使用 /profile <name> 或「新對話」選擇器與其他智慧體開始對話。`,
    unknownProfile: '未知智慧體',
    noProfileNamed: (target, available) => `沒有名為「${target}」的智慧體。可用的：${available}`,
    newChatsProfile: name => `新對話將使用智慧體 ${name}。`,
    setProfileFailed: '無法切換智慧體',
    resumeStrandedBody: '與此對話的連線失敗，自動重試已停止。請確認 Hermes 正在該機器上執行，然後重試。',
    poolSlotTimeoutBody:
      '同時執行的智慧體過多，已達此電腦的上限。請在「設定」→「進階」中提高上限，或等待其中一個完成後重試。',
    handoff: {
      timedOut: '等待 Hermes 逾時。`hermes gateway` 是否正在執行？'
    },
    sessionUnavailable: '對話不可用',
    createSessionFailed: '無法建立新對話',
    yoloArmed: '此對話已啟用 YOLO',
    yoloSystem: active => `此對話 YOLO ${active ? '已開啟' : '已關閉'}`,
    resumeStrandedTitle: '無法載入此對話',
    branchNeedsChat: '分支前請先開始或繼續一個對話。',
    sessionBusy: '對話忙碌中',
    branchStopCurrent: '分支此對話前請先停止目前回合。',
    cwdStagedMessage: '重新啟動桌面後端後，工作目錄變更才會套用至此作用中對話。',
    sessionExported: '對話已匯出',
    sessionExportFailed: '無法匯出對話',
    sessionOpenFailed: '無法開啟此對話'
  },
  tips: {
    items: {
      messaging: {
        text: '接上 Telegram、Discord、Slack 等 — 同一個智慧體，同一份記憶。',
        title: '你的智慧體，不在電腦前也能用'
      },
      cron: {
        title: '例行任務自動執行',
        text: '例行任務會按小時、每晚，或依 cron 運算式自動執行提示詞。'
      },
      profiles: {
        title: '智慧體彼此獨立',
        text: '每個智慧體都有自己的指令、設定、記憶和對話。'
      },
      skills: {
        text: '技能是一組說明檔，智慧體會在需要時自行載入。'
      },
      artifacts: {
        title: '智慧體做過的一切',
        text: '所有對話的圖片、檔案與連結，都彙整在你的資源庫中。'
      },
      'command-palette': {
        text: '對話、設定、技能與指令都能從命令選單找到。'
      }
    }
  },
  keybinds: {
    actions: {
      'agents.new': '新增智慧體（智慧體頁面）'
    }
  },
  firstRun: {
    guide: {
      upgradeRequired:
        '此機器上的 Hermes 需要更新才能進行無對話設定。請在該機器上執行 hermes update，重新啟動 Hermes，然後重試。你的對話保持不變。',
      intro: '完成幾項可選的偏好設定後，即可開始正常對話。設定過程不會建立對話。'
    },
    intro: {
      greeting: '你好，我是你的智慧體。'
    }
  },
  catalog: {
    installDescription: '此技能將於新的對話中可用。請僅安裝可信來源的內容。',
    installHint: '安裝前請檢查原始碼。變更將於新的對話生效。'
  },
  sessionImport: {
    action: '匯入對話',
    search: '搜尋已載入的對話',
    scanError: '無法尋找對話',
    emptyHelp: '此機器上的 Claude Code 和 Codex 對話將顯示在這裡。',
    searchHelp: '嘗試其他標題或資料夾，或載入更多對話。',
    more: '載入更多對話',
    chooseHelp: '選擇對話，在匯入 Hermes 前查看歷程記錄。'
  },
  titlebar: {
    searchTitle: '搜尋對話、檢視和動作',
    unreadSessions: count => (count === 1 ? '1 個未讀對話' : `${count} 個未讀對話`)
  },
  skillDeepLink: {
    installDescription: '此技能將於新的對話中可用。請僅安裝可信來源的內容。'
  },
  starmap: {
    emptyDesc: '當智慧體為你的工作建立技能與記憶時，會顯示在這裡。'
  },
  artifacts: {
    search: '搜尋資源庫…',
    refresh: '重新整理資源庫',
    refreshing: '正在重新整理資源庫',
    indexing: '正在索引最近對話的輸出',
    noArtifactsTitle: '資源庫中尚無內容',
    noArtifactsDesc: '對話產生圖片和檔案輸出時，它們會顯示在這裡。',
    failedLoad: '資源庫載入失敗',
    colSession: '對話',
    chat: '對話',
    partialIndex: (failed, total) => `索引資源庫時略過了 ${total} 個最近對話中的 ${failed} 個。`
  },
  artifactPreview: {
    missingTitle: '項目無法使用',
    missingBody: '此項目已不在資源庫中。'
  },
  onboarding: {
    headerTitle: '開始設定你的智慧體',
    headerDesc: '連線模型提供方即可開始對話。大多數選項只需一次點擊。',
    flowSubtitles: {
      external: '先在終端機登入一次，然後回來繼續對話'
    }
  },
  interfaceMode: {
    hint: '只改變顯示的內容，不改變智慧體的能力。',
    simple: {
      description: '用於與智慧體對話。只有側邊欄和對話；沒有終端機、檔案或差異面板。'
    }
  },
  zones: {
    newSessionTab: '新增對話分頁'
  },
  errors: {
    boundaryDesc: '此檢視遇到意外錯誤。您的對話和設定是安全的。'
  }
}
