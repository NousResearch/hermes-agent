import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale } from './define-locale'

export const ja = defineLocale({
  common: {
    apply: '適用',
    back: '戻る',
    save: '保存',
    saving: '保存中…',
    cancel: 'キャンセル',
    change: '変更',
    choose: '選択',
    clear: 'クリア',
    close: '閉じる',
    collapse: '折りたたむ',
    confirm: '確認',
    connect: '接続',
    connecting: '接続中',
    continue: '続ける',
    copied: 'コピーしました',
    copy: 'コピー',
    copyFailed: 'コピーに失敗しました',
    defaultName: 'デフォルト',
    delete: '削除',
    docs: 'ドキュメント',
    done: '完了',
    error: 'エラー',
    expand: '展開',
    failed: '失敗',
    formatJson: 'JSON を整形',
    free: '無料',
    loading: '読み込み中…',
    notSet: '未設定',
    refresh: '更新',
    remove: '削除',
    replace: '置き換え',
    retry: '再試行',
    run: '実行',
    send: '送信',
    set: '設定',
    skip: 'スキップ',
    update: '更新',
    tryHint: term => `「${term}」を試す`,
    on: 'オン',
    off: 'オフ'
  },

  fileMenu: {
    revealFinder: 'Finder で表示',
    revealExplorer: 'エクスプローラーで表示',
    revealFileManager: '格納フォルダーを開く',
    revealInSidebar: 'ファイルツリーで表示',
    copyPath: 'パスをコピー',
    copyRelativePath: '相対パスをコピー',
    rename: '名前を変更…',
    delete: '削除',
    renameTitle: '名前を変更',
    renameLabel: '新しい名前',
    deleteTitle: name => `${name} を削除しますか？`,
    deleteBody: 'ゴミ箱に移動します。そこから復元できます。',
    pathCopied: 'パスをコピーしました'
  },

  boot: {
    ready: 'Hermes Desktop の準備ができました',
    desktopBootFailedWithMessage: message => `デスクトップの起動に失敗しました: ${message}`,
    steps: {
      connectingGateway: 'ライブデスクトップゲートウェイに接続中',
      loadingSettings: 'Hermes の設定を読み込み中',
      loadingSessions: '最近のセッションを読み込み中',
      startingDesktopConnection: 'デスクトップ接続を開始中',
      startingHermesDesktop: 'Hermes Desktop を起動中…'
    },
    errors: {
      backgroundExited: 'Hermes バックグラウンドプロセスが終了しました。',
      backgroundExitedDuringStartup: '起動中に Hermes バックグラウンドプロセスが終了しました。',
      backendStopped: 'バックエンドが停止しました',
      desktopBootFailed: 'デスクトップの起動に失敗しました',
      gatewayConnectionLost: 'ゲートウェイへの接続が切断されました',
      gatewaySignInRequired: 'ゲートウェイへのサインインが必要です',
      ipcBridgeUnavailable: 'デスクトップ IPC ブリッジが利用できません。'
    },
    failure: {
      title: 'Hermes を起動できませんでした',
      description:
        'バックグラウンドゲートウェイが起動しませんでした。以下の回復手順をお試しください。チャットや設定は削除されません。',
      remoteTitle: 'リモートゲートウェイへのサインインが必要です',
      remoteDescription:
        'リモートゲートウェイのセッションが期限切れです。再接続するにはもう一度サインインしてください。チャットや設定は削除されません。',
      retry: '再試行',
      repairInstall: 'インストールを修復',
      useLocalGateway: 'ローカルゲートウェイを使用',
      gatewaySettings: 'ゲートウェイ設定',
      back: '戻る',
      openLogs: 'ログを開く',
      repairHint: '修復はインストーラーを再実行します。新しいマシンでは数分かかる場合があります。',
      remoteSignInHint: signInLabel =>
        `保存済みのリモートブラウザセッションからサインアウトし、${signInLabel}を開きます。代わりにバンドルされたバックエンドに切り替えるには「ローカルゲートウェイを使用」を選択してください。`,
      signOutAndSignIn: 'サインアウトして再サインイン',
      remoteFailureHint:
        '「ゲートウェイ設定」でゲートウェイの URL とサインインを確認するか、ローカルゲートウェイに切り替えてください。',
      hideRecentLogs: '最近のログを非表示',
      showRecentLogs: '最近のログを表示',
      signedInTitle: 'サインインしました',
      signedInMessage: 'リモートゲートウェイに再接続中…',
      signInIncompleteTitle: 'サインインが完了していません',
      signInIncompleteMessage: '認証が完了する前にログインウィンドウが閉じられました。',
      signInFailed: 'サインインに失敗しました',
      signInToRemoteGateway: 'リモートゲートウェイにサインイン',
      signInWithProvider: provider => `${provider} でサインイン`,
      identityProvider: 'ID プロバイダー'
    }
  },

  notifications: {
    region: '通知',
    hide: '非表示',
    show: '表示',
    more: count => `他 ${count} 件の通知`,
    clearAll: 'すべてクリア',
    dismiss: '通知を閉じる',
    details: '詳細',
    copyDetail: '詳細をコピー',
    copyDetailFailed: '通知の詳細をコピーできませんでした',
    backendOutOfDateTitle: 'バックエンドが古いです',
    backendOutOfDateMessage:
      'Hermes バックエンドがこのデスクトップビルドより古く、正常に動作しない場合があります。更新して揃えてください。',
    installMethodUnsupportedTitle: 'サポート対象外のインストール方法',
    updateHermes: 'Hermes を更新',
    updateReadyTitle: '更新の準備ができました',
    updateReadyMessage: count => `${count} 件の新しい変更が利用可能です。`,
    seeWhatsNew: '新機能を見る',
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT には ELEVENLABS_API_KEY が必要です。',
      elevenLabsRejectedKey: 'ElevenLabs が API キーを拒否しました (401)。',
      gatewayAuthFailed: 'ゲートウェイ認証に失敗しました — API_SERVER_KEY を確認してください。',
      methodNotAllowed:
        'デスクトップバックエンドがそのリクエストを拒否しました (405 Method Not Allowed)。Hermes Desktop を再起動してください。',
      microphonePermission: 'マイクのアクセス許可が拒否されました。',
      openaiRejectedApiKey: 'OpenAI が API キーを拒否しました。',
      openaiRejectedApiKeyWithStatus: status => `OpenAI が API キーを拒否しました (${status} invalid_api_key)。`,
      openaiTtsNeedsKey: 'OpenAI TTS には VOICE_TOOLS_OPENAI_KEY または OPENAI_API_KEY が必要です。'
    },
    voice: {
      configureSpeechToText: '音声モードを使用するには音声認識を設定してください。',
      couldNotStartSession: '音声セッションを開始できませんでした',
      microphoneAccessDenied: 'マイクへのアクセスが拒否されました。',
      microphoneConstraintsUnsupported: 'このデバイスはマイクの制約をサポートしていません。',
      microphoneFailed: 'マイクが失敗しました',
      microphoneInUse: 'マイクは他のアプリで使用中です。',
      microphonePermissionDenied: 'マイクのアクセス許可が拒否されました。',
      microphoneStartFailed: 'マイクの録音を開始できませんでした。',
      microphoneUnsupported: 'このランタイムはマイク録音をサポートしていません。',
      noMicrophone: 'マイクが見つかりませんでした。',
      noSpeechDetected: '音声が検出されませんでした',
      playbackFailed: '音声再生に失敗しました',
      recordingFailed: '音声録音に失敗しました',
      transcriptionFailed: '音声文字起こしに失敗しました',
      transcriptionUnavailable: '音声文字起こしはまだ利用できません。',
      tryRecordingAgain: 'もう一度録音してください。',
      unavailable: '音声は利用できません'
    },
    native: {
      approvalTitle: '承認が必要です',
      approveAction: '承認',
      rejectAction: '拒否',
      inputTitle: '入力が必要です',
      inputBody: 'Hermes が応答を待っています。',
      turnDoneTitle: 'Hermes が完了しました',
      turnDoneBody: '応答の準備ができました。',
      turnErrorTitle: 'ターンが失敗しました',
      backgroundDoneTitle: 'バックグラウンドタスクが完了しました',
      backgroundFailedTitle: 'バックグラウンドタスクが失敗しました',
      creditsTitle: 'クレジット'
    },
    gatewayErrorTitle: 'Hermes エラー',
    gatewayErrorFallback: 'Hermes がエラーを報告しました'
  },

  remoteDisplayBanner: {
    message: reason =>
      `ソフトウェアレンダリングが有効です — リモートディスプレイを検出しました（${reason}）。ちらつきを防ぐため GPU アクセラレーションは無効化されています。`
  },

  billingBlock: {
    titleNous: 'Nous クレジットが不足しています',
    titleProvider: provider => `クレジット不足 — ${provider}`,
    fallbackMessage: 'アカウントのクレジットが不足しています。続行するにはクレジットを追加してください。',
    openBilling: '請求を開く',
    addCredits: 'クレジットを追加',
    dismiss: '閉じる'
  },

  billingPage: {
    title: '請求',
    paymentAndCredits: '支払いとクレジット',
    usage: '使用量'
  },

  titlebar: {
    hideSidebar: 'サイドバーを非表示',
    showSidebar: 'サイドバーを表示',
    search: '検索',
    searchTitle: 'セッション、ビュー、アクションを検索',
    swapSidebarSides: 'サイドバーの向きを切り替え',
    swapSidebarSidesTitle: 'セッションとファイルブラウザーの位置を入れ替える',
    hideRightSidebar: '右サイドバーを非表示',
    showRightSidebar: '右サイドバーを表示',
    muteHaptics: '触覚フィードバックをオフ',
    unmuteHaptics: '触覚フィードバックをオン',
    openSettings: '設定を開く',
    openStarmap: 'メモリグラフを開く'
  },

  language: {
    label: '言語',
    description: 'デスクトップインターフェイスの言語を選択します。',
    saving: '言語を保存中…',
    saveError: '言語の更新に失敗しました',
    switchTo: '言語を切り替え',
    searchPlaceholder: '言語を検索…',
    noResults: '言語が見つかりません'
  },

  settings: {
    closeSettings: '設定を閉じる',
    exportConfig: '設定を書き出す',
    importConfig: '設定を読み込む',
    resetToDefaults: 'デフォルトに戻す',
    resetConfirm: 'すべての設定を Hermes のデフォルトに戻しますか？',
    exportFailed: '書き出しに失敗しました',
    resetFailed: 'リセットに失敗しました',
    nav: {
      providers: 'プロバイダー',
      providerAccounts: 'アカウント',
      providerApiKeys: 'API キー',
      providerCustomEndpoints: 'カスタムエンドポイント',
      gateway: 'ゲートウェイ',
      apiKeys: 'ツールとキー',
      keybinds: 'キーボードショートカット',
      keysTools: 'ツール',
      keysSettings: '設定',
      mcp: 'MCP',
      archivedChats: 'アーカイブ済みチャット',
      about: '情報',
      billing: '請求',
      notifications: '通知'
    },
    notifications: {
      title: '通知',
      intro: 'アプリ内トーストとは別の、ネイティブのデスクトップ通知です。設定は端末ごとに保存されます。',
      enableAll: '通知を有効にする',
      enableAllDesc: 'マスタースイッチ。オフにすると以下のすべての通知を無効にします。',
      focusedHint: '完了通知は Hermes がバックグラウンドにあるときのみ表示されます。',
      kinds: {
        approval: {
          label: '承認が必要',
          description: 'コマンドが承認または拒否を待っています。'
        },
        input: {
          label: '入力が必要',
          description: 'Hermes が質問したか、パスワードやシークレットを必要としています。'
        },
        turnDone: {
          label: '応答完了',
          description: 'Hermes がバックグラウンドのときにターンが完了しました。'
        },
        turnError: {
          label: 'ターン失敗',
          description: 'ターンがエラーで終了しました。'
        },
        backgroundDone: {
          label: 'バックグラウンドタスク完了',
          description: 'バックグラウンドのターミナルコマンドが完了しました。'
        },
        credits: {
          label: 'クレジット通知',
          description: 'クレジットの利用が停止または復旧しました。'
        }
      },
      test: 'テスト通知を送信',
      testTitle: 'Hermes',
      testBody: '通知は正常に動作しています。',
      testSent:
        'テストを送信しました。表示されない場合は、OS の通知許可と集中モード／おやすみモードを確認してください。',
      testUnsupported: 'このシステムはネイティブ通知に対応していません。',
      completionSoundTitle: '完了サウンド',
      completionSoundDesc: 'エージェントのターン終了時に再生されます。プリセットを選んでここで試聴できます。',
      completionSoundPreview: '試聴'
    },
    sections: {
      model: 'モデル',
      chat: 'チャット',
      appearance: '外観',
      workspace: 'ワークスペース',
      safety: '安全性',
      memory: 'メモリとコンテキスト',
      voice: '音声',
      advanced: '詳細'
    },
    searchPlaceholder: {
      about: 'Hermes Desktop について',
      config: '設定を検索…',
      gateway: 'ゲートウェイ接続…',
      keys: 'API キーを検索…',
      mcp: 'MCP サーバーを検索…',
      sessions: 'アーカイブ済みセッションを検索…'
    },
    modeOptions: {
      light: { label: 'ライト', description: '明るいデスクトップ表示' },
      dark: { label: 'ダーク', description: 'まぶしさを抑えたワークスペース' },
      system: { label: 'システム', description: 'OS の外観に合わせる' }
    },
    appearance: {
      title: '外観',
      intro:
        'デスクトップ専用の表示設定です。モードは明るさ、テーマはアクセントカラーとチャット面のスタイルを制御します。',
      colorMode: 'カラーモード',
      colorModeDesc: '固定モードを選ぶか、Hermes をシステム設定に合わせます。',
      toolViewTitle: 'ツール呼び出しの表示',
      toolViewDesc: 'プロダクト表示は生のツールペイロードを隠し、テクニカル表示は入出力をすべて表示します。',
      uiScaleTitle: 'UI スケール',
      uiScaleDesc: (percent: number) =>
        `アプリ全体の文字と UI を拡大縮小します。Cmd/Ctrl と +、-、0 でも変更できます。現在: ${percent}%`,
      translucencyTitle: 'ウィンドウの透過',
      translucencyDesc: 'ウィンドウ全体を透過させてデスクトップを表示します。macOS と Windows のみ。',
      backdropTitle: 'チャット背景',
      backdropDesc: '会話の背後に表示される淡い彫像の画像。',
      embedsTitle: 'インライン埋め込み',
      embedsDesc:
        'リッチプレビューは第三者サイト（YouTube、X など）から読み込まれます。確認は許可するまでプレースホルダーを表示し、常には自動で読み込み、オフはリンクのままにします。',
      embedsAsk: '確認',
      embedsAlways: '常に',
      embedsOff: 'オフ',
      embedsReset: (count: number) => `許可した${count}件のサービスをリセット`,
      product: 'プロダクト',
      productDesc: '読みやすいツール活動と簡潔な要約を表示します。',
      technical: 'テクニカル',
      technicalDesc: '生のツール引数、結果、低レベルの詳細を含めます。',
      themeTitle: 'テーマ',
      themeDesc: 'デスクトップ専用のパレットです。選択したモードの上に適用されます。',
      themeProfileNote: profile =>
        `「${profile}」プロファイルに保存されます。プロファイルごとに個別のテーマを保持します。`,
      installTitle: 'VS Code から導入',
      installDesc:
        'Marketplace の拡張機能 ID（例: dracula-theme.theme-dracula）を貼り付けると、その配色テーマをデスクトップ用パレットに変換します。',
      installPlaceholder: 'publisher.extension',
      installButton: 'インストール',
      installing: 'インストール中…',
      installError: 'そのテーマをインストールできませんでした。',
      installed: name => `「${name}」をインストールしました。`,
      removeTheme: 'テーマを削除',
      importedBadge: 'インポート済み',
      pet: {
        title: 'ペット',
        intro:
          'アプリ上に浮かぶ petdex のアニメーションマスコットを採用しましょう。ツール実行中は走り、成功すると喜び、エラーでしょんぼりと、Hermes の状態に反応します。',
        restartHint:
          'ペット機能には再起動が必要です。この機能が追加される前に起動したアプリが動作中です。Hermes を終了して再度開き、このページに戻ってください。',
        scaleTitle: 'サイズ',
        scaleDesc: '浮遊マスコットの大きさを変更します。すべての画面に即時反映されます。',
        roamTitle: '散歩',
        roamDesc: 'アイドル中にペットがウィンドウ内を自由に歩き回ります。',
        on: 'オン',
        off: 'オフ',
        chooseTitle: 'ペットを選ぶ',
        chooseDesc: '選ぶと（必要に応じて）インストールされ、アクティブになります。',
        searchPlaceholder: 'ペットを検索…',
        unreachable: 'petdex ギャラリーに接続できませんでした。接続を確認してこのページを開き直してください。',
        noMatch: query => `「${query}」に一致するペットがありません。`,
        installedTag: 'インストール済み',
        generatedTag: '生成',
        countCapped: (cap, total) => `${total} 件中 ${cap} 件を表示中——入力して絞り込めます。`,
        count: n => `${n} 件のペット。`,
        uninstall: name => `${name} をアンインストール`,
        delete: name => `${name} を削除`,
        deleteTitle: name => `${name} を削除しますか？`,
        deleteBody: 'ペットを完全に削除します。再インストールはできません。',
        deleteConfirm: '削除',
        rename: name => `${name} の名前を変更`,
        renameTitle: 'ペットの名前を変更',
        renamePlaceholder: 'ペットに名前を付ける',
        renameSave: '保存',
        exportPet: name => `${name} をエクスポート`,
        adoptFailed: slug => `${slug} を採用できませんでした`,
        uninstallFailed: slug => `${slug} をアンインストールできませんでした`,
        renameFailed: slug => `${slug} の名前を変更できませんでした`,
        exportFailed: slug => `${slug} をエクスポートできませんでした`,
        noneAvailable: 'オンにできるペットがありません。',
        turnOnFailed: 'ペットをオンにできませんでした。',
        turnOffFailed: 'ペットをオフにできませんでした。'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'デフォルトモデル',
      modelContextLength: 'コンテキストウィンドウ',
      fallbackProviders: 'フォールバックモデル',
      toolsets: '有効なツールセット',
      timezone: 'タイムゾーン',
      display: {
        personality: '人格',
        showReasoning: '推論ブロック'
      },
      desktop: {
        repoScanEnabled: 'リポジトリの自動検出',
        repoScanRoots: 'リポジトリの検索ルート',
        repoScanExcludePaths: '除外するリポジトリパス'
      },
      agent: {
        maxTurns: '最大エージェントステップ',
        imageInputMode: '画像添付',
        apiMaxRetries: 'API 再試行回数',
        serviceTier: 'サービス階層',
        toolUseEnforcement: 'ツール使用の強制'
      },
      terminal: {
        cwd: '作業ディレクトリ',
        backend: '実行バックエンド',
        timeout: 'コマンドタイムアウト',
        persistentShell: '永続シェル',
        envPassthrough: '環境変数の引き継ぎ',
        dockerImage: 'Docker イメージ',
        singularityImage: 'Singularity イメージ',
        modalImage: 'Modal イメージ',
        daytonaImage: 'Daytona イメージ'
      },
      fileReadMaxChars: 'ファイル読み取り上限',
      toolOutput: {
        maxBytes: 'ターミナル出力上限',
        maxLines: 'ファイルページ上限',
        maxLineLength: '行長上限'
      },
      codeExecution: {
        mode: 'コード実行モード'
      },
      approvals: {
        mode: '承認モード',
        timeout: '承認タイムアウト',
        mcpReloadConfirm: 'MCP 再読み込みの確認'
      },
      commandAllowlist: 'コマンド許可リスト',
      security: {
        redactSecrets: 'シークレットを伏せる',
        allowPrivateUrls: 'プライベート URL を許可'
      },
      browser: {
        allowPrivateUrls: 'ブラウザーのプライベート URL',
        autoLocalForPrivateUrls: 'プライベート URL にはローカルブラウザーを使用'
      },
      checkpoints: {
        enabled: 'ファイルチェックポイント',
        maxSnapshots: 'チェックポイント上限'
      },
      voice: {
        recordKey: '音声ショートカット',
        maxRecordingSeconds: '最大録音時間',
        autoTts: '応答を読み上げる'
      },
      stt: {
        enabled: '音声認識',
        echoTranscripts: '文字起こしのエコー表示',
        provider: '音声認識プロバイダー',
        local: {
          model: 'ローカル文字起こしモデル',
          language: '文字起こし言語'
        },
        openai: {
          model: 'OpenAI STT モデル'
        },
        groq: {
          model: 'Groq STT モデル'
        },
        mistral: {
          model: 'Mistral STT モデル'
        },
        elevenlabs: {
          modelId: 'ElevenLabs STT モデル',
          languageCode: 'ElevenLabs 言語',
          tagAudioEvents: '音声イベントをタグ付け',
          diarize: '話者分離'
        }
      },
      tts: {
        provider: '音声合成プロバイダー',
        edge: {
          voice: 'Edge 音声'
        },
        openai: {
          model: 'OpenAI TTS モデル',
          voice: 'OpenAI 音声'
        },
        elevenlabs: {
          voiceId: 'ElevenLabs 音声',
          modelId: 'ElevenLabs モデル'
        },
        xai: {
          voiceId: 'xAI (Grok) 音声',
          language: 'xAI 言語',
          speed: '再生速度',
          autoSpeechTags: '自動音声タグ',
          optimizeStreamingLatency: 'ストリーミング遅延最適化',
          sampleRate: 'サンプルレート',
          bitRate: 'ビットレート'
        },
        minimax: {
          model: 'MiniMax TTS モデル',
          voiceId: 'MiniMax 音声'
        },
        mistral: {
          model: 'Mistral TTS モデル',
          voiceId: 'Mistral 音声'
        },
        gemini: {
          model: 'Gemini TTS モデル',
          voice: 'Gemini 音声'
        },
        neutts: {
          model: 'NeuTTS モデル',
          device: 'NeuTTS デバイス'
        },
        kittentts: {
          model: 'KittenTTS モデル',
          voice: 'KittenTTS 音声'
        },
        piper: {
          voice: 'Piper 音声'
        }
      },
      memory: {
        memoryEnabled: '永続メモリ',
        userProfileEnabled: 'ユーザープロファイル',
        memoryCharLimit: 'メモリ予算',
        userCharLimit: 'プロファイル予算',
        provider: 'メモリプロバイダー'
      },
      context: {
        engine: 'コンテキストエンジン'
      },
      compression: {
        enabled: '自動圧縮',
        threshold: '圧縮しきい値',
        targetRatio: '圧縮目標',
        protectLastN: '保護する直近メッセージ'
      },
      delegation: {
        model: 'サブエージェントモデル',
        provider: 'サブエージェントプロバイダー',
        maxIterations: 'サブエージェントターン上限',
        maxConcurrentChildren: '並列サブエージェント',
        childTimeoutSeconds: 'サブエージェントタイムアウト',
        reasoningEffort: 'サブエージェント推論強度'
      },
      updates: {
        nonInteractiveLocalChanges: 'アプリ内更新時のローカル変更'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'コンポーザーで別のモデルを選ばない限り、新しいチャットで使用されます。',
      modelContextLength: '0 のままにすると、選択したモデルから検出されたコンテキストウィンドウを使用します。',
      fallbackProviders: 'デフォルトモデルが失敗したときに試す provider:model 形式のバックアップです。',
      display: {
        personality: '新しいセッションのデフォルトのアシスタントスタイルです。',
        showReasoning: 'バックエンドが推論内容を提供したときに表示します。'
      },
      desktop: {
        repoScanEnabled: 'ローカルフォルダを検索して Git リポジトリをプロジェクトに表示します。',
        repoScanRoots: '検索するフォルダです。空の場合はホームディレクトリを検索します。',
        repoScanExcludePaths: 'リポジトリ検出時に除外するフォルダとその配下です。'
      },
      timezone:
        'Hermes がローカル時刻のコンテキストを必要とするときに使用します。空欄ならシステムのタイムゾーンを使います。',
      agent: {
        imageInputMode: '画像添付をモデルへ送る方法を制御します。',
        maxTurns: 'Hermes が 1 回の実行を停止するまでのツール呼び出しターン上限です。'
      },
      terminal: {
        cwd: 'ツールとターミナル作業のデフォルトプロジェクトフォルダーです。',
        persistentShell: 'バックエンドが対応している場合、コマンド間でシェル状態を保持します。',
        envPassthrough: 'ツール実行へ渡す環境変数です。'
      },
      codeExecution: {
        mode: 'コード実行を現在のプロジェクトにどれだけ厳密に制限するかを設定します。'
      },
      fileReadMaxChars: 'Hermes が 1 回のファイル読み取りで取得できる最大文字数です。',
      approvals: {
        mode: '明示的な承認が必要なコマンドを Hermes がどう扱うかを設定します。',
        timeout: '承認プロンプトがタイムアウトするまで待つ時間です。'
      },
      security: {
        redactSecrets: '検出したシークレットを、可能な限りモデルから見える内容から隠します。'
      },
      checkpoints: {
        enabled: 'ファイル編集前にロールバック用スナップショットを作成します。'
      },
      memory: {
        memoryEnabled: '将来のセッションに役立つ永続メモリを保存します。',
        userProfileEnabled: 'ユーザーの好みをまとめた簡潔なプロファイルを維持します。'
      },
      context: {
        engine: '長い会話がコンテキスト上限に近づいたときの管理戦略です。'
      },
      compression: {
        enabled: '会話が大きくなったとき、古いコンテキストを要約します。'
      },
      voice: {
        autoTts: 'アシスタントの応答を自動で読み上げます。'
      },
      stt: {
        enabled: 'ローカルまたはプロバイダーによる音声文字起こしを有効にします。',
        elevenlabs: {
          languageCode: '任意の ISO-639-3 言語コードです。空欄なら ElevenLabs が自動検出します。'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'アプリから Hermes 自身を更新するとき、ローカルのソース変更を保持するか破棄するかを選びます。ターミナル更新では常に確認されます。'
      }
    }),
    about: {
      heading: 'Hermes Desktop',
      version: value => `バージョン ${value}`,
      versionUnavailable: 'バージョンを取得できません',
      updates: '更新',
      checkNow: '今すぐ確認',
      checking: '確認中…',
      seeWhatsNew: '新機能を見る',
      updateNow: '今すぐ更新',
      releaseNotes: 'リリースノート',
      onLatest: '最新バージョンです。',
      installing: '更新をインストール中です。',
      cantUpdate: 'このビルドはアプリ内から更新できません。',
      cantReach: '更新サーバーに接続できませんでした。',
      tapCheck: '更新を探すには「今すぐ確認」を押してください。',
      updateReady: count => `新しい更新の準備ができました (${count} 件の変更を含みます)。`,
      lastChecked: age => `前回確認: ${age}`,
      justNowSuffix: ' · たった今',
      automaticUpdates: '自動更新',
      automaticUpdatesDesc: 'Hermes はバックグラウンドで自動的に更新を確認し、利用可能になったら通知します。',
      branchCommit: (branch, commit) => `ブランチ ${branch} · コミット ${commit}`,
      never: '未確認',
      justNow: 'たった今',
      minAgo: count => `${count} 分前`,
      hoursAgo: count => `${count} 時間前`,
      daysAgo: count => `${count} 日前`
    },
    config: {
      none: 'なし',
      noneParen: '(なし)',
      builtinOnly: '内蔵のみ',
      notSet: '未設定',
      commaSeparated: 'カンマ区切りの値',
      loading: 'Hermes の設定を読み込み中...',
      emptyTitle: '設定項目がありません',
      emptyDesc: 'このセクションには調整できる設定がありません。',
      failedLoad: '設定の読み込みに失敗しました',
      autosaveFailed: '自動保存に失敗しました',
      imported: '設定をインポートしました',
      invalidJson: '設定 JSON が無効です',
      keepAwakeTitle: 'コンピューターをスリープさせない',
      keepAwakeDesc: '本体のスリープを防ぎ、長時間や夜通しの実行を継続します。画面は暗転できます。'
    },
    credentials: {
      pasteKey: 'キーを貼り付け',
      pasteLabelKey: label => `${label} キーを貼り付け`,
      optional: '省略可能',
      enterValueFirst: '最初に値を入力してください。',
      couldNotSave: '認証情報を保存できませんでした。',
      remove: '削除',
      getKey: 'キーを取得',
      saving: '保存中'
    },
    envActions: {
      actionsFor: label => `${label} のアクション`,
      credentialActions: '認証情報のアクション',
      manageInKeys: 'API キーで管理',
      docs: 'ドキュメント',
      hideValue: '値を非表示',
      revealValue: '値を表示',
      replace: '置き換え',
      set: '設定',
      clear: 'クリア'
    },
    gateway: {
      loading: 'ゲートウェイ設定を読み込み中...',
      unavailableTitle: 'ゲートウェイ設定は利用できません',
      unavailableDesc: 'デスクトップ IPC ブリッジはゲートウェイ設定を公開していません。',
      title: 'ゲートウェイ接続',
      envOverride: 'env オーバーライド',
      intro:
        'Hermes Desktop はデフォルトで独自のローカルゲートウェイを起動します。別のマシンや信頼できるプロキシの背後で既に動作している Hermes バックエンドをこのアプリで制御する場合は、リモートゲートウェイを使用してください。以下でプロファイルを選択して、それぞれのリモートホストを設定します。',
      appliesTo: '適用対象',
      allProfiles: 'すべてのプロファイル',
      defaultConnection: '独自のオーバーライドがないすべてのプロファイルのデフォルト接続。',
      profileConnection: profile =>
        `"${profile}" がアクティブプロファイルのときのみ使用される接続。ローカルに設定するとデフォルトを継承します。`,
      envOverrideTitle: '環境変数がこのデスクトップセッションを制御しています。',
      envOverrideDesc:
        '保存された設定を使用するには HERMES_DESKTOP_REMOTE_URL と HERMES_DESKTOP_REMOTE_TOKEN の設定を解除してください。',
      localTitle: 'ローカルゲートウェイ',
      localDesc:
        'ローカルホストでプライベートな Hermes バックエンドを起動します。これがデフォルトで、オフラインでも動作します。',
      remoteTitle: 'リモートゲートウェイ',
      remoteDesc:
        'このデスクトップシェルをリモートの Hermes バックエンドに接続します。ホスト型ゲートウェイは OAuth またはユーザー名とパスワードを使用します。自己ホスト型はセッショントークンを使用する場合があります。',
      remoteUrlTitle: 'リモート URL',
      remoteUrlDesc:
        'リモートダッシュボードバックエンドのベース URL。/hermes などのパスプレフィックスもサポートしています。',
      probing: 'このゲートウェイの認証方法を確認中…',
      probeError: 'このゲートウェイにまだ到達できません。URL を確認してください。応答後に認証方法が表示されます。',
      signedIn: 'サインイン済み',
      signIn: 'サインイン',
      signOut: 'サインアウト',
      signInWith: provider => `${provider} でサインイン`,
      authTitle: '認証',
      authSignedInPassword:
        'このゲートウェイはユーザー名とパスワードを使用します。サインイン済みです。セッションは自動的に更新されます。',
      authSignedInOauth:
        'このゲートウェイは OAuth を使用します。サインイン済みです。セッションは自動的に更新されます。',
      authNeedsPassword:
        'このゲートウェイはユーザー名とパスワードを使用します。このデスクトップアプリを承認するにはサインインしてください。',
      authNeedsOauth: provider =>
        `このゲートウェイは OAuth を使用します。このデスクトップアプリを承認するには ${provider} でサインインしてください。`,
      tokenTitle: 'セッショントークン',
      tokenDesc:
        'REST および WebSocket アクセスに使用するダッシュボードセッショントークン。保存済みトークンを維持するには空欄にしてください。',
      existingToken: value => `既存のトークン ${value}`,
      savedToken: '保存済み',
      pasteSessionToken: 'セッショントークンを貼り付け',
      testRemote: 'リモートをテスト',
      saveForRestart: '次回起動時のために保存',
      saveAndReconnect: '保存して再接続',
      diagnostics: '診断',
      diagnosticsDesc: 'ファイルマネージャーで desktop.log を表示します。ゲートウェイの起動に失敗した際に役立ちます。',
      openLogs: 'ログを開く',
      incompleteTitle: 'リモートゲートウェイの設定が不完全です',
      incompleteSignIn: 'リモートに切り替える前にリモート URL を入力してサインインしてください。',
      incompleteToken: 'リモートに切り替える前にリモート URL とセッショントークンを入力してください。',
      incompleteSignInTest: 'テストする前にリモート URL を入力してサインインしてください。',
      incompleteTokenTest: 'テストする前にリモート URL とセッショントークンを入力してください。',
      enterUrlFirst: '最初にリモート URL を入力してください。',
      restartingTitle: 'ゲートウェイ接続を再起動中',
      savedTitle: 'ゲートウェイ設定を保存しました',
      restartingMessage: 'Hermes Desktop は保存された設定を使用して再接続します。',
      savedMessage: '次回起動時に保存されます。',
      connectedTo: (baseUrl, version) => `${baseUrl}${version ? ` · Hermes ${version}` : ''} に接続しました`,
      reachableTitle: 'リモートゲートウェイに到達可能',
      signedOutTitle: 'サインアウトしました',
      signedOutMessage: 'リモートゲートウェイセッションをクリアしました。',
      failedLoad: 'ゲートウェイ設定の読み込みに失敗しました',
      signInFailed: 'サインインに失敗しました',
      signOutFailed: 'サインアウトに失敗しました',
      testFailed: 'リモートゲートウェイのテストに失敗しました',
      applyFailed: 'ゲートウェイ設定を適用できませんでした',
      saveFailed: 'ゲートウェイ設定を保存できませんでした',
      sshTitle: 'SSH で接続',
      sshDesc:
        'Hermes は SSH 経由でリモート上に起動され、このアプリにトンネルされます。リモート側で何かを起動・公開する必要はありません。ホストへの鍵ベースの SSH アクセスが前提です。',
      sshTrustHint: '初回に提示されたホスト鍵を信頼して固定し、以後の変更は拒否します。',
      sshHostTitle: 'ホスト',
      sshHostDesc: 'user@host、または ~/.ssh/config の Host エイリアス。',
      sshHostPick: 'ホストを選択…',
      sshHostPickTitle: 'ホスト',
      sshHostPickDesc: '~/.ssh/config の Host エイリアス、または「カスタム」で手入力。',
      sshHostCustom: 'カスタム（手入力）…',
      sshUserTitle: 'ユーザー',
      sshUserDesc: '空欄 = ~/.ssh/config または現在のユーザー。',
      sshUserPlaceholder: '~/.ssh/config から',
      sshPortTitle: 'ポート',
      sshPortDesc: '空欄 = 22 または ~/.ssh/config のポート。',
      sshKeyTitle: '鍵ファイル',
      sshKeyDesc: '秘密鍵のパス。空欄 = ssh-agent または ~/.ssh/config。',
      sshHermesPathTitle: 'Hermes パス（任意）',
      sshHermesPathDesc: 'リモートの hermes バイナリへのフルパス。空欄 = 自動検出。',
      sshHermesPathPlaceholder: '自動検出',
      sshTestConnection: 'SSH をテスト',
      sshConnect: '接続',
      sshButtonsHint: '「保存」は次回起動時に適用され、「接続」は今すぐ再接続します。',
      sshReachable: (host, platform) => `接続可能: ${host}（${platform}）— Hermes を検出`,
      sshIncompleteHost: '接続する前に SSH ホストを入力してください。',
      sshErrUnreachable: 'SSH でそのホストに到達できませんでした。ホスト、ポート、ネットワークを確認してください。',
      sshErrAuth:
        'SSH 認証に失敗しました。鍵を ssh-agent に読み込む（ssh-add）か、~/.ssh/config に IdentityFile を設定してください。Hermes は非対話的に ssh を実行します。',
      sshErrHostKey:
        '前回の接続以降、ホスト鍵が変更されています。想定どおりか確認し、ssh-keygen -R <host> を実行してから再接続してください。',
      sshErrNotInstalled:
        'リモートホストに Hermes がインストールされていません。リモートでインストールする（curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh）か、Hermes パスを設定してください。',
      sshErrPlatform:
        'サポートされていないリモートプラットフォームです。Hermes Desktop の SSH モードは Linux、macOS、Windows のリモートホストに対応しています。',
      sshErrTimeout: 'SSH 接続がタイムアウトしました。ホストが到達不能、またはスリープ中の可能性があります。',
      sshErrUpdateRequired: 'Desktop SSH で接続する前に、リモートホストの Hermes を更新してください。',
      sshErrUnknown: 'SSH 接続に失敗しました。'
    },
    keys: {
      loading: 'API キーと認証情報を読み込み中...',
      failedLoad: 'API キーの読み込みに失敗しました',
      empty: 'このカテゴリーにはまだ設定がありません。'
    },
    envKeys: {
      NOUS_BASE_URL: { description: 'Nous Portal ベース URL の上書き' },
      OPENROUTER_API_KEY: { description: 'OpenRouter API キー（ビジョン、ウェブ抽出ヘルパー、MOA 用）' },
      GOOGLE_API_KEY: { description: 'Google AI Studio API キー（GEMINI_API_KEY としても認識）' },
      GEMINI_API_KEY: { description: 'Google AI Studio API キー（GOOGLE_API_KEY のエイリアス）' },
      GEMINI_BASE_URL: { description: 'Google AI Studio ベース URL の上書き' },
      VERTEX_CREDENTIALS_PATH: {
        description:
          'Vertex AI (Gemini) 用の Google Cloud サービスアカウント JSON のパス。Vertex は静的 API キーではなく OAuth2 を使用し、Hermes はこの資格情報から短期トークンを発行します。GOOGLE_APPLICATION_CREDENTIALS、次いで ADC (gcloud auth application-default login) にフォールバックします。プロジェクト/リージョンは config.yaml の vertex: で設定します。'
      },
      XAI_API_KEY: { description: 'xAI API キー' },
      XAI_BASE_URL: { description: 'xAI ベース URL の上書き' },
      NVIDIA_API_KEY: { description: 'NVIDIA NIM API キー（build.nvidia.com またはローカル NIM エンドポイント）' },
      NVIDIA_BASE_URL: { description: 'NVIDIA NIM ベース URL の上書き（ローカル NIM の http://localhost:8000/v1 など）' },
      LM_API_KEY: { description: '認証を有効にした LM Studio ローカルサーバー用の Bearer トークン' },
      LM_BASE_URL: { description: 'LM Studio ベース URL の上書き' },
      GLM_API_KEY: { description: 'Z.AI / GLM API キー（ZAI_API_KEY / Z_AI_API_KEY としても認識）' },
      ZAI_API_KEY: { description: 'Z.AI API キー（GLM_API_KEY のエイリアス）' },
      Z_AI_API_KEY: { description: 'Z.AI API キー（GLM_API_KEY のエイリアス）' },
      GLM_BASE_URL: { description: 'Z.AI / GLM ベース URL の上書き' },
      KIMI_API_KEY: { description: 'Kimi / Moonshot API キー' },
      KIMI_BASE_URL: { description: 'Kimi / Moonshot ベース URL の上書き' },
      KIMI_CN_API_KEY: { description: 'Kimi / Moonshot 中国リージョンの API キー' },
      STEPFUN_API_KEY: { description: 'StepFun Step Plan API キー' },
      STEPFUN_BASE_URL: { description: 'StepFun Step Plan ベース URL の上書き' },
      ARCEEAI_API_KEY: { description: 'Arcee AI API キー' },
      ARCEE_BASE_URL: { description: 'Arcee AI ベース URL の上書き' },
      GMI_API_KEY: { description: 'GMI Cloud API キー' },
      GMI_BASE_URL: { description: 'GMI Cloud ベース URL の上書き' },
      FIREWORKS_API_KEY: { description: 'Fireworks AI API キー' },
      MINIMAX_API_KEY: { description: 'MiniMax API キー（インターナショナル）' },
      MINIMAX_BASE_URL: { description: 'MiniMax ベース URL の上書き' },
      MINIMAX_CN_API_KEY: { description: 'MiniMax API キー（中国エンドポイント）' },
      MINIMAX_CN_BASE_URL: { description: 'MiniMax（中国）ベース URL の上書き' },
      DEEPSEEK_API_KEY: { description: 'DeepSeek 直接アクセス用の API キー' },
      DEEPSEEK_BASE_URL: { description: 'カスタム DeepSeek API ベース URL（上級者向け）' },
      DASHSCOPE_API_KEY: { description: 'Alibaba Cloud DashScope API キー（Qwen + マルチプロバイダーモデル）' },
      DASHSCOPE_BASE_URL: { description: 'カスタム DashScope ベース URL（デフォルト: coding-intl の OpenAI 互換エンドポイント）' },
      HERMES_QWEN_BASE_URL: { description: 'Qwen Portal ベース URL の上書き（デフォルト: https://portal.qwen.ai/v1）' },
      OPENCODE_ZEN_API_KEY: { description: 'OpenCode Zen API キー（従量課金で厳選モデルを利用）' },
      OPENCODE_ZEN_BASE_URL: { description: 'OpenCode Zen ベース URL の上書き' },
      OPENCODE_GO_API_KEY: { description: 'OpenCode Go API キー（月額 10 ドルのサブスクリプションでオープンモデルを利用）' },
      OPENCODE_GO_BASE_URL: { description: 'OpenCode Go ベース URL の上書き' },
      HF_TOKEN: {
        description: 'Inference Providers 用の Hugging Face トークン（router.huggingface.co 経由で 20+ のオープンモデル）'
      },
      HF_BASE_URL: { description: 'Hugging Face Inference Providers ベース URL の上書き' },
      OLLAMA_API_KEY: { description: 'Ollama Cloud API キー（ollama.com — クラウドホストのオープンモデル）' },
      OLLAMA_BASE_URL: { description: 'Ollama Cloud ベース URL の上書き（デフォルト: https://ollama.com/v1）' },
      XIAOMI_API_KEY: { description: 'Xiaomi MiMo API キー（mimo-v2.5-pro、mimo-v2.5 などの MiMo モデル用）' },
      XIAOMI_BASE_URL: { description: 'Xiaomi MiMo ベース URL の上書き（デフォルト: https://api.xiaomimimo.com/v1）' },
      UPSTAGE_API_KEY: { description: 'Solar LLM モデル用の Upstage API キー' },
      UPSTAGE_BASE_URL: { description: 'Upstage ベース URL の上書き（デフォルト: https://api.upstage.ai/v1）' },
      AWS_REGION: { description: 'Bedrock API 呼び出しの AWS リージョン（us-east-1、eu-central-1 など）' },
      AWS_PROFILE: { description: 'Bedrock 認証用の AWS 名前付きプロファイル（~/.aws/credentials から）' },
      AZURE_FOUNDRY_API_KEY: { description: 'カスタム Azure エンドポイント用の Azure Foundry API キー' },
      AZURE_FOUNDRY_BASE_URL: { description: 'Azure Foundry ベース URL（エンドポイント別の設定は hermes model で）' },
      ALIBABA_CODING_PLAN_API_KEY: { description: 'Alibaba Cloud (Coding Plan) API キー' },
      ALIBABA_CODING_PLAN_BASE_URL: { description: 'Alibaba Cloud (Coding Plan) ベース URL の上書き' },
      ANTHROPIC_API_KEY: { description: 'Anthropic API キー' },
      ANTHROPIC_TOKEN: { description: 'Anthropic API キー' },
      CLAUDE_CODE_OAUTH_TOKEN: { description: 'Anthropic API キー' },
      DEEPINFRA_API_KEY: { description: 'DeepInfra API キー' },
      DEEPINFRA_BASE_URL: { description: 'DeepInfra ベース URL の上書き' },
      KILOCODE_API_KEY: { description: 'Kilocode API キー' },
      KIMI_CODING_API_KEY: { description: 'Kimi Coding API キー' },
      NOVITA_API_KEY: { description: 'NovitaAI API キー' },
      NOVITA_BASE_URL: { description: 'NovitaAI ベース URL の上書き' },
      EXA_API_KEY: { description: 'AI ネイティブなウェブ検索とコンテンツ取得のための Exa API キー' },
      PARALLEL_API_KEY: { description: 'AI ネイティブなウェブ検索と抽出のための Parallel API キー' },
      FIRECRAWL_API_KEY: { description: 'ウェブ検索とスクレイピングのための Firecrawl API キー' },
      FIRECRAWL_API_URL: { description: 'セルフホストの Firecrawl インスタンス用 API URL（任意）' },
      FIRECRAWL_GATEWAY_URL: { description: 'Nous 購読者専用の Firecrawl ツールゲートウェイの上書き（任意）' },
      TOOL_GATEWAY_DOMAIN: {
        description:
          'Nous 購読者専用の共有ツールゲートウェイのドメインサフィックス。ベンダーホストの導出に使用（例: nousresearch.com -> firecrawl-gateway.nousresearch.com）'
      },
      TOOL_GATEWAY_SCHEME: {
        description: 'Nous 購読者専用の共有ツールゲートウェイの URL スキーム（デフォルト https、ローカルテストでは http）'
      },
      TOOL_GATEWAY_USER_TOKEN: {
        description: 'ツールゲートウェイリクエスト用の Nous 購読者アクセストークン（任意。省略時は Hermes 認証ストアから取得）'
      },
      TAVILY_API_KEY: { description: 'AI ネイティブなウェブ検索と抽出のための Tavily API キー' },
      SEARXNG_URL: { description: '無料セルフホストのウェブ検索用 SearXNG インスタンスの URL' },
      BRAVE_SEARCH_API_KEY: { description: 'Brave Search API サブスクリプショントークン（無料枠: 月 2,000 クエリ）' },
      BROWSERBASE_API_KEY: { description: 'クラウドブラウザ用の Browserbase API キー（任意 — ローカルブラウザには不要）' },
      BROWSERBASE_PROJECT_ID: { description: 'Browserbase プロジェクト ID（任意 — クラウドブラウザのみ必要）' },
      BROWSER_USE_API_KEY: { description: 'クラウドブラウザ用の Browser Use API キー（任意 — ローカルブラウザには不要）' },
      FIRECRAWL_BROWSER_TTL: { description: 'Firecrawl ブラウザセッションの TTL（秒、任意、デフォルト 300）' },
      AGENT_BROWSER_ENGINE: {
        description: 'ローカルモードのブラウザエンジン: auto（デフォルト Chrome）、lightpanda（高速、スクリーンショットなし）、chrome'
      },
      CAMOFOX_URL: { description: 'ローカル検出回避ブラウジング用の Camofox ブラウザサーバー URL（例: http://localhost:9377）' },
      CAMOFOX_API_KEY: { description: 'リモート/認証付き Camofox サーバーへ送る任意の Bearer トークン' },
      FAL_KEY: { description: '画像・動画生成のための FAL API キー' },
      KREA_API_KEY: { description: 'Krea 2 画像生成のための Krea API キー（Medium + Large）' },
      VOICE_TOOLS_OPENAI_KEY: { description: '音声文字起こし (Whisper) と OpenAI TTS 用の OpenAI API キー' },
      ELEVENLABS_API_KEY: { description: '高品質音声合成と Scribe 文字起こしのための ElevenLabs API キー' },
      MISTRAL_API_KEY: { description: 'Voxtral TTS と文字起こし (STT) のための Mistral API キー' },
      GITHUB_TOKEN: { description: 'スキルハブ用の GitHub トークン（API レート制限の緩和、スキル公開）' },
      HONCHO_API_KEY: { description: 'AI ネイティブ永続メモリのための Honcho API キー' },
      HONCHO_BASE_URL: { description: 'セルフホスト Honcho インスタンスのベース URL（API キー不要）' },
      HINDSIGHT_API_KEY: { description: 'グラフ対応の永続メモリのための Hindsight API キー' },
      HINDSIGHT_API_URL: { description: 'Hindsight API のベース URL（デフォルト: https://api.hindsight.vectorize.io）' },
      SUPERMEMORY_API_KEY: { description: '会話スコープの永続メモリのための Supermemory API キー' },
      MEM0_API_KEY: { description: 'セマンティック永続メモリのための Mem0 Platform API キー' },
      RETAINDB_API_KEY: { description: '永続メモリのための RetainDB API キー' },
      RETAINDB_BASE_URL: { description: 'セルフホスト RetainDB インスタンスのベース URL（デフォルト: https://api.retaindb.com）' },
      BRV_API_KEY: { description: 'ByteRover API キー（任意、クラウド同期用 — デフォルトはローカル優先）' },
      OPENVIKING_API_KEY: { description: 'OpenViking API キー（ローカル開発モードでは空欄可）' },
      OPENVIKING_ENDPOINT: { description: 'OpenViking サーバー URL（デフォルト: http://127.0.0.1:1933）' },
      HERMES_LANGFUSE_PUBLIC_KEY: { description: 'Langfuse プロジェクト公開キー (pk-lf-...)' },
      HERMES_LANGFUSE_SECRET_KEY: { description: 'Langfuse プロジェクト秘密キー (sk-lf-...)' },
      HERMES_LANGFUSE_BASE_URL: { description: 'Langfuse サーバー URL（デフォルト: https://cloud.langfuse.com）' },
      NOTION_API_KEY: { description: 'Notion 統合トークン（notion スキルで使用）' },
      LINEAR_API_KEY: { description: 'Linear 個人 API キー（linear スキルで使用）' },
      AIRTABLE_API_KEY: { description: 'Airtable 個人アクセストークン（airtable スキルで使用）' },
      TENOR_API_KEY: { description: 'GIF 検索のための Tenor API キー（gif-search スキルで使用）' },
      SUDO_PASSWORD: {
        description: 'root 権限が必要なターミナルコマンドで使う sudo パスワード。明示的な空文字を設定するとプロンプトなしで空を試します'
      },
      HERMES_PREFILL_MESSAGES_FILE: { description: 'few-shot プライミング用の一時プリフィルメッセージ JSON ファイルのパス' },
      HERMES_EPHEMERAL_SYSTEM_PROMPT: { description: 'API 呼び出し時に注入される一時システムプロンプト（セッションには保存されません）' },
      RAFT_PROFILE: { description: 'Raft エージェントプロファイルの slug — 設定するとアダプターが自動で有効になります' },
      GATEWAY_ALLOW_ALL_USERS: {
        description: 'すべてのユーザーにメッセージングボットとの対話を許可します（true/false）。デフォルト: false。'
      },
      GATEWAY_PROXY_URL: {
        description:
          'メッセージを転送するリモート Hermes API サーバーの URL（プロキシモード）。設定するとゲートウェイはプラットフォーム I/O のみを担当し、エージェント処理はすべてリモートサーバーに委任されます。ホストエージェントに中継する Docker E2EE コンテナ向け。config.yaml の gateway.proxy_url でも設定可能。'
      },
      GATEWAY_PROXY_KEY: {
        description:
          'リモート Hermes API サーバーとの認証用 Bearer トークン（プロキシモード）。リモートホストの API_SERVER_KEY と一致させる必要があります。'
      }
    },
    mcp: {
      loading: 'MCP サーバーを読み込み中...',
      failedLoad: 'MCP 設定の読み込みに失敗しました',
      nameRequiredTitle: '名前が必要です',
      nameRequiredMessage: 'この MCP サーバーに設定キーを付けてください。',
      objectRequired: 'サーバー設定は JSON オブジェクトである必要があります',
      invalidJson: '無効な MCP JSON',
      saveFailed: '保存に失敗しました',
      removeFailed: '削除に失敗しました',
      gatewayUnavailableTitle: 'ゲートウェイが利用できません',
      gatewayUnavailableMessage: 'MCP を再読み込みする前にゲートウェイを再接続してください。',
      reloadedTitle: 'MCP ツールを再読み込みしました',
      reloadedMessage: '新しいツールスキーマは新しいターンに適用されます。',
      reloadFailed: 'MCP の再読み込みに失敗しました',
      savedTitle: 'MCP サーバーを保存しました',
      savedMessage: name => `${name} は MCP の再読み込み後に適用されます。`,
      newServer: '新しいサーバー',
      reload: 'MCP を再読み込み',
      reloading: '再読み込み中...',
      emptyTitle: 'MCP サーバーがありません',
      emptyDesc: 'MCP ツールを公開するには stdio または HTTP サーバーを追加してください。',
      disabled: '無効',
      editServer: 'サーバーを編集',
      name: '名前',
      serverJson: 'サーバー JSON',
      remove: '削除',
      saveServer: 'サーバーを保存',
      capabilitySummary: (tools, prompts, resources) =>
        `${[`ツール ${tools} 個`, ...(prompts ? [`プロンプト ${prompts} 個`] : []), ...(resources ? [`リソース ${resources} 個`] : [])].join('、')} を有効化`,
      statusConnecting: '接続中…',
      statusNeedsAuth: '認証が必要です',
      statusError: 'エラー',
      statusOff: 'オフ',
      allServers: 'すべてのサーバー',
      authenticatedTitle: '認証済み',
      authenticatedMessage: (server, count) => `${server}: ツール ${count} 個`,
      waitingForBrowser: 'ブラウザを待機中…',
      authenticate: '認証',
      unsavedConnect: '未保存 — 接続するには mcp.json を保存してください。',
      enableTool: tool => `${tool} を有効化`,
      disableTool: tool => `${tool} を無効化`,
      noOutput: 'まだ出力がありません。'
    },
    model: {
      loading: 'モデル設定を読み込み中...',
      appliesDesc:
        '新しいセッションに適用されます。コンポーザーのモデルピッカーを使ってアクティブなチャットをホットスワップできます。',
      provider: 'プロバイダー',
      model: 'モデル',
      applying: '適用中...',
      auxiliaryTitle: '補助モデル',
      resetAllToMain: 'すべてメインにリセット',
      auxiliaryDesc:
        'ヘルパータスクはデフォルトでメインモデルで実行されます。タスクに専用モデルを割り当てることでオーバーライドできます。',
      setToMain: 'メインに設定',
      change: '変更',
      autoUseMain: '自動 · メインモデルを使用',
      providerDefault: '(プロバイダーのデフォルト)',
      staleAuxPrefix: (count, names) => `${count} 件の補助タスク (${names}) は引き続き `,
      staleAuxOtherProviders: '他のプロバイダー',
      staleAuxSuffix: ' で実行されます — メインモデルではありません。',
      pasteKeyPlaceholder: keyEnv => `${keyEnv} を貼り付け`,
      activate: '有効化',
      activating: '有効化中...',
      setUpProvider: name => `${name} を設定`,
      needsApiKeyHint: name => `${name} には API キーが必要です — 設定するとモデルを選択できます。`,
      oauthHint: name => `${name} はブラウザーでサインインします — Hermes がフローを代行します。`,
      moa: {
        title: 'Mixture of Agents',
        description:
          'Mixture of Agents プロバイダーの下にモデルとして表示される名前付きプリセットを設定します。アグリゲーターが実際に応答するモデルです。',
        presetPlaceholder: 'プリセット',
        setDefault: 'デフォルトに設定',
        deletePreset: '削除',
        newPresetPlaceholder: '新しいプリセット名',
        addPreset: 'プリセットを追加',
        defaultLabel: 'デフォルト:',
        referenceTitle: index => `リファレンス ${index}`,
        removeReference: '削除',
        addReference: 'リファレンスモデルを追加',
        aggregatorTitle: 'アグリゲーター'
      },
      tasks: {
        vision: { label: 'ビジョン', hint: '画像分析' },
        web_extract: { label: 'ウェブ抽出', hint: 'ページの要約' },
        compression: { label: '圧縮', hint: 'コンテキストの圧縮' },
        skills_hub: { label: 'スキルハブ', hint: 'スキル検索' },
        approval: { label: '承認', hint: 'スマート自動承認' },
        mcp: { label: 'MCP', hint: 'MCP ツールルーティング' },
        title_generation: { label: 'タイトル生成', hint: 'セッションタイトル' },
        curator: { label: 'キュレーター', hint: 'スキル使用レビュー' }
      }
    },
    providers: {
      connectAccount: 'アカウントを接続',
      haveApiKey: 'API キーをお持ちですか？',
      intro:
        'サブスクリプションでサインインします。API キーのコピーは不要です。Hermes がアプリ内でブラウザーサインインを代行します。',
      connected: '接続済み',
      collapse: '折りたたむ',
      connectAnother: '別のプロバイダーを接続',
      otherProviders: 'その他のプロバイダー',
      removeConfirm: provider => `${provider} を削除しますか？`,
      removeKeyManaged: provider => `${provider} は API キーで設定されています。API Keys から削除してください。`,
      removedTitle: 'アカウントを削除しました',
      removedMessage: provider => `${provider} を削除しました。`,
      failedRemove: provider => `${provider} を削除できませんでした`,
      noProviderKeys: '利用可能なプロバイダー API キーがありません。',
      searchKeys: 'プロバイダーを検索…',
      noKeysMatch: '一致するプロバイダーがありません。',
      localEndpoint: {
        title: 'ローカル / カスタムエンドポイント',
        description: 'OpenAI 互換のエンドポイント（Zyphra、vLLM、llama.cpp、Ollama など）を指定します。'
      },
      loading: 'プロバイダーを読み込み中...'
    },
    sessions: {
      loading: 'アーカイブ済みセッションを読み込み中…',
      archivedTitle: 'アーカイブ済みセッション',
      archivedIntro:
        'アーカイブ済みチャットはサイドバーでは非表示になりますが、すべてのメッセージは保持されます。サイドバーのチャットを Ctrl/⌘ クリックするとアーカイブできます。',
      emptyArchivedTitle: 'アーカイブがありません',
      emptyArchivedDesc: 'チャットをアーカイブするとここに表示されます。',
      unarchive: 'アーカイブを解除',
      deletePermanently: '完全に削除',
      messages: count => `${count} 件のメッセージ`,
      restored: '復元しました',
      deleteConfirm: title => `"${title}" を完全に削除しますか？この操作は元に戻せません。`,
      defaultDirTitle: 'デフォルトのプロジェクトディレクトリ',
      defaultDirDesc:
        '別のフォルダーを選択しない限り、新しいセッションはこのフォルダーで開始します。未設定の場合はホームディレクトリが使用されます。',
      defaultDirUpdated: 'デフォルトのプロジェクトディレクトリを更新しました',
      defaultsTo: label => `デフォルト: ${label}。`,
      change: '変更',
      choose: '選択',
      clear: 'クリア',
      notSet: '未設定',
      failedLoad: 'アーカイブ済みセッションを読み込めませんでした',
      unarchiveFailed: 'アーカイブ解除に失敗しました',
      deleteFailed: '削除に失敗しました',
      updateDirFailed: 'デフォルトディレクトリを更新できませんでした',
      clearDirFailed: 'デフォルトディレクトリをクリアできませんでした'
    },
    toolsets: {
      loadingConfig: '設定を読み込み中',
      savedTitle: '認証情報を保存しました',
      savedMessage: key => `${key} を更新しました。`,
      removedTitle: '認証情報を削除しました',
      removedMessage: key => `${key} を削除しました。`,
      failedSave: key => `${key} の保存に失敗しました`,
      failedRemove: key => `${key} の削除に失敗しました`,
      failedReveal: key => `${key} の表示に失敗しました`,
      removeConfirm: key => `.env から ${key} を削除しますか？`,
      set: '設定済み',
      notSet: '未設定',
      selectedTitle: 'プロバイダーを選択しました',
      selectedMessage: provider => `${provider} が有効になりました。`,
      failedSelect: provider => `${provider} の選択に失敗しました`,
      failedLoad: 'ツール設定の読み込みに失敗しました',
      noProviderOptions:
        'このツールセットにはプロバイダーのオプションがありません。有効にすれば現在の設定で動作します。',
      noProviders: '現在このツールセットに利用可能なプロバイダーがありません。',
      ready: '準備完了',
      needsSignIn: 'サインインが必要',
      needsSetup: 'セットアップが必要',
      badgeTokens: {
        recommended: 'おすすめ',
        free: '無料',
        local: 'ローカル',
        'self-hosted': 'セルフホスト',
        paid: '有料',
        preview: 'プレビュー',
        subscription: 'サブスクリプション',
        'no key': 'キー不要',
        'search only': '検索のみ',
        'optional gateway': 'ゲートウェイ任意'
      },
      tagCopy: {
        '30 prebuilt voices, controllable via prompts': '30 種のプリセット音声、プロンプトで制御可能',
        'Anti-detection browser (Firefox/Camoufox)': '検出回避ブラウザ (Firefox/Camoufox)',
        'Background computer-use via cua-driver — does NOT steal your cursor or focus. Works with any model.':
          'cua-driver によるバックグラウンドのコンピューター操作 — カーソルやフォーカスを奪いません。あらゆるモデルで動作。',
        'Browser login at accounts.x.ai — no API key required': 'accounts.x.ai でブラウザログイン — API キー不要',
        'Chatterbox, Qwen3-TTS, … — live catalog from api.deepinfra.com':
          'Chatterbox、Qwen3-TTS など — api.deepinfra.com のライブカタログ',
        'Direct xAI API billing via XAI_API_KEY': 'XAI_API_KEY による xAI API 直接課金',
        'Good quality, no API key needed': '良好な品質、API キー不要',
        'Grok voices — uses xAI Grok OAuth or XAI_API_KEY': 'Grok 音声 — xAI Grok OAuth または XAI_API_KEY を使用',
        'Headless Chromium, no API key needed': 'ヘッドレス Chromium、API キー不要',
        'High quality voices': '高品質な音声',
        'Hosted Langfuse (cloud.langfuse.com)': 'ホスト版 Langfuse (cloud.langfuse.com)',
        'Lightweight local ONNX TTS (~25MB), no API key': '軽量ローカル ONNX TTS（約 25MB）、API キー不要',
        'Local neural TTS, 44 languages (voices ~20-90MB)': 'ローカルのニューラル TTS、44 言語対応（音声 約 20-90MB）',
        'Managed Browser Use billed to your subscription': 'マネージド Browser Use、サブスクリプションに課金',
        'Managed FAL image generation billed to your subscription': 'マネージド FAL 画像生成、サブスクリプションに課金',
        'Managed FAL video generation billed to your subscription': 'マネージド FAL 動画生成、サブスクリプションに課金',
        'Managed Firecrawl billed to your subscription': 'マネージド Firecrawl、サブスクリプションに課金',
        'Managed OpenAI TTS billed to your subscription': 'マネージド OpenAI TTS、サブスクリプションに課金',
        'Most natural voices': '最も自然な音声',
        'Multilingual, native Opus': '多言語対応、ネイティブ Opus',
        'PKCE OAuth — opens the setup wizard': 'PKCE OAuth — セットアップウィザードを開きます',
        'REST API integration': 'REST API 連携',
        'Run your own Firecrawl instance (Docker)': '自前の Firecrawl インスタンスを実行 (Docker)',
        'Self-hosted Langfuse instance': 'セルフホストの Langfuse インスタンス',
        "Agentic web search via Grok's web_search tool — uses xAI Grok OAuth or XAI_API_KEY.":
          'Grok の web_search ツールによるエージェント型ウェブ検索 — xAI Grok OAuth または XAI_API_KEY を使用。',
        'Cloud browser with remote execution': 'リモート実行対応のクラウドブラウザ',
        'Cloud browser with stealth and proxies': 'ステルスとプロキシ対応のクラウドブラウザ',
        'FLUX, Qwen-Image, … — live catalog from api.deepinfra.com':
          'FLUX、Qwen-Image など — api.deepinfra.com のライブカタログ',
        'Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.':
          '無料でプライバシーに配慮したメタ検索。SEARXNG_URL を自分のインスタンスに向けてください。',
        'Free-tier API key — 2k queries/mo, search only.': '無料枠 API キー — 月 2,000 クエリ、検索のみ。',
        'Full search + extract; supports direct API and Nous tool-gateway routing.':
          'フル検索 + 抽出。直接 API と Nous ツールゲートウェイ経由の両方に対応。',
        'Gemini Flash Image & more via OpenRouter; uses OPENROUTER_API_KEY':
          'OpenRouter 経由の Gemini Flash Image など。OPENROUTER_API_KEY を使用',
        'Krea 2 foundation model — Medium ($0.03), Large ($0.06), Medium Turbo ($0.015). Style transfer, moodboards, reference-guided generation. Direct key or managed Nous Subscription gateway.':
          'Krea 2 基盤モデル — Medium ($0.03)、Large ($0.06)、Medium Turbo ($0.015)。スタイル転送、ムードボード、参照ガイド生成。直接キーまたはマネージド Nous サブスクリプションゲートウェイ。',
        'LTX, Pixverse, Veo 3.1, Seedance 2.0, Kling 4K, Happy Horse — text-to-video & image-to-video':
          'LTX、Pixverse、Veo 3.1、Seedance 2.0、Kling 4K、Happy Horse — テキストから動画・画像から動画',
        'Objective-tuned search + parallel page extraction.': '目的別にチューニングされた検索 + 並列ページ抽出。',
        'Pick from flux-2-klein, flux-2-pro, gpt-image, nano-banana, etc. — text-to-image & image editing':
          'flux-2-klein、flux-2-pro、gpt-image、nano-banana などから選択 — テキストから画像・画像編集',
        'Reference-grounded image generation via Nous Portal (OpenRouter-backed)':
          'Nous Portal 経由の参照グラウンディング画像生成（OpenRouter バックエンド）',
        'Search + extract in one provider.': '検索 + 抽出を 1 つのプロバイダーで。',
        'Search via the ddgs Python package — no API key (pair with any extract provider)':
          'ddgs Python パッケージによる検索 — API キー不要（任意の抽出プロバイダーと併用可）',
        'Semantic + neural web search with content extraction.': 'セマンティック + ニューラルのウェブ検索とコンテンツ抽出。',
        'Wan, p-video, … — live catalog from api.deepinfra.com; text-to-video & image-to-video':
          'Wan、p-video など — api.deepinfra.com のライブカタログ。テキストから動画・画像から動画',
        'gpt-image-2 at low/medium/high quality tiers — text-to-image & image editing':
          'gpt-image-2、低/中/高の品質ティア — テキストから画像・画像編集',
        'gpt-image-2 via ChatGPT/Codex OAuth — no API key required; supports text and image inputs':
          'ChatGPT/Codex OAuth 経由の gpt-image-2 — API キー不要。テキストと画像の入力に対応',
        'grok-imagine-image - text-to-image & image editing; uses xAI Grok OAuth or XAI_API_KEY. xAI Imagine storage is enabled so generated media gets a reusable public URL without an automatic expiry. xAI may bill for stored files and public URL hosting. Disable this with `image_gen.xai.storage.enabled: false` or set `expires_after` to change the retention.':
          'grok-imagine-image — テキストから画像・画像編集。xAI Grok OAuth または XAI_API_KEY を使用。xAI Imagine ストレージが有効なため、生成メディアは自動失効しない再利用可能な公開 URL を取得します。保存ファイルと公開 URL ホスティングに xAI が課金する場合があります。`image_gen.xai.storage.enabled: false` で無効化、または `expires_after` で保持期間を変更できます。',
        'grok-imagine-video for text/reference; grok-imagine-video-1.5 for image-to-video; edit/extend: pass the stored public HTTPS MP4 (`video` / `public_url` from a prior Imagine result); uses xAI Grok OAuth or XAI_API_KEY. xAI Imagine storage is enabled so generated media gets a reusable public URL without an automatic expiry. xAI may bill for stored files and public URL hosting. Disable this with `video_gen.xai.storage.enabled: false` or set `expires_after` to change the retention.':
          'grok-imagine-video はテキスト/参照から生成、grok-imagine-video-1.5 は画像から動画。編集/延長は以前の Imagine 結果の公開 HTTPS MP4（`video` / `public_url`）を渡します。xAI Grok OAuth または XAI_API_KEY を使用。xAI Imagine ストレージが有効なため、生成メディアは自動失効しない再利用可能な公開 URL を取得します。保存ファイルと公開 URL ホスティングに xAI が課金する場合があります。`video_gen.xai.storage.enabled: false` で無効化、または `expires_after` で保持期間を変更できます。'
      },
      nousIncluded: 'Nous サブスクリプションに含まれています。有効にするには Nous Portal にサインインしてください。',
      nousAuthNeededTitle: 'Nous Portal にサインイン',
      nousAuthNeededMessage: provider =>
        `${provider} は保存されましたが、Nous Portal にサインインするまで有効になりません。`,
      nousAuthSignIn: 'サインイン',
      nousAuthDoneTitle: 'Nous Portal に接続しました',
      nousAuthDoneMessage: 'サブスクリプションのバックエンドが有効になりました。',
      nousAuthFailed: 'Nous Portal のサインインが完了しませんでした',
      noApiKeyRequired: 'API キーは不要です。',
      postSetupHint: step =>
        `このバックエンドは一度だけインストールが必要です (${step})。このマシン上で実行され、数分かかる場合があります。`,
      postSetupInstalledHint: 'インストール済みです。問題がある場合のみセットアップを再実行してください。',
      postSetupRun: 'セットアップを実行',
      postSetupRerun: 'セットアップを再実行',
      postSetupInstalled: 'インストール済み',
      postSetupRunning: 'インストール中…',
      postSetupStarting: '開始中…',
      postSetupCompleteTitle: 'セットアップ完了',
      postSetupCompleteMessage: step => `${step} をインストールしました。`,
      postSetupErrorTitle: 'セットアップはエラーで終了しました',
      postSetupErrorMessage: step => `${step} のログを確認してください。`,
      postSetupFailed: step => `${step} のセットアップの実行に失敗しました`,
      webSearchActive: backend => `検索: ${backend}`,
      webExtractActive: backend => `抽出: ${backend}`,
      webCapabilityUnset: '未設定',
      webUseForSearch: '検索に使用',
      webUseForExtract: '抽出に使用',
      webUsedForSearch: '検索バックエンド',
      webUsedForExtract: '抽出バックエンド',
      webCapabilitySelectedMessage: (provider, capability) =>
        `${provider} がウェブ${capability === 'search' ? '検索' : '抽出'}を担当します。`,
      failedSelectCapability: provider => `${provider} の設定に失敗しました`,
      terminalBackend: {
        sectionTitle: '実行バックエンド',
        loading: '実行バックエンドを確認中…',
        failedLoad: 'ターミナルバックエンドの読み込みに失敗しました',
        ready: '準備完了',
        needsSetup: 'セットアップが必要',
        unavailable: '利用不可',
        inUse: '使用中',
        selectedTitle: 'バックエンドを選択しました',
        selectedMessage: backend => `ターミナルコマンドは ${backend} で実行されます。新しいセッションに適用されます。`,
        failedSelect: backend => `${backend} の選択に失敗しました`,
        needsSetupHint: 'このバックエンドは今すぐ選択できますが、セットアップが完了するまでコマンドは失敗します。'
      },
      computerUse: {
        checking: 'コンピュータ操作のステータスを確認中…',
        statusReadFailed: 'コンピュータ操作のステータスを読み取れませんでした',
        unsupported: platform => `このプラットフォーム（${platform}）ではコンピュータ操作はサポートされていません。`,
        installHint: 'このマシンを操作するには、下の cua-driver バックエンドをインストールしてください。',
        installGrantHint: 'その後、ここでアクセシビリティと画面収録の権限を付与してください。',
        platformNotes: {
          linux: 'X11/XWayland アクセシビリティスタック経由でデスクトップを操作します——権限プロンプトはありません。',
          win32:
            '初回実行時に cua-driver の UIAccess ワーカーに対して Windows SmartScreen の警告が表示される場合があります——許可してください。'
        },
        macGrantNote:
          '権限は Hermes ではなく CuaDriver 自身の識別子（com.trycua.driver）に付与されます——ダイアログには実際に Mac を操作するプロセスが表示されます。',
        recheck: '再チェック',
        accessibility: 'アクセシビリティ',
        accessibilityHint: 'cua-driver によるクリック・キー入力の送信とアクセシビリティツリーの読み取りを許可します。',
        screenRecording: '画面収録',
        screenRecordingHint: 'cua-driver によるアプリウィンドウのスクリーンショット取得を許可します。',
        driverHealth: 'ドライバーの状態',
        granted: '許可済み',
        notGranted: '未許可',
        ready: '準備完了',
        notReady: '未準備',
        unknown: '不明',
        readyMessage: 'コンピュータ操作の準備ができました。エージェントにアプリのキャプチャやクリック操作を依頼できます。',
        grantPermissions: '権限を付与',
        waitingApproval: '承認を待機中…',
        grantFailed: '権限をリクエストできませんでした',
        approveTitle: 'システム設定で承認してください',
        approveMessage: 'macOS に CuaDriver に帰属する権限ダイアログが表示されます。承認後、ここに戻ってください。'
      }
    }
  },

  skills: {
    tabSkills: 'スキル',
    tabToolsets: 'ツールセット',
    tabMcp: 'MCP',
    all: 'すべて',
    searchSkills: 'スキルを検索...',
    searchToolsets: 'ツールセットを検索...',
    refresh: 'スキルを更新',
    refreshing: 'スキルを更新中',
    loading: '機能を読み込み中...',
    noSkillsTitle: 'スキルが見つかりません',
    noSkillsDesc: '検索を広げるか、別のカテゴリーを試してください。',
    noToolsetsTitle: 'ツールセットが見つかりません',
    noToolsetsDesc: '検索キーワードを広げてください。',
    noDescription: '説明はありません。',
    toolsetDescriptions: {
      browser: 'ウェブ操作のためのブラウザ自動化（ナビゲート、クリック、入力、スクロール、iframe、長押し）と URL 検索用のウェブ検索',
      clarify: 'ユーザーに確認の質問をします（選択式または自由回答）',
      code_execution: 'ツールをプログラム的に呼び出す Python スクリプトを実行します（LLM の往復を削減）',
      coding: 'コーディング向けツールセット: ファイル、ターミナル、検索、ウェブドキュメント、スキル、Todo、委任、ビジョン、ブラウザ',
      computer_use:
        'cua-driver によるバックグラウンドのデスクトップ操作（macOS/Windows/Linux）— スクリーンショット、マウス、キーボード、スクロール、ドラッグ。ユーザーのカーソルやキーボードフォーカスを奪いません。ツール対応のあらゆるモデルで動作します。',
      context_engine: 'アクティブなコンテキストエンジンが公開するランタイムツール',
      cronjob: 'Cron ジョブ管理ツール — スケジュールタスクの作成、一覧、更新、一時停止、再開、削除、実行',
      debugging: 'デバッグとトラブルシューティングのツールキット',
      delegation: '複雑なサブタスクのために隔離コンテキストのサブエージェントを生成します',
      discord: 'Discord の閲覧・参加ツール（メッセージ取得、メンバー検索、スレッド作成）',
      discord_admin: 'Discord サーバー管理（チャンネル/ロール一覧、メッセージのピン留め、ロール割り当て）',
      feishu_doc: 'Feishu / Lark ドキュメントの内容を読み取ります',
      feishu_drive: 'Feishu / Lark ドキュメントのコメント操作（一覧、返信、追加）',
      file: 'ファイル操作ツール: 読み取り、書き込み、パッチ（あいまい一致対応）、検索（内容 + ファイル）',
      'hermes-acp': 'エディター統合（VS Code、Zed、JetBrains）— メッセージング・音声・確認 UI を除いたコーディング向けツール',
      'hermes-api-server':
        'OpenAI 互換 API サーバー — HTTP 経由で全エージェントツールにアクセス（clarify や send_message などの対話型 UI ツールは除く）',
      'hermes-bluebubbles': 'BlueBubbles iMessage ボットツールセット — ローカルの BlueBubbles サーバー経由の Apple iMessage',
      'hermes-cli': '完全な対話型 CLI ツールセット — すべてのデフォルトツールと Cron ジョブ管理',
      'hermes-cron': 'デフォルトの Cron ツールセット — hermes-cli と同じコアツール。hermes tools で制御',
      'hermes-dingtalk': 'DingTalk ボットツールセット — エンタープライズメッセージングプラットフォーム（フルアクセス）',
      'hermes-discord': 'Discord ボットツールセット — フルアクセス（ターミナルは危険コマンド承認の安全チェック付き）',
      'hermes-email': 'メールボットツールセット — メール (IMAP/SMTP) で Hermes と対話',
      'hermes-feishu': 'Feishu / Lark ボットツールセット — Feishu / Lark 経由のエンタープライズメッセージング（フルアクセス）',
      'hermes-gateway': 'ゲートウェイツールセット — すべてのメッセージングプラットフォームツールの統合',
      'hermes-homeassistant': 'Home Assistant ボットツールセット — スマートホームのイベント監視と制御',
      'hermes-matrix': 'Matrix ボットツールセット — 分散型暗号化メッセージング（フルアクセス）',
      'hermes-mattermost': 'Mattermost ボットツールセット — セルフホストのチームメッセージング（フルアクセス）',
      'hermes-qqbot': 'QQBot ツールセット — 公式 Bot API v2 経由の QQ メッセージング（フルアクセス）',
      'hermes-signal': 'Signal ボットツールセット — 暗号化メッセージングプラットフォーム（フルアクセス）',
      'hermes-slack': 'Slack ボットツールセット — ワークスペース利用のフルアクセス（ターミナルは安全チェック付き）',
      'hermes-sms': 'SMS ボットツールセット — SMS (Twilio) で Hermes と対話',
      'hermes-telegram': 'Telegram ボットツールセット — 個人利用のフルアクセス（ターミナルは安全チェック付き）',
      'hermes-webhook': 'Webhook ツールセット — 外部 Webhook イベントの受信と処理',
      'hermes-wecom': 'WeCom ボットツールセット — エンタープライズ WeChat メッセージング（フルアクセス）',
      'hermes-wecom-callback': 'WeCom コールバックツールセット — 企業の自社構築アプリメッセージング（フルアクセス）',
      'hermes-weixin': 'Weixin ボットツールセット — iLink 経由の個人 WeChat メッセージング（フルアクセス）',
      'hermes-whatsapp': 'WhatsApp ボットツールセット — Telegram に類似（個人メッセージング、より信頼度が高い）',
      'hermes-yuanbao': 'Yuanbao メッセージングプラットフォームツールセット — グループ情報、メンバー照会、DM、スタンプ',
      homeassistant: 'Home Assistant スマートホームの制御と監視',
      image_gen: 'クリエイティブ生成ツール（画像）',
      kanban:
        'カンバンのマルチエージェント連携 — カンバンディスパッチャーから生成されたエージェント（HERMES_KANBAN_TASK 環境変数設定時）のみ有効。ディスパッチャーはデフォルトでゲートウェイ内で実行されます（config.yaml の kanban.dispatch_in_gateway 参照）。ワーカーは構造化された引き継ぎでタスクを完了し、人間の入力待ちでブロックし、長い操作中にハートビートを送り、スレッドにコメントし、ファイルを添付できます（オーケストレーターはさらにタスクの一覧・ブロック解除・分配が可能）。',
      memory: 'セッションをまたぐ永続メモリ（個人メモ + ユーザープロファイル）',
      project: 'デスクトッププロジェクト — 名前付きワークスペースの作成/切り替え（GUI セッションのみ）',
      safe: 'ターミナルアクセスを除いた安全なツールキット',
      search: 'ウェブ検索のみ（コンテンツ抽出/スクレイピングなし）',
      session_search: '過去の会話を検索して要約付きで振り返ります',
      skills: '専門的な指示と知識を持つスキルドキュメントへのアクセス、作成、編集、管理',
      spotify: 'ネイティブの Spotify 再生、検索、プレイリスト、アルバム、ライブラリツール',
      terminal: 'ターミナル/コマンド実行とプロセス管理ツール',
      todo: '複数ステップの作業のためのタスク計画と追跡',
      tts: 'テキスト読み上げ: Edge TTS（無料）、ElevenLabs、OpenAI、xAI でテキストを音声に変換',
      video: '動画の分析・理解ツール（オプトイン、デフォルトツールセット外）',
      video_gen:
        '動画生成ツール。単一の video_generate ツールがテキストから動画（プロンプトのみ）と画像から動画（プロンプト + image_url）、参照から動画をカバーします。プロバイダー固有の編集/延長ワークフローは別ツールとして現れる場合があります。hermes tools → Video Generation で設定。',
      vision: '画像分析とビジョンツール',
      x_search:
        'xAI 内蔵の x_search Responses ツールで X (Twitter) の投稿とスレッドを検索します。xAI 資格情報（SuperGrok OAuth または XAI_API_KEY）の設定時に利用可能。デフォルトはオフ。hermes tools → X (Twitter) Search で有効化。',
      yuanbao: 'Yuanbao プラットフォームツール — グループ情報、メンバー照会、DM、スタンプ'
    },
    toolsetLabels: {
      web: 'ウェブ検索とスクレイピング',
      browser: 'ブラウザ自動化',
      terminal: 'ターミナルとプロセス',
      file: 'ファイル操作',
      code_execution: 'コード実行',
      vision: 'ビジョン / 画像分析',
      video: '動画分析',
      image_gen: '画像生成',
      video_gen: '動画生成',
      x_search: 'X (Twitter) 検索',
      tts: 'テキスト読み上げ',
      skills: 'スキル',
      todo: 'タスク計画',
      memory: 'メモリ',
      context_engine: 'コンテキストエンジン',
      session_search: 'セッション検索',
      clarify: '確認の質問',
      delegation: 'タスク委任',
      cronjob: 'Cron ジョブ',
      discord: 'Discord（閲覧/参加）',
      discord_admin: 'Discord サーバー管理',
      yuanbao: 'Yuanbao（元宝）',
      computer_use: 'コンピューター操作 (macOS/Windows/Linux)'
    },
    configured: '設定済み',
    needsKeys: 'キーが必要',
    visionModelHint:
      'ビジョンは補助モデル設定を使用します。画像対応モデルはそこで選択され、ここでプロバイダーごとに選ぶものではありません。',
    visionModelLink: '設定 → モデル でビジョンモデルを選択',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} ツールセットが有効`,
    configureToolset: label => `${label} を設定`,
    toggleToolset: label => `${label} ツールセットを切り替え`,
    skillsLoadFailed: 'スキルの読み込みに失敗しました',
    toolsetsRefreshFailed: 'ツールセットの更新に失敗しました',
    skillEnabled: 'スキルを有効にしました',
    skillDisabled: 'スキルを無効にしました',
    toolsetEnabled: 'ツールセットを有効にしました',
    toolsetDisabled: 'ツールセットを無効にしました',
    appliesToNewSessions: name => `${name} は新しいセッションに適用されます。`,
    failedToUpdate: name => `${name} の更新に失敗しました`,
    sortMostUsed: '使用頻度順',
    sortAlpha: 'A–Z',
    sortMostUsedDesc: '↓ 使用頻度順',
    sortLeastUsedAsc: '↑ 使用頻度が低い順',
    enableAll: 'すべて有効化',
    disableAll: 'すべて無効化',
    disableUnused: '未使用を無効化',
    bulkUpdated: count => `${count} 件を新しいセッション向けに更新しました。`,
    bulkNoChange: '変更するものはありません。',
    usageCount: count => `${count} 回使用`,
    provenance: {
      agent: '学習済み',
      bundled: '組み込み',
      hub: 'ハブ'
    },
    emptyNoneFound: noun => `${noun} が見つかりません`,
    emptyNothingMatches: query => `「${query}」に一致するものはありません。`,
    emptyNoneAvailable: noun => `利用可能な ${noun} はまだありません。`,
    changesApplyNewSessions: '変更は新しいセッションに適用されます。',
    skillUpdated: 'スキルを更新しました',
    edit: '編集',
    archive: 'アーカイブ',
    skillArchivedTitle: 'スキルをアーカイブしました',
    skillArchivedMessage: 'hermes curator restore で復元できます。'
  },

  starmap: {
    title: 'メモリグラフ',
    subtitle: (nodes, clusters) => `${clusters} カテゴリの ${nodes} スキル`,
    close: 'メモリグラフを閉じる',
    refresh: '更新',
    memory: 'メモリ',
    filterAll: 'すべて',
    filterUsed: '使用済み',
    filterLearned: '学習済み',
    viewGraph: 'グラフ',
    loadFailed: 'メモリグラフを読み込めませんでした',
    loading: '読み込み中…',
    emptyTitle: 'まだ学習はありません',
    emptyDesc: 'Hermes がスキルやメモリを蓄積すると、ここに表示されます。'
  },
  agents: {
    close: 'エージェントを閉じる',
    title: 'スポーンツリー',
    subtitle: '現在のターンのライブサブエージェントのアクティビティ。',
    emptyTitle: 'ライブサブエージェントはありません',
    emptyDesc: 'ターンで作業を委任すると、子エージェントの進捗状況がここにストリームされます。',
    running: '実行中',
    failed: '失敗',
    done: '完了',
    streaming: 'ストリーミング中',
    files: 'ファイル',
    moreFiles: count => `+${count} 件のファイル`,
    delegation: index => `委任 ${index}`,
    workers: count => `${count} ワーカー`,
    workersActive: count => `${count} アクティブ`,
    agentsCount: count => `${count} エージェント`,
    activeCount: count => `${count} アクティブ`,
    failedCount: count => `${count} 失敗`,
    toolsCount: count => `${count} ツール`,
    filesCount: count => `${count} ファイル`,
    updatedAgo: age => `${age} に更新`,
    ageNow: 'たった今',
    ageSeconds: seconds => `${seconds}秒前`,
    ageMinutes: minutes => `${minutes}分前`,
    ageHours: hours => `${hours}時間前`,
    ageDays: days => `${days}日前`,
    durationSeconds: seconds => `${seconds}秒`,
    durationMinutes: (minutes, seconds) => `${minutes}分 ${seconds}秒`,
    tokens: value => `${value} トーク`
  },

  commandCenter: {
    close: 'コマンドセンターを閉じる',
    paletteTitle: 'コマンドパレット',
    back: '戻る',
    searchPlaceholder: 'セッション、ビュー、アクションを検索',
    goTo: '移動',
    goToSession: 'セッションへ移動',
    branches: 'ブランチ',
    startInBranch: branch => `${branch} で新しい会話`,
    commandCenter: 'コマンドセンター',
    appearance: '外観',
    settings: '設定',
    changeTheme: 'テーマを変更',
    changeColorMode: 'カラーモードを変更…',
    pets: {
      title: 'ペット',
      placeholder: 'ペットを検索…',
      loading: 'petdex ギャラリーを読み込み中…',
      error: 'petdex ギャラリーに接続できません。',
      staleBackend: 'ペット機能を使うには Hermes を再起動してください。',
      empty: '一致するペットがありません。',
      turnOff: 'オフ',
      turnOn: 'オン',
      installed: 'インストール済み',
      generatedTag: '生成',
      adoptFailed: 'ペットを採用できませんでした。',
      toggleFailed: 'ペットを切り替えできませんでした。',
      noneAvailable: '利用可能なペットがありません。'
    },
    generatePet: {
      title: 'ペットを生成',
      placeholder: '生成するペットを説明…',
      promptHint: '説明を入力して Enter を押すと、4 つの見た目を生成します。',
      readyHint: 'Enter を押すと、説明から 4 つの見た目を生成します。',
      generate: '生成',
      generating: '生成中…',
      retry: '再試行',
      hatch: '孵化',
      spawning: 'スポーン中…',
      hatching: 'ペットを孵化しています…',
      hatchingSub: '命を吹き込んでいます…',
      hatched: '孵化しました！',
      hatchRow: (_state, done, total) => `フレームを描画中… ${done}/${total}`,
      hatchComposing: 'まとめています…',
      hatchSaving: 'もうすぐです…',
      namePlaceholder: 'ペットに名前を付ける',
      staleBackend: 'ペットを生成するには Hermes を更新してください。',
      backgroundHint: 'このウィンドウは閉じても大丈夫です。完了したら Hermes が通知します。',
      slowProviderHint: '数分かかることがあります',
      remix: 'リミックス',
      remixConfirmTitle: 'この見た目でリミックスしますか？',
      remixConfirmBody: 'これを起点に新しい候補を生成します。数分かかることがあります。',
      genericError: '生成に失敗しました。もう一度試すか、候補を選んでください。',
      referenceImageTooLarge: '参照画像が大きすぎます。16 MB 未満の画像を使ってください。',
      referenceImageInvalid: '参照画像を読み込めませんでした。PNG/JPG/WebP/GIF を試してください。',
      adopt: '迎え入れる',
      startOver: 'やり直す'
    },
    installTheme: {
      title: 'テーマをインストール…',
      pageTitle: 'テーマをインストール',
      placeholder: 'VS Code Marketplace を検索...',
      loading: 'Marketplace を検索中...',
      error: 'Marketplace に接続できませんでした。',
      empty: '一致するテーマがありません。',
      install: 'インストール',
      installing: 'インストール中...',
      installed: 'インストール済み',
      installs: count => `${count} 回インストール`
    },
    settingsFields: '設定フィールド',
    mcpServers: 'MCP サーバー',
    archivedChats: 'アーカイブ済みチャット',
    sections: { sessions: 'セッション', system: 'システム', usage: '使用状況' },
    sectionDescriptions: {
      sessions: 'セッションの検索と管理',
      system: 'ステータス、ログ、システムアクション',
      usage: 'トークン、コスト、スキルの活動履歴'
    },
    nav: {
      newChat: { title: '新しいセッション', detail: '新しいセッションを開始' },
      settings: { title: '設定', detail: 'Hermes デスクトップを設定' },
      skills: { title: 'スキルとツール', detail: 'スキル、ツールセット、プロバイダーを有効化' },
      messaging: { title: 'メッセージング', detail: 'Telegram、Slack、Discord などを設定' },
      artifacts: { title: 'アーティファクト', detail: '生成された出力を閲覧' }
    },
    sectionEntries: {
      sessions: { title: 'セッションパネル', detail: 'セッションの検索、ピン留め、管理' },
      system: { title: 'システムパネル', detail: 'ゲートウェイのステータス、ログ、再起動/更新' },
      usage: { title: '使用状況パネル', detail: 'トークン、コスト、スキルの活動' }
    },
    providerNavigate: 'ナビゲート',
    providerSessions: 'セッション',
    refresh: '更新',
    refreshing: '更新中...',
    noResults: '一致する結果が見つかりません。',
    pinSession: 'セッションをピン留め',
    unpinSession: 'セッションのピン留めを解除',
    exportSession: 'セッションをエクスポート',
    deleteSession: 'セッションを削除',
    noSessions: 'セッションはまだありません。',
    gatewayRunning: 'メッセージングゲートウェイが実行中',
    gatewayStopped: 'メッセージングゲートウェイが停止中',
    hermesActiveSessions: (version, count) => `Hermes ${version} · アクティブセッション ${count}`,
    restartGateway: 'ゲートウェイを再起動',
    gatewayRestartFailed: 'ゲートウェイの再起動に失敗しました。',
    updateHermes: 'Hermes を更新',
    actionRunning: '実行中',
    actionDone: '完了',
    actionFailed: '失敗',
    actionStartedWaiting: 'アクションが開始されました。ステータスを待機中...',
    loadingStatus: 'ステータスを読み込み中...',
    recentLogs: '最近のログ',
    noLogs: 'ログはまだ読み込まれていません。',
    days: count => `${count}日`,
    statSessions: 'セッション',
    statApiCalls: 'API コール',
    statTokens: 'トークン入力/出力',
    statCost: '推定コスト',
    actualCost: cost => `実際 ${cost}`,
    loadingUsage: '使用状況を読み込み中...',
    noUsage: period => `過去 ${period} 日間に使用履歴がありません。`,
    retry: '再試行',
    dailyTokens: '日別トークン',
    input: '入力',
    output: '出力',
    noDailyActivity: '日別アクティビティがありません。',
    topModels: 'よく使うモデル',
    noModelUsage: 'モデルの使用履歴はまだありません。',
    topSkills: 'よく使うスキル',
    noSkillActivity: 'スキルのアクティビティはまだありません。',
    actions: count => `${count} アクション`
  },

  messaging: {
    search: 'メッセージングを検索...',
    loading: 'メッセージングプラットフォームを読み込み中...',
    loadFailed: 'メッセージングプラットフォームの読み込みに失敗しました',
    states: {
      connected: '接続済み',
      connecting: '接続中',
      disabled: '無効',
      fatal: 'エラー',
      gateway_stopped: 'メッセージングゲートウェイが停止中',
      not_configured: '設定が必要',
      pending_restart: '再起動が必要',
      retrying: '再試行中',
      startup_failed: '起動失敗'
    },
    unknown: '不明',
    hintPendingRestart: 'この変更を適用するにはステータスバーからゲートウェイを再起動してください。',
    hintGatewayStopped: 'ステータスバーからゲートウェイを起動して接続してください。',
    credentialsSet: '認証情報を設定しました',
    needsSetup: '設定が必要',
    gatewayStopped: 'メッセージングゲートウェイが停止中',
    getCredentials: '認証情報を取得',
    openSetupGuide: 'セットアップガイドを開く',
    required: '必須',
    recommended: '推奨',
    advanced: count => `詳細設定 (${count})`,
    noTokenNeeded:
      'このプラットフォームはここでトークンが必要ありません。上のセットアップガイドを使用してから、以下で有効にしてください。',
    enabled: '有効',
    disabled: '無効',
    unsavedChanges: '未保存の変更',
    saving: '保存中...',
    saveChanges: '変更を保存',
    saved: '保存しました',
    replaceValue: '現在の値を置き換え',
    openDocs: 'ドキュメントを開く',
    clearField: key => `${key} をクリア`,
    enableAria: name => `${name} を有効にする`,
    disableAria: name => `${name} を無効にする`,
    platformEnabled: name => `${name} を有効にしました`,
    platformDisabled: name => `${name} を無効にしました`,
    restartToApply: 'この変更はゲートウェイの再起動後に有効になります。',
    setupSaved: name => `${name} の設定を保存しました`,
    restartToReconnect: '新しい認証情報はゲートウェイの再起動後に有効になります。',
    keyCleared: key => `${key} をクリアしました`,
    setupUpdated: name => `${name} の設定が更新されました。`,
    failedUpdate: name => `${name} の更新に失敗しました`,
    failedSave: name => `${name} の保存に失敗しました`,
    failedClear: key => `${key} のクリアに失敗しました`,
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'ボットトークン',
        help: '@BotFather でボットを作成し、表示されたトークンを貼り付けてください。',
        placeholder: 'Telegram ボットトークンを貼り付け'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: '許可する Telegram ユーザー ID',
        help: '推奨。@userinfobot の数値 ID をカンマ区切りで。設定しないと誰でもボットに DM できます。'
      },
      TELEGRAM_PROXY: { label: 'プロキシ URL', help: 'Telegram がブロックされているネットワークでのみ必要です。' },
      DISCORD_BOT_TOKEN: {
        label: 'ボットトークン',
        help: 'Discord Developer Portal でアプリケーションを作成し、ボットを追加してからトークンを貼り付けてください。'
      },
      DISCORD_ALLOWED_USERS: {
        label: '許可する Discord ユーザー ID',
        help: '推奨。カンマ区切りの Discord ユーザー ID。'
      },
      DISCORD_REPLY_TO_MODE: { label: '返信スタイル', help: 'first、all、または off。' },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'すべての Discord ユーザーを許可',
        help: '開発用のみ。true にすると、許可リストなしで誰でもボットに DM できます。'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'ホームチャンネル ID',
        help: 'ボットがプロアクティブなメッセージを送信するチャンネル（Cron 出力、リマインダー）。'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'ホームチャンネル名',
        help: 'ログやステータス出力でのホームチャンネルの表示名。'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'すべての iMessage ユーザーを許可',
        help: 'true にすると BlueBubbles の許可リストをスキップします。'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'すべての Mattermost ユーザーを許可' },
      MATTERMOST_HOME_CHANNEL: { label: 'ホームチャンネル' },
      QQ_ALLOW_ALL_USERS: { label: 'すべての QQ ユーザーを許可' },
      QQBOT_HOME_CHANNEL: { label: 'QQ ホームチャンネル', help: 'Cron 配信のデフォルトチャンネルまたはグループ。' },
      QQBOT_HOME_CHANNEL_NAME: { label: 'QQ ホームチャンネル名' },
      SLACK_BOT_TOKEN: {
        label: 'Slack ボットトークン',
        help: 'Slack アプリをインストール後、OAuth & Permissions のボットトークンを使用してください。',
        placeholder: 'Slack ボットトークンを貼り付け'
      },
      SLACK_APP_TOKEN: {
        label: 'Slack アプリトークン',
        help: 'Socket Mode に必要なアプリレベルのトークンを使用してください。',
        placeholder: 'Slack アプリトークンを貼り付け'
      },
      SLACK_ALLOWED_USERS: {
        label: '許可する Slack ユーザー ID',
        help: '推奨。カンマ区切りの Slack ユーザー ID。'
      },
      MATTERMOST_URL: { label: 'サーバー URL', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: 'ボットトークン' },
      MATTERMOST_ALLOWED_USERS: {
        label: '許可するユーザー ID',
        help: '推奨。カンマ区切りの Mattermost ユーザー ID。'
      },
      MATRIX_HOMESERVER: { label: 'ホームサーバー URL', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: 'アクセストークン' },
      MATRIX_USER_ID: { label: 'ボットユーザー ID', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: '許可する Matrix ユーザー ID',
        help: '推奨。@user:server 形式のカンマ区切りユーザー ID。'
      },
      SIGNAL_HTTP_URL: {
        label: 'Signal ブリッジ URL',
        placeholder: 'http://127.0.0.1:8080',
        help: '実行中の signal-cli REST ブリッジの URL。'
      },
      SIGNAL_ACCOUNT: { label: '電話番号', help: 'signal-cli ブリッジに登録した番号。' },
      SIGNAL_ALLOWED_USERS: {
        label: '許可する Signal ユーザー',
        help: '推奨。カンマ区切りの Signal 識別子。'
      },
      WHATSAPP_ENABLED: {
        label: 'WhatsApp ブリッジを有効にする',
        help: '以下のトグルで自動的に設定されます。必要な場合を除いてそのままにしてください。'
      },
      WHATSAPP_MODE: { label: 'ブリッジモード' },
      WHATSAPP_ALLOWED_USERS: {
        label: '許可する WhatsApp ユーザー',
        help: '推奨。カンマ区切りの電話番号または WhatsApp ID。'
      },
      IRC_SERVER: { label: 'IRC サーバー', help: 'IRC サーバーのホスト名（例: irc.libera.chat）。', placeholder: 'irc.libera.chat' },
      IRC_CHANNEL: { label: 'IRC チャンネル', help: '参加する IRC チャンネル（例: #hermes）。' },
      IRC_NICKNAME: { label: 'ボットのニックネーム', help: 'IRC 上のボットのニックネーム（デフォルト: hermes-bot）。' },
      IRC_SERVER_PASSWORD: { label: 'サーバーパスワード', help: 'IRC サーバーのパスワード（必要な場合）。' },
      IRC_NICKSERV_PASSWORD: { label: 'NickServ パスワード', help: 'ニックネーム認証用の NickServ パスワード。' },
      IRC_PORT: { label: 'IRC ポート', help: 'IRC サーバーのポート（デフォルト: TLS は 6697、非 TLS は 6667）。' },
      IRC_USE_TLS: { label: 'TLS を使用', help: 'IRC 接続に TLS を使用（1/true/yes で有効。ポート 6697 ではデフォルト有効）。' },
      IRC_ALLOWED_USERS: { label: '許可するニックネーム', help: 'ボットと会話できる IRC ニックネーム。カンマ区切り。' },
      IRC_ALLOW_ALL_USERS: { label: 'すべてのユーザーを許可', help: '開発用のみ。チャンネル内の誰でもボットと会話できます。' },
      IRC_HOME_CHANNEL: { label: 'ホームチャンネル', help: 'Cron / 通知配信のチャンネル（デフォルトは IRC_CHANNEL）。' },
      GOOGLE_CHAT_SERVICE_ACCOUNT_JSON: {
        label: 'サービスアカウント JSON',
        help: 'サービスアカウント JSON キーのパス（またはインライン JSON）。空欄なら Cloud Run / GCE のアプリケーションデフォルト認証情報 (ADC) を使用し、GOOGLE_APPLICATION_CREDENTIALS にフォールバックします。'
      },
      GOOGLE_CHAT_HTTP_EVENTS_URL: { label: 'HTTP イベントコールバック URL', help: 'Chat メッセージイベント用の認証済み HTTP エンドポイント。' },
      GOOGLE_CHAT_HTTP_EVENTS_AUDIENCE: {
        label: 'HTTP イベントトークンのオーディエンス',
        help: 'Google 署名の HTTP イベント Bearer トークンに期待するオーディエンス。デフォルトは GOOGLE_CHAT_HTTP_EVENTS_URL。'
      },
      GOOGLE_CHAT_HTTP_EVENTS_SERVICE_ACCOUNT_EMAIL: {
        label: 'HTTP イベントサービスアカウントメール',
        help: 'HTTP イベント Bearer トークンに期待する Google サービスアカウントのメールアドレス。'
      },
      GOOGLE_CHAT_PROJECT_ID: {
        label: 'GCP プロジェクト ID',
        help: '任意の Pub/Sub 受信モード用 GCP プロジェクト ID。GOOGLE_CLOUD_PROJECT にフォールバック。'
      },
      GOOGLE_CHAT_SUBSCRIPTION_NAME: { label: 'Pub/Sub サブスクリプション名', help: 'プルモード受信イベント用の任意の Pub/Sub サブスクリプションパス。' },
      GOOGLE_CHAT_ALLOWED_USERS: { label: '許可するユーザーメール', help: 'ボットと対話できるユーザーのメールアドレス。カンマ区切り。' },
      GOOGLE_CHAT_HOME_CHANNEL: { label: 'ホームスペース ID', help: 'Cron / 通知配信のデフォルトスペース（例: spaces/AAAA...）。' },
      LINE_CHANNEL_ACCESS_TOKEN: {
        label: 'チャネルアクセストークン',
        help: 'LINE チャネルの長期アクセストークン（LINE Developers コンソール > Messaging API > チャネルアクセストークン）。'
      },
      LINE_CHANNEL_SECRET: { label: 'チャネルシークレット', help: 'LINE チャネルシークレット（HMAC-SHA256 Webhook 署名検証に使用）。' },
      LINE_PORT: { label: 'Webhook ポート', help: 'Webhook のリッスンポート（デフォルト: 8646）。' },
      LINE_HOST: { label: 'Webhook ホスト', help: 'Webhook のバインドホスト（デフォルト: 0.0.0.0）。' },
      LINE_PUBLIC_URL: {
        label: '公開 HTTPS ベース URL',
        help: 'LINE へ画像/音声/動画を配信するための公開 HTTPS ベース URL（例: https://my-tunnel.example.com）。バインドアドレスに直接到達できない場合、メディア送信に必須。'
      },
      LINE_ALLOWED_USERS: { label: '許可するユーザー ID', help: 'ボットに DM できる LINE ユーザー ID（U で始まる）。カンマ区切り。' },
      LINE_ALLOWED_GROUPS: { label: '許可するグループ ID', help: 'ボットが応答する LINE グループ ID（C で始まる）。カンマ区切り。' },
      LINE_ALLOWED_ROOMS: { label: '許可するルーム ID', help: 'ボットが応答する LINE ルーム ID（R で始まる）。カンマ区切り。' },
      LINE_ALLOW_ALL_USERS: {
        label: 'すべてのユーザーを許可',
        help: '開発用のみ。すべての LINE ユーザーがボットと会話できます（許可リストを無効化）。'
      },
      LINE_HOME_CHANNEL: { label: 'ホームチャンネル ID', help: 'Cron / 通知配信のデフォルトのユーザー/グループ/ルーム ID。' },
      LINE_SLOW_RESPONSE_THRESHOLD: {
        label: '低速応答しきい値（秒）',
        help: '低速 LLM ポストバックボタンが作動するまでの秒数（デフォルト: 45。0 で無効化し常に Push フォールバック）。'
      },
      NTFY_TOPIC: { label: '購読トピック', help: '購読するトピック名（例: hermes-in）。' },
      NTFY_SERVER_URL: { label: 'サーバー URL', help: 'ntfy サーバーの URL（デフォルト: https://ntfy.sh）。' },
      NTFY_TOKEN: { label: '認証トークン', help: 'Bearer トークンまたは Basic 認証用の user:pass（任意）。' },
      NTFY_PUBLISH_TOPIC: { label: '発行トピック', help: '返信を発行するトピック（デフォルトは NTFY_TOPIC）。' },
      NTFY_MARKDOWN: { label: 'Markdown を有効化', help: 'X-Markdown: true ヘッダー付きで返信を送信（true/false、デフォルト: false）。' },
      NTFY_ALLOWED_USERS: { label: '許可するトピック名', help: '許可するトピック名（許可リスト）。カンマ区切り。' },
      NTFY_ALLOW_ALL_USERS: {
        label: 'すべてのトピックを許可',
        help: '開発用のみ。あらゆるトピックがボットと会話できます（許可リストを無効化）。'
      },
      NTFY_HOME_CHANNEL: { label: 'ホームトピック', help: 'Cron / 通知配信のデフォルトトピック。' },
      NTFY_HOME_CHANNEL_NAME: { label: 'ホームトピック名', help: 'ホームチャンネルの表示名（デフォルトはトピック名）。' },
      PHOTON_PROJECT_ID: {
        label: 'Spectrum プロジェクト ID',
        help: 'Spectrum プロジェクト ID（プロジェクトの spectrumProjectId。hermes photon setup で設定）。'
      },
      PHOTON_PROJECT_SECRET: { label: 'プロジェクトシークレット', help: 'Spectrum プロジェクト ID と対になるシークレット（hermes photon setup で設定）。' },
      PHOTON_SIDECAR_PORT: { label: 'サイドカー制御ポート', help: 'Node サイドカーの制御 + 受信チャネル用ループバックポート（デフォルト 8789）。' },
      PHOTON_SIDECAR_AUTOSTART: { label: 'サイドカーを自動起動', help: '接続時に Node サイドカーを起動（true/false、デフォルト true）。' },
      PHOTON_NODE_BIN: { label: 'Node 実行ファイルのパス', help: 'node バイナリのパス（デフォルト: PATH 上の node）。' },
      PHOTON_DASHBOARD_HOST: { label: 'Dashboard ホスト', help: 'Photon Dashboard API ホスト（デフォルト https://app.photon.codes）。' },
      PHOTON_SPECTRUM_HOST: { label: 'Spectrum API ホスト', help: 'Photon Spectrum API ホスト（デフォルト https://spectrum.photon.codes）。' },
      PHOTON_ALLOWED_USERS: { label: '許可するユーザー', help: 'ボットと会話できる E.164 電話番号。カンマ区切り。' },
      PHOTON_ALLOW_ALL_USERS: {
        label: 'すべてのユーザーを許可',
        help: '開発用のみ。あらゆる送信者がボットをトリガーできます（許可リストを無効化）。'
      },
      PHOTON_REQUIRE_MENTION: {
        label: 'グループチャットでメンションを必須にする',
        help: 'メンションのウェイクワードに一致しない限りグループチャットのメッセージを無視します（true/false、デフォルト false）。'
      },
      PHOTON_MENTION_PATTERNS: {
        label: 'グループメンションパターン',
        help: 'グループチャット用メンションウェイクワードの正規表現（JSON リストまたはカンマ/改行区切り。デフォルトは Hermes のウェイクワード）。'
      },
      PHOTON_HOME_CHANNEL: {
        label: 'ホーム Photon ターゲット',
        help: 'Cron / 通知配信のデフォルト Photon ターゲット: Spectrum スペース ID、DM GUID、または素の E.164 電話番号。'
      },
      PHOTON_HOME_CHANNEL_NAME: { label: 'ホームチャンネル名', help: 'ホームチャンネルの表示名。' },
      PHOTON_TELEMETRY: {
        label: 'Spectrum テレメトリを有効化',
        help: 'サイドカーで Spectrum SDK テレメトリを有効にします（true/false、デフォルト false。hermes photon telemetry on|off で切り替え）。'
      },
      PHOTON_MARKDOWN: {
        label: '返信を Markdown でレンダリング',
        help: '返信を Markdown で送信します — iMessage はネイティブ表示、他の Spectrum プラットフォームはプレーンテキストに劣化（true/false、デフォルト true）。'
      },
      PHOTON_REACTIONS: {
        label: 'リアクションタップバックを有効化',
        help: '処理状況として 👀/👍/👎 をタップバックし、ボットメッセージへのタップバックをエージェントに転送します（true/false、デフォルト false）。'
      },
      SIMPLEX_WS_URL: { label: 'デーモン WebSocket URL', help: 'simplex-chat デーモンの WebSocket URL（例: ws://127.0.0.1:5225）。' },
      SIMPLEX_ALLOWED_USERS: { label: '許可する連絡先 ID', help: 'ボットと会話できる SimpleX 連絡先 ID。カンマ区切り。' },
      SIMPLEX_ALLOW_ALL_USERS: {
        label: 'すべての連絡先を許可',
        help: '開発用のみ。あらゆる連絡先がボットと会話できます（許可リストを無効化）。'
      },
      SIMPLEX_AUTO_ACCEPT: { label: '連絡先リクエストを自動承認', help: '受信した連絡先リクエストを自動承認します（デフォルト: true）。' },
      SIMPLEX_GROUP_ALLOWED: {
        label: '許可するグループ ID',
        help: 'ボットが参加する SimpleX グループ ID（カンマ区切り）、または * で任意のグループを許可。省略するとグループメッセージを完全に無視します（より安全なデフォルト — さもないとグループ内のボットは全メンバーのトラフィックを処理します）。'
      },
      SIMPLEX_HOME_CHANNEL: { label: 'ホーム連絡先/グループ ID', help: 'Cron / 通知配信のデフォルト連絡先/グループ ID。' },
      SIMPLEX_HOME_CHANNEL_NAME: { label: 'ホームチャンネル名', help: 'ホームチャンネルの表示名（デフォルトは ID）。' },
      HERMES_SIMPLEX_TEXT_BATCH_DELAY: {
        label: 'テキストバッチ遅延（秒）',
        help: '連続して届く受信テキストを 1 つのメッセージイベントに結合する静穏期間の秒数（デフォルト: 0.8）— Telegram のテキストバッチングと同じパターン。'
      },
      SMS_ALLOWED_USERS: { label: '許可する番号', help: 'ボットと会話できる電話番号。カンマ区切り。' },
      SMS_HOME_CHANNEL: { label: 'ホーム番号', help: 'Cron / 通知配信のデフォルト電話番号。' },
      TEAMS_CLIENT_ID: { label: 'Azure AD クライアント ID', help: 'Azure AD アプリケーション（Bot Framework）のクライアント ID。' },
      TEAMS_CLIENT_SECRET: { label: 'Azure AD クライアントシークレット', help: 'Azure AD アプリケーションのクライアントシークレット。' },
      TEAMS_TENANT_ID: { label: 'Azure AD テナント ID', help: 'ボットアプリケーションをホストする Azure AD テナント ID。' },
      TEAMS_PORT: { label: 'Webhook ポート', help: 'Webhook のリッスンポート（Bot Framework デフォルト: 3978）。' },
      TEAMS_ALLOWED_USERS: { label: '許可するユーザー', help: 'ボットと会話できる Teams ユーザー ID / UPN。カンマ区切り。' },
      TEAMS_ALLOW_ALL_USERS: { label: 'すべてのユーザーを許可', help: '開発用のみ。すべての Teams ユーザーがボットをトリガーできます。' },
      TEAMS_HOME_CHANNEL: { label: 'ホームチャンネル', help: 'Cron / 通知配信のデフォルトのチャット/チャンネル ID。' },
      TEAMS_HOME_CHANNEL_NAME: { label: 'ホームチャンネル名', help: 'Teams ホームチャンネルの表示名。' },
      WECOM_WEBSOCKET_URL: { label: 'WebSocket URL', help: 'WeCom スマートロボットの WebSocket URL。' },
      WECOM_HOME_CHANNEL: { label: 'ホーム会話 ID', help: 'Cron / 通知配信のデフォルトチャット ID。' },
      WECOM_ALLOWED_USERS: { label: '許可するユーザー', help: 'ボットと会話できる WeCom ユーザー ID。カンマ区切り。' },
      TELEGRAM_ALLOW_ALL_USERS: {
        label: 'すべての Telegram ユーザーを許可',
        help: '開発用のみ。すべての Telegram ユーザーがボットを利用できます。'
      },
      TELEGRAM_HOME_CHANNEL: { label: 'ホームチャンネル ID', help: 'Cron / 通知配信のデフォルトチャット ID。' },
      TELEGRAM_HOME_CHANNEL_NAME: { label: 'ホームチャンネル名', help: 'Telegram ホームチャンネルの表示名。' },
      SLACK_ALLOW_ALL_USERS: {
        label: 'すべての Slack ユーザーを許可',
        help: '開発用のみ。すべての Slack ユーザーがボットを利用できます。'
      },
      SLACK_HOME_CHANNEL: { label: 'ホームチャンネル ID', help: 'Cron / 通知配信のデフォルトチャンネル ID（C で始まる）。' },
      SLACK_HOME_CHANNEL_NAME: { label: 'ホームチャンネル名', help: 'Slack ホームチャンネルの表示名。' },
      SLACK_THREAD_REQUIRE_MENTION: {
        label: 'スレッド内で @メンションを必須にする',
        help: 'Slack スレッドの返信に明示的な @メンションを必須にします。トップレベルの自由応答チャンネルには影響しません。'
      },
      MATTERMOST_ALLOWED_CHANNELS: {
        label: '許可するチャンネル ID',
        help: '設定するとボットはこれらのチャンネルでのみ応答します（ホワイトリスト）。カンマ区切り。'
      },
      MATTERMOST_FREE_RESPONSE_CHANNELS: {
        label: '自由応答チャンネル ID',
        help: '@メンションなしでボットが応答する Mattermost チャンネル ID。カンマ区切り。'
      },
      MATTERMOST_REPLY_MODE: { label: '返信モード', help: 'thread（ネスト）または off（フラット）。デフォルト: off。' },
      MATTERMOST_REQUIRE_MENTION: {
        label: 'チャンネル内で @メンションを必須にする',
        help: 'Mattermost チャンネルで @メンションを必須にします（デフォルト: true）。false にするとすべてのメッセージに応答します。'
      },
      MATRIX_ALLOW_ALL_USERS: {
        label: 'すべての Matrix ユーザーを許可',
        help: '開発用のみ。すべての Matrix ユーザーがボットを利用できます。'
      },
      MATRIX_AUTO_THREAD: {
        label: 'ルームでスレッドを自動作成',
        help: 'Matrix ルームのメッセージにスレッドを自動作成します（デフォルト: true）。'
      },
      MATRIX_DEVICE_ID: {
        label: 'デバイス ID',
        help: 'E2EE 永続化のための再起動後も変わらない Matrix デバイス ID（例: HERMES_BOT）。'
      },
      MATRIX_DM_AUTO_THREAD: {
        label: 'DM でスレッドを自動作成',
        help: 'Matrix の DM にスレッドを自動作成します（デフォルト: false）。'
      },
      MATRIX_FREE_RESPONSE_ROOMS: {
        label: '自由応答ルーム ID',
        help: '@メンションなしでボットが応答する Matrix ルーム ID。カンマ区切り。'
      },
      MATRIX_HOME_CHANNEL: { label: 'ホームルーム ID', help: 'Cron / 通知配信のデフォルトルーム ID。' },
      MATRIX_HOME_CHANNEL_NAME: { label: 'ホームルーム名', help: 'Matrix ホームルームの表示名。' },
      MATRIX_PASSWORD: { label: 'Matrix パスワード', help: 'Matrix アカウントのパスワード（アクセストークンの代替）。' },
      MATRIX_RECOVERY_KEY: {
        label: 'リカバリーキー',
        help: 'デバイスキーのローテーション後にクロス署名検証へ使うリカバリーキー（Element: 設定 → セキュリティ → リカバリーキー）。'
      },
      MATRIX_REQUIRE_MENTION: {
        label: 'ルームで @メンションを必須にする',
        help: 'Matrix ルームで @メンションを必須にします（デフォルト: true）。false にするとすべてのメッセージに応答します。'
      },
      WHATSAPP_DM_POLICY: { label: 'DM ポリシー', help: 'WhatsApp ダイレクトメッセージの承認方法。' },
      WHATSAPP_ALLOW_ALL_USERS: {
        label: 'すべての WhatsApp ユーザーを許可',
        help: '開発用のみ。すべての WhatsApp ユーザーがボットを利用できます。'
      },
      WHATSAPP_HOME_CHANNEL: { label: 'ホームチャンネル ID', help: 'Cron / 通知配信のデフォルトチャット ID。' },
      WHATSAPP_HOME_CHANNEL_NAME: { label: 'ホームチャンネル名', help: 'WhatsApp ホームチャンネルの表示名。' },
      BLUEBUBBLES_SERVER_URL: {
        label: 'サーバー URL',
        help: 'iMessage 連携用の BlueBubbles サーバー URL。',
        placeholder: 'http://192.168.1.10:1234'
      },
      BLUEBUBBLES_PASSWORD: {
        label: 'サーバーパスワード',
        help: 'BlueBubbles サーバーのパスワード（BlueBubbles Server → 設定 → API）。'
      },
      BLUEBUBBLES_ALLOWED_USERS: {
        label: '許可する iMessage アドレス',
        help: '推奨。カンマ区切りの iMessage アドレス（メールまたは電話番号）。'
      },
      HASS_URL: {
        label: 'Home Assistant URL',
        help: 'Home Assistant のベース URL。',
        placeholder: 'http://homeassistant.local:8123'
      },
      HASS_TOKEN: { label: '長期アクセストークン', help: 'Home Assistant の長期アクセストークン。' },
      EMAIL_ADDRESS: { label: 'メールアドレス', help: 'メールアカウントのアドレス。' },
      EMAIL_PASSWORD: { label: 'メールパスワード', help: 'メールアカウントのパスワード / アプリパスワード。' },
      EMAIL_IMAP_HOST: { label: 'IMAP ホスト', help: '受信ポーリングに使う IMAP ホスト。', placeholder: 'imap.gmail.com' },
      EMAIL_SMTP_HOST: { label: 'SMTP ホスト', help: '送信に使う SMTP ホスト。', placeholder: 'smtp.gmail.com' },
      EMAIL_ALLOWED_USERS: {
        label: '許可するメールアドレス',
        help: '推奨。ボットと会話できるメールアドレス。カンマ区切り。'
      },
      EMAIL_HOME_ADDRESS: { label: 'ホームアドレス', help: 'Cron / 通知配信のデフォルトメールアドレス。' },
      EMAIL_SMTP_PORT: { label: 'SMTP ポート', help: 'SMTP ポート（デフォルト 587）。' },
      TWILIO_ACCOUNT_SID: { label: 'Twilio Account SID', help: 'Twilio コンソールの Account SID。' },
      TWILIO_AUTH_TOKEN: { label: 'Twilio Auth Token', help: 'Twilio コンソールの Auth Token。' },
      TWILIO_PHONE_NUMBER: { label: 'Twilio 電話番号', help: 'SMS を送信できる Twilio の番号（E.164 形式）。' },
      DINGTALK_CLIENT_ID: { label: 'Client ID (App Key)', help: 'DingTalk アプリの App Key（Client ID）。' },
      DINGTALK_CLIENT_SECRET: { label: 'Client Secret', help: 'DingTalk アプリの App Secret（Client Secret）。' },
      DINGTALK_ALLOWED_USERS: {
        label: '許可するユーザー',
        help: 'ボットと会話できるスタッフ / 送信者 ID。カンマ区切り（* は全員）。'
      },
      DINGTALK_HOME_CHANNEL: { label: 'ホーム会話 ID', help: 'Cron / 通知配信のデフォルト会話 ID。' },
      DINGTALK_HOME_CHANNEL_NAME: { label: 'ホーム会話名', help: 'DingTalk ホーム会話の表示名。' },
      DINGTALK_WEBHOOK_URL: {
        label: 'ロボット Webhook URL',
        help: 'クロスプラットフォーム / Cron 配信用の固定ロボット Webhook URL（任意）。'
      },
      FEISHU_APP_ID: { label: 'App ID', help: 'Feishu / Lark アプリの App ID。' },
      FEISHU_APP_SECRET: { label: 'App Secret', help: 'Feishu / Lark アプリの App Secret。' },
      FEISHU_ENCRYPT_KEY: { label: '暗号化キー (Encrypt Key)', help: 'Feishu / Lark のイベント暗号化キー。' },
      FEISHU_VERIFICATION_TOKEN: { label: '検証トークン (Verification Token)', help: 'Feishu / Lark のイベント検証トークン。' },
      FEISHU_ALLOWED_USERS: {
        label: '許可するユーザー ID',
        help: '推奨。ボットと会話できる Feishu ユーザー ID。カンマ区切り。'
      },
      FEISHU_ALLOW_ALL_USERS: {
        label: 'すべての Feishu ユーザーを許可',
        help: '開発用のみ。すべての Feishu ユーザーがボットを利用できます。'
      },
      FEISHU_DOMAIN: { label: 'ドメイン (feishu/lark)', help: 'feishu（中国版）または lark（国際版）。' },
      FEISHU_HOME_CHANNEL: { label: 'ホームチャット ID', help: 'Cron / 通知配信のデフォルトチャット ID。' },
      FEISHU_HOME_CHANNEL_NAME: { label: 'ホームチャット名', help: 'Feishu ホームチャットの表示名。' },
      WECOM_BOT_ID: { label: 'ボット ID', help: 'WeCom スマートロボットのボット ID。' },
      WECOM_SECRET: { label: 'ボット Secret', help: 'WeCom スマートロボットの secret。' },
      WECOM_CALLBACK_CORP_ID: { label: '企業 ID (Corp ID)', help: 'WeCom コールバックモードの企業 ID（自社構築アプリ）。' },
      WECOM_CALLBACK_CORP_SECRET: { label: 'アプリ Secret', help: 'WeCom コールバックモードのアプリ Secret。' },
      WECOM_CALLBACK_AGENT_ID: { label: 'アプリ Agent ID', help: 'WeCom コールバックモードのアプリ Agent ID。' },
      WECOM_CALLBACK_TOKEN: { label: 'コールバックトークン', help: 'WeCom コールバック検証トークン。' },
      WECOM_CALLBACK_ENCODING_AES_KEY: {
        label: 'EncodingAESKey',
        help: 'メッセージ暗号化用の WeCom コールバック EncodingAESKey。'
      },
      WEIXIN_ACCOUNT_ID: {
        label: 'iLink Bot アカウント ID',
        help: 'hermes gateway setup の QR ログインで取得した iLink Bot アカウント ID。'
      },
      WEIXIN_TOKEN: {
        label: 'iLink Bot トークン',
        help: 'hermes gateway setup の QR ログインで取得した iLink Bot トークン。'
      },
      WEIXIN_BASE_URL: {
        label: 'iLink API ベース URL',
        help: 'QR ログインで保存された iLink API ベース URL（デフォルト: https://ilinkai.weixin.qq.com）。'
      },
      QQ_APP_ID: { label: 'App ID', help: 'QQ オープンプラットフォーム (q.qq.com) のボット App ID。' },
      QQ_CLIENT_SECRET: { label: 'Client Secret', help: 'QQ オープンプラットフォームのボット Client Secret。' },
      QQ_ALLOWED_USERS: { label: '許可する QQ ユーザー', help: '推奨。ボットを利用できる QQ ユーザー ID。カンマ区切り。' },
      QQ_GROUP_ALLOWED_USERS: { label: '許可する QQ グループ', help: 'ボットと対話できる QQ グループ ID。カンマ区切り。' },
      QQ_SANDBOX: { label: 'サンドボックスモード', help: '開発テスト用に QQ サンドボックスモードを有効にします（true/false）。' },
      API_SERVER_ENABLED: {
        label: 'API サーバーを有効にする',
        help: 'OpenAI 互換の API サーバーを有効にします（true/false）。Open WebUI や LobeChat などのフロントエンドが接続できます。'
      },
      API_SERVER_KEY: {
        label: '認証キー',
        help: 'API サーバー認証用の Bearer トークン。API サーバーを有効にする場合は必須で、未設定だとサーバーは起動を拒否します。'
      },
      API_SERVER_PORT: { label: 'ポート', help: 'API サーバーのポート（デフォルト: 8642）。' },
      API_SERVER_HOST: {
        label: 'バインドアドレス',
        help: 'API サーバーのバインドアドレス（デフォルト: 127.0.0.1）。ループバックのみでも認証キーは必須です。'
      },
      API_SERVER_MODEL_NAME: {
        label: 'モデル名',
        help: '/v1/models で公開されるモデル名。デフォルトはプロファイル名（デフォルトプロファイルでは hermes-agent）。OpenWebUI のマルチユーザー構成に便利です。'
      },
      WEBHOOK_ENABLED: {
        label: 'Webhook を有効にする',
        help: 'GitHub や GitLab などからイベントを受信する Webhook アダプターを有効にします。'
      },
      WEBHOOK_PORT: { label: 'ポート', help: 'Webhook HTTP サーバーのポート（デフォルト: 8644）。' },
      WEBHOOK_SECRET: {
        label: '署名シークレット',
        help: 'Webhook 署名検証用のグローバル HMAC シークレット（config.yaml でルートごとに上書き可能）。'
      }
    },
    platformIntro: {
      telegram:
        'Telegram で @BotFather に話しかけて /newbot を実行し、表示されたトークンをコピーします。次に @userinfobot から数値のユーザー ID を取得します。',
      discord:
        'Discord Developer Portal を開いてアプリケーションを作成し、Bot を追加してそのトークンをコピーします。適切なスコープでボットをサーバーに招待してください。',
      slack:
        'Slack アプリを作成し、Socket Mode を有効にしてワークスペースにインストールし、ボットトークンとアプリレベルトークンをコピーします。',
      mattermost:
        'Mattermost サーバーでボットアカウントまたはパーソナルアクセストークンを作成し、サーバー URL とトークンをここに貼り付けます。',
      matrix: 'ボットアカウントでホームサーバーにサインインし、アクセストークン、ユーザー ID、ホームサーバー URL をコピーします。',
      signal: '到達可能な場所で signal-cli REST ブリッジを実行し、その URL と登録済みの電話番号を Hermes に設定します。',
      whatsapp: 'Hermes 同梱の WhatsApp ブリッジを起動し、初回実行時に QR コードをスキャンしてからプラットフォームを有効にします。',
      bluebubbles:
        'iMessage が使える Mac で BlueBubbles Server を実行して API を公開し、サーバーパスワードとともに Hermes をその URL に向けます。',
      homeassistant:
        'Home Assistant でプロフィールを開き、長期アクセストークンを作成します。HA の URL と一緒にここに貼り付けてください。',
      email:
        '専用メールボックスを使ってください。Gmail/Workspace ではアプリパスワードを作成し、imap.gmail.com / smtp.gmail.com を使用します。',
      sms: 'Twilio コンソールから Account SID と Auth Token、SMS 送信可能な電話番号を取得します。',
      dingtalk: '開発者コンソールで DingTalk アプリを作成し、Client ID (App key) と Client Secret をここにコピーします。',
      feishu: 'Feishu / Lark アプリを作成し、ボット機能を設定して、App ID、App secret、イベント暗号化キーをコピーします。',
      wecom:
        'WeCom でグループロボットを追加し、その webhook key を WECOM_BOT_ID としてコピーします。送信専用です — 双方向には WeCom (アプリ) を使ってください。',
      wecom_callback:
        'WeCom の自社構築アプリを設定し、コールバック URL を公開して、corp ID、secret、agent ID、AES key を指定します。',
      weixin:
        '`hermes gateway setup` を実行して Weixin を選択し、個人の WeChat アカウントで QR コードをスキャンして確認します。Hermes は Tencent の iLink Bot API 経由で接続し、資格情報を保存します。',
      qqbot: 'QQ オープンプラットフォーム (q.qq.com) でアプリを登録し、App ID と Client Secret をコピーします。',
      api_server:
        'Hermes を OpenAI 互換 API として公開します。認証キーを設定し、Open WebUI / LobeChat などを host:port に向けてください。',
      webhook:
        '他のツール (GitHub、GitLab、カスタムアプリ) が POST できる HTTP サーバーを実行します。シークレットで署名を検証します。'
    },
    platformDescription: {
      telegram: 'Telegram の DM、グループ、トピックで Hermes を使います。',
      discord: 'Discord の DM、チャンネル、スレッドに Hermes を接続します。',
      slack: 'Socket Mode 経由で Slack から Hermes を使います。許可する Slack メンバー ID を追加すると接続済みボットが応答します。',
      mattermost: 'Mattermost のチャンネルとダイレクトメッセージに Hermes を接続します。',
      matrix: 'Matrix のルームとダイレクトメッセージで Hermes を使います。',
      signal: 'signal-cli REST ブリッジ経由で接続します。',
      whatsapp: '同梱の WhatsApp ブリッジと QR 認証で Hermes を使います。',
      bluebubbles: 'BlueBubbles サーバー経由の iMessage で Hermes を使います。',
      homeassistant: 'Home Assistant 経由で Hermes からスマートホームを操作します。',
      email: 'IMAP/SMTP メールボックスを通じて Hermes と会話します。',
      sms: 'Twilio 経由でテキストメッセージを送受信します。',
      dingtalk: 'DingTalk（釘釘）のグループに Hermes を接続します。',
      feishu: 'Feishu / Lark の中で Hermes を使います。',
      google_chat: 'Cloud Pub/Sub 経由で Google Chat に Hermes を接続します。',
      wecom: 'Webhook 経由の送信専用 WeCom グループボット。',
      wecom_callback: 'コールバックアプリによる双方向の WeCom 連携。',
      weixin: 'Tencent の iLink Bot API を通じて個人の WeChat アカウントを接続します。',
      qqbot: 'QQ オープンプラットフォームの QQ ボットに Hermes を接続します。',
      yuanbao: 'Tencent Yuanbao に Hermes を接続します。',
      api_server: 'Hermes を OpenAI 互換 HTTP API として公開し、Open WebUI などのツールから使えるようにします。',
      webhook: 'GitHub、GitLab などの Webhook ソースからイベントを受信します。'
    }
  },

  profiles: {
    close: 'プロファイルを閉じる',
    nameHint: '小文字、数字、ハイフン、アンダースコア。文字または数字で始める必要があります。',
    title: 'プロファイル',
    count: count => `${count} プロファイル`,
    search: 'プロファイルを検索...',
    loading: 'プロファイルを読み込み中...',
    newProfile: '新しいプロファイル',
    allProfiles: 'すべてのプロファイル',
    showAllProfiles: 'すべてのプロファイルを表示',
    switchToProfile: name => `${name} に切り替え`,
    manageProfiles: 'プロファイルを管理…',
    actionsFor: name => `${name} のアクション`,
    color: 'カラー…',
    colorFor: name => `${name} のカラー`,
    setColor: color => `カラー ${color} に設定`,
    autoColor: '自動',
    noProfiles: 'プロファイルが見つかりません。',
    selectPrompt: '詳細を表示するにはプロファイルを選択してください。',
    refresh: 'プロファイルを更新',
    refreshing: 'プロファイルを更新中',
    default: 'デフォルト',
    skills: count => `${count} スキル`,
    env: 'env',
    defaultBadge: 'デフォルト',
    rename: '名前を変更',
    renameMenu: '名前を変更…',
    editSoul: 'SOUL.md を編集…',
    copySetup: 'セットアップをコピー',
    copying: 'コピー中...',
    modelLabel: 'モデル',
    skillsLabel: 'スキル',
    notSet: '未設定',
    soulDesc: 'このプロファイルに組み込まれたシステムプロンプトとペルソナの指示。',
    soulOptional: '省略可能',
    soulPlaceholder: mode =>
      `このプロファイルのシステムプロンプト / ペルソナ。\n空欄のままにすると ${mode} のデフォルトを使用します。`,
    soulPlaceholderCloned: 'クローン済み',
    soulPlaceholderEmpty: '空',
    unsavedChanges: '未保存の変更',
    loadingSoul: 'SOUL.md を読み込み中...',
    emptySoul: '空の SOUL.md — ペルソナの記述を始めてください...',
    saving: '保存中...',
    saveSoul: 'SOUL を保存',
    deleteTitle: 'プロファイルを削除しますか？',
    deleteDescPrefix: 'これにより ',
    deleteDescMid: ' が削除され、その ',
    deleteDescSuffix: ' ディレクトリが削除されます。この操作は元に戻せません。',
    deleting: '削除中...',
    createDesc: 'プロファイルは独立した Hermes 環境です：設定、スキル、SOUL.md が別々になります。',
    nameLabel: '名前',
    namePlaceholder: '例: my-profile',
    cloneFrom: '複製元',
    cloneFromNone: 'なし（空）',
    cloneFromDesc: '選択したプロファイルから設定、スキル、SOUL.md をコピーします。',
    cloneFromDefault: 'デフォルトプロファイルから設定を複製',
    cloneFromDefaultDesc: 'デフォルトプロファイルから設定、スキル、SOUL.md をコピーします。',
    invalidName: hint => `無効なプロファイル名。${hint}`,
    nameRequired: '名前は必須です',
    creating: '作成中...',
    createAction: 'プロファイルを作成',
    renameTitle: 'プロファイルの名前を変更',
    renameDescPrefix: '名前を変更するとプロファイルディレクトリと ',
    renameDescSuffix: ' 内のラッパースクリプトが更新されます。',
    newNameLabel: '新しい名前',
    renaming: '名前を変更中...',
    created: '作成しました',
    renamed: '名前を変更しました',
    deleted: '削除しました',
    setupCopied: 'セットアップコマンドをコピーしました',
    soulSaved: 'SOUL.md を保存しました',
    failedLoad: 'プロファイルの読み込みに失敗しました',
    failedDelete: 'プロファイルの削除に失敗しました',
    failedCopy: 'セットアップコマンドのコピーに失敗しました',
    failedLoadSoul: 'SOUL.md の読み込みに失敗しました',
    failedSaveSoul: 'SOUL.md の保存に失敗しました',
    failedCreate: 'プロファイルの作成に失敗しました',
    failedRename: 'プロファイルの名前変更に失敗しました'
  },

  cron: {
    close: 'Cron を閉じる',
    title: 'スケジュール済みジョブ',
    count: count => `${count} 件のジョブ`,
    search: 'Cron ジョブを検索...',
    loading: 'Cron ジョブを読み込み中...',
    states: {
      enabled: '有効',
      scheduled: 'スケジュール済み',
      running: '実行中',
      paused: '一時停止中',
      disabled: '無効',
      error: 'エラー',
      completed: '完了'
    },
    deliveryLabels: {
      local: 'このデスクトップ',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'メール'
    },
    scheduleLabels: {
      daily: '毎日',
      weekdays: '平日',
      weekly: '毎週',
      monthly: '毎月',
      hourly: '毎時',
      'every-15-minutes': '15 分ごと',
      custom: 'カスタム'
    },
    scheduleHints: {
      daily: '毎日午前 9:00',
      weekdays: '月曜日から金曜日の午前 9:00',
      weekly: '毎週月曜日午前 9:00',
      monthly: '毎月 1 日午前 9:00',
      hourly: '毎時 0 分',
      'every-15-minutes': '15 分ごと',
      custom: 'Cron 構文または自然言語'
    },
    days: {
      '0': '日曜日',
      '1': '月曜日',
      '2': '火曜日',
      '3': '水曜日',
      '4': '木曜日',
      '5': '金曜日',
      '6': '土曜日',
      '7': '日曜日'
    },
    dayFallback: value => `${value}日`,
    everyDayAt: time => `毎日 ${time} に`,
    weekdaysAt: time => `平日 ${time} に`,
    everyDayOfWeekAt: (day, time) => `毎週 ${day} ${time} に`,
    monthlyOnDayAt: (dayOfMonth, time) => `毎月 ${dayOfMonth} 日 ${time} に`,
    topOfHour: '毎時 0 分',
    everyHourAt: minute => `毎時 :${minute} に`,
    newCron: '新しい Cron',
    emptyDescNew:
      'Cron 式でプロンプトを実行するスケジュールを設定します。Hermes が実行して、選択した宛先に結果を送信します。',
    emptyDescSearch: '検索キーワードを広げてください。',
    emptyTitleNew: 'スケジュールされたジョブがまだありません',
    emptyTitleSearch: '一致なし',
    last: '前回',
    next: '次回',
    noRuns: 'まだ実行されていません',
    manage: '管理',
    showRuns: '実行履歴を表示',
    hideRuns: '実行履歴を隠す',
    runHistory: '実行履歴',
    actionsFor: title => `${title} のアクション`,
    actionsTitle: 'Cron ジョブのアクション',
    resume: '再開',
    pause: '一時停止',
    resumeTitle: '再開',
    pauseTitle: '一時停止',
    triggerNow: '今すぐ実行',
    edit: 'Cron を編集',
    deleteTitle: 'Cron ジョブを削除しますか？',
    deleteDescPrefix: 'これにより ',
    deleteDescSuffix: ' が完全に削除され、即座に実行が停止されます。',
    deleting: '削除中...',
    resumed: 'Cron を再開しました',
    paused: 'Cron を一時停止しました',
    triggered: 'Cron をトリガーしました',
    deleted: 'Cron を削除しました',
    created: 'Cron を作成しました',
    updated: 'Cron を更新しました',
    failedLoad: 'Cron ジョブの読み込みに失敗しました',
    failedUpdate: 'Cron ジョブの更新に失敗しました',
    failedTrigger: 'Cron ジョブのトリガーに失敗しました',
    failedDelete: 'Cron ジョブの削除に失敗しました',
    failedSave: 'Cron ジョブの保存に失敗しました',
    editTitle: 'Cron ジョブを編集',
    createTitle: '新しい Cron ジョブ',
    editDesc: 'スケジュール、プロンプト、または配信先を更新します。変更は次回の実行時に適用されます。',
    createDesc:
      'プロンプトを自動実行するスケジュールを設定します。Cron 構文または「15 分ごと」などのフレーズを使用します。',
    nameLabel: '名前',
    namePlaceholder: '例: 日次サマリー',
    promptLabel: 'プロンプト',
    promptPlaceholder: '実行ごとにエージェントが行う内容は？',
    frequencyLabel: '頻度',
    deliverLabel: '配信先',
    modelLabel: 'モデル',
    modelDefault: 'デフォルト（グローバルモデル）',
    customScheduleLabel: 'カスタムスケジュール',
    customPlaceholder: '0 9 * * * または weekdays at 9am',
    customHint: 'Cron 式、または「every hour」「weekdays at 9am」のようなフレーズ。',
    optional: '省略可能',
    promptRequired: 'プロンプトは必須です。',
    promptScheduleRequired: 'プロンプトとスケジュールは必須です。',
    scheduleRequired: 'スケジュールは必須です。',
    scriptOnlyEditHint: 'スクリプトのみのジョブ（AI プロンプトなし）。ジョブ ID:',
    saveChanges: '変更を保存',
    createAction: 'Cron を作成'
  },

  artifacts: {
    search: 'アーティファクトを検索...',
    refresh: 'アーティファクトを更新',
    refreshing: 'アーティファクトを更新中',
    indexing: '最近のセッションのアーティファクトをインデックス中',
    tabAll: 'すべて',
    tabImages: '画像',
    tabFiles: 'ファイル',
    tabLinks: 'リンク',
    noArtifactsTitle: 'アーティファクトが見つかりません',
    noArtifactsDesc: 'セッションで生成された画像やファイルの出力がここに表示されます。',
    failedLoad: 'アーティファクトの読み込みに失敗しました',
    openFailed: '開くことができませんでした',
    itemsImage: '画像',
    itemsLink: 'リンク',
    itemsFile: 'ファイル',
    itemsGeneric: '項目',
    zero: '0',
    rangeOf: (start, end, total) => `${total} 件中 ${start}-${end}`,
    goToPage: (itemLabel, page) => `${itemLabel} ページ ${page} に移動`,
    colTitleLink: 'リンクタイトル',
    colTitleFile: '名前',
    colTitleDefault: 'タイトル / 名前',
    colLocationLink: 'URL',
    colLocationFile: 'パス',
    colLocationDefault: '場所',
    colSession: 'セッション',
    kindImage: '画像',
    kindFile: 'ファイル',
    kindLink: 'リンク',
    chat: 'チャット',
    copyUrl: 'URL をコピー',
    copyPath: 'パスをコピー'
  },

  sidebar: {
    nav: {
      'new-session': '新しいセッション',
      skills: 'スキルとツール',
      messaging: 'メッセージング',
      artifacts: 'アーティファクト'
    },
    searchAria: 'セッションを検索',
    searchPlaceholder: 'セッションを検索…',
    clearSearch: '検索をクリア',
    noMatch: query => `"${query}" に一致するセッションがありません。`,
    results: '結果',
    pinned: 'ピン留め',
    sessions: 'セッション',
    cronJobs: 'Cronジョブ',
    groupAriaGrouped: 'セッションを単一リストとして表示',
    groupAriaUngrouped: 'ワークスペースごとにセッションをグループ化',
    showProjects: 'プロジェクトを表示',
    showSessions: 'セッションを表示',
    groupTitleGrouped: 'セッションのグループ化を解除',
    groupTitleUngrouped: 'ワークスペースでグループ化',
    allPinned: 'ここにあるものはすべてピン留めされています。チャットのピン留めを解除すると最近のものに表示されます。',
    shiftClickHint: 'Shift クリックでピン留め · ドラッグで並べ替え',
    noWorkspace: 'ワークスペースなし',
    noProject: 'プロジェクトなし',
    projectEmpty: 'セッションはまだありません',
    noSessions: 'セッションはまだありません',
    projects: {
      sectionLabel: 'プロジェクト',
      newButton: '新規プロジェクト',
      createTitle: '新規プロジェクト',
      createDesc: 'ワークスペースに名前を付け、1つ以上のフォルダを追加します。',
      renameTitle: 'プロジェクト名を変更',
      addFolderTitle: 'フォルダを追加',
      namePlaceholder: '例: Skunkworks',
      foldersLabel: 'フォルダ',
      ideaLabel: 'アイデア',
      ideaPlaceholder: 'このプロジェクトは何ですか？（IDEA.md に保存）',
      ideaGenerate: 'アイデアを生成',
      ideaGenerating: '生成中…',
      ideaShuffle: 'テンプレートをシャッフル',
      noFolders: 'まだフォルダがありません。',
      addFolder: 'フォルダを追加',
      primaryBadge: 'メイン',
      removeFolder: '削除',
      create: '作成',
      menu: 'プロジェクト操作',
      menuRename: '名前を変更',
      menuAppearance: '外観',
      noColor: '色なし',
      menuAddFolder: 'フォルダを追加',
      menuSetActive: 'アクティブに設定',
      menuDelete: '削除',
      reveal: 'フォルダで表示',
      copyPath: 'パスをコピー',
      removeFromSidebar: 'サイドバーから削除',
      createFailed: 'プロジェクトを作成できませんでした',
      staleBackend:
        'プロジェクトを作成するには Hermes バックエンドを更新してください。バックエンドがこのデスクトップアプリより古いです（設定 → 更新 → バックエンド）。',
      deleteConfirm:
        'Hermes から保存済みプロジェクトを削除します。ファイル・git リポジトリ・ワークツリーはそのまま残ります。',
      startWork: '新しいワークツリー',
      newWorktreeTitle: '新しいワークツリー',
      newWorktreeDesc: 'このワークツリーのブランチ名を入力してください。',
      branchPlaceholder: '例: my-feature',
      branchOff: () => ({ after: ' から分岐', before: '' }),
      baseBranchPlaceholder: 'ブランチを検索…',
      baseBranchNone: 'ブランチが見つかりません',
      startWorkFailed: 'ワークツリーを作成できませんでした',
      convertBranch: 'ブランチを変換…',
      convertBranchTitle: 'ブランチを変換',
      convertBranchDesc: 'チェックアウト済みのブランチを開くか、空いているブランチのワークツリーを作成します。',
      convertBranchPlaceholder: 'ブランチを検索…',
      convertBranchInstead: '既存のブランチを変換',
      branchOpenExisting: '開く',
      branchSwitchHome: 'ホームを切替',
      branchCreateWorktree: '新しいワークツリー',
      branchesLoading: 'ブランチを読み込み中…',
      noBranches: 'ブランチが見つかりません',
      removeWorktree: 'ワークツリーを削除',
      removeWorktreeFailed: 'ワークツリーを削除できませんでした（コミットされていない変更？）',
      removeWorktreeConfirm:
        'git から削除（ワークツリーのディレクトリを削除しますが、ブランチは残ります）するか、サイドバーからレーンを隠してワークツリーをディスク上に残します。',
      removeWorktreeDirty:
        'このワークツリーにはコミットされていない変更があります。強制削除（変更を破棄）するか、レーンを隠してディスク上に残します。',
      forceRemove: '強制削除',
      enter: label => `${label} を開く`
    },
    newSessionIn: label => `${label} で新しいセッション`,
    showMoreIn: (count, label) => `${label} でさらに ${count} 件を表示`,
    loading: '読み込み中…',
    loadMore: 'さらに読み込む',
    loadCount: step => `さらに ${step} 件を読み込む`,
    row: {
      pin: 'ピン留め',
      unpin: 'ピン留めを解除',
      copyId: 'ID をコピー',
      export: 'エクスポート',
      branchFrom: '分岐',
      rename: '名前を変更',
      archive: 'アーカイブ',
      newWindow: '新しいウィンドウ',
      copyIdFailed: 'セッション ID をコピーできませんでした',
      actionsFor: title => `${title} のアクション`,
      sessionActions: 'セッションアクション',
      sessionRunning: 'セッション実行中',
      needsInput: '入力が必要です',
      waitingForAnswer: '回答を待っています',
      finishedUnread: '完了 — 未読',
      backgroundRunning: 'バックグラウンドタスク実行中',
      handoffOrigin: platform => `${platform} から引き継ぎ`,
      ownedByProfile: profile => `プロファイル: ${profile}`,
      renamed: '名前を変更しました',
      renameFailed: '名前の変更に失敗しました',
      renameTitle: 'セッションの名前を変更',
      renameDesc: 'このチャットにわかりやすいタイトルをつけてください。空欄にするとクリアされます。',
      untitledPlaceholder: '無題のセッション',
      untitledChat: id => `セッション ${id}`,
      ageNow: 'たった今',
      ageDay: '日',
      ageHour: '時間',
      ageMin: '分'
    }
  },

  composer: {
    message: 'メッセージ',
    wakingProfile: profile => `${profile} を起動中…`,
    placeholderStarting: 'Hermes を起動中...',
    placeholderReconnecting: 'Hermes に再接続中…',
    placeholderFollowUp: 'フォローアップを送信',
    newSessionPlaceholders: [
      '何を作りますか？',
      'Hermes にタスクを与える',
      '何か考えていることはありますか？',
      '必要なことを説明してください',
      '何に取り組みますか？',
      '何でも聞いてください',
      '目標から始める'
    ],
    followUpPlaceholders: [
      'フォローアップを送信',
      'さらにコンテキストを追加',
      'リクエストを改善',
      '次は何ですか？',
      '続けましょう',
      'さらに進める',
      '調整または続行'
    ],
    startVoice: '音声会話を開始',
    queueMessage: 'メッセージをキューに入れる',
    stop: '停止',
    send: '送信',
    speaking: '話しています',
    transcribing: '文字起こし中',
    thinking: '考え中',
    muted: 'ミュート',
    listening: '聴いています',
    muteMic: 'マイクをミュート',
    unmuteMic: 'マイクのミュートを解除',
    stopListening: '聴き取りを停止して送信',
    stopShort: '停止',
    endConversation: '音声会話を終了',
    endShort: '終了',
    stopDictation: '口述を停止',
    transcribingDictation: '口述を文字起こし中',
    voiceDictation: '音声口述',
    speakReplies: '返信を読み上げる',
    stopSpeakingReplies: '返信の読み上げを停止',
    lookupLoading: '検索中…',
    lookupNoMatches: '一致なし。',
    lookupTry: '試す',
    lookupOr: 'または',
    commonCommands: '一般的なコマンド',
    hotkeys: 'ホットキー',
    helpFooter: 'フルパネルを開く · Backspace で閉じる',
    commandDescs: {
      '/help': 'コマンドとホットキーの全リスト',
      '/clear': '新しいセッションを開始',
      '/resume': '以前のセッションを再開',
      '/details': 'トランスクリプトの詳細レベルを制御',
      '/copy': '選択または最後のアシスタントメッセージをコピー',
      '/quit': 'hermes を終了'
    },
    hotkeyDescs: {
      'composer.mention': 'ファイル、フォルダー、URL、Git を参照',
      'composer.slash': 'スラッシュコマンドパレット',
      'composer.help': 'クイックヘルプ（削除で閉じる）',
      'composer.sendNewline': '送信 · 改行は Shift+Enter',
      'composer.sendQueued': '次のキュー済みターンを送信',
      'keybinds.openPanel': 'すべてのキーボードショートカット',
      'composer.cancel': 'ポップオーバーを閉じる · 実行をキャンセル',
      'composer.history': 'ポップオーバー / 履歴を切り替え'
    },
    attachUrlTitle: 'URL を添付',
    attachUrlDesc: 'Hermes がページを取得し、このターンのコンテキストとして含めます。',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: '完全な URL を入力してください。例: ',
    attach: '添付',
    queued: count => `${count} 件キュー済み`,
    queuedPaused: count => `${count} 件キュー済み — 一時停止中`,
    attachmentOnly: '添付のみのターン',
    emptyTurn: '空のターン',
    attachments: count => `${count} 件の添付`,
    editingInComposer: 'コンポーザーで編集中',
    editingQueuedInComposer: 'コンポーザーでキュー済みターンを編集中',
    queueEdit: '編集',
    queueSendNext: '次に送信',
    queueSend: '送信',
    queueDelete: '削除',
    queueResume: '再開',
    queueResumeTip: '停止により一時停止中 — キュー済みターンの送信を再開します',
    queueStuckTitle: 'キュー内のメッセージを送信できません',
    queueStuckBody:
      'キューに入れたターンの送信が繰り返し失敗しました。まだキューに残っています。もう一度送信してください。',
    previewUnavailable: 'プレビューは利用できません',
    previewLabel: label => `${label} のプレビュー`,
    couldNotPreview: label => `${label} をプレビューできませんでした`,
    removeAttachment: label => `${label} を削除`,
    dictating: '口述中',
    preparingAudio: '音声を準備中',
    speakingResponse: '応答を読み上げ中',
    readingAloud: '読み上げ中',
    themeSuggestions: 'デスクトップテーマの候補',
    noMatchingThemes: '一致するテーマがありません。',
    themeTryPre: '試してみる: ',
    themeTryPost: '。',
    attachLabel: '添付',
    files: 'ファイル…',
    folder: 'フォルダー…',
    images: '画像…',
    pasteImage: '画像を貼り付け',
    url: 'URL…',
    promptSnippets: 'プロンプトスニペット…',
    tipPre: 'ヒント: ',
    tipPost: ' と入力してファイルをインラインで参照。',
    snippetsTitle: 'プロンプトスニペット',
    snippetsDesc: 'スターターのプロンプトをコンポーザーに挿入します。',
    dropFiles: 'ファイルをドロップして添付',
    dropSession: 'ドロップしてこのチャットをリンク',
    snippets: {
      codeReview: {
        label: 'コードレビュー',
        description: '回帰、エッジケースの欠落、テストの欠如を確認します。',
        text: 'バグ、回帰、テストの欠如を確認してください。'
      },
      implementationPlan: {
        label: '実装計画',
        description: 'コードに手をつける前にアプローチを概説して、差分を集中させます。',
        text: 'コードを変更する前に簡潔な実装計画を立ててください。'
      },
      explainThis: {
        label: 'これを説明する',
        description: '選択したコードがどのように機能するかを説明し、主要なファイルにリンクします。',
        text: 'これがどのように機能するか説明し、主要なファイルを教えてください。'
      }
    }
  },

  statusStack: {
    agents: 'エージェント',
    background: count => `バックグラウンド ${count} 件`,
    subagents: count => `サブエージェント ${count} 件`,
    todos: (done, total) => `タスク ${done}/${total}`,
    running: '実行中',
    stop: '停止',
    dismiss: '閉じる',
    exit: code => `終了コード ${code}`,
    coding: {
      title: 'ワークツリー',
      noBranch: 'ブランチなし',
      detached: 'デタッチ',
      clean: 'クリーン',
      changed: count => `${count} 件変更`,
      ahead: count => `${count} 先行`,
      behind: count => `${count} 遅延`,
      review: 'レビュー',
      close: '閉じる',
      openChanges: '変更を開く',
      openFile: 'ファイルを開く',
      stage: 'ステージ',
      unstage: 'ステージ解除',
      stageAll: 'すべてステージ',
      viewAsTree: 'ツリー表示',
      viewAsList: 'リスト表示',
      revert: '取り消し',
      revertAll: 'すべて取り消し',
      revertConfirm: 'このファイルの変更を破棄してコミット済みの状態に戻しますか？この操作は元に戻せません。',
      revertAllConfirm: 'すべての変更を破棄してコミット済みの状態に戻しますか？この操作は元に戻せません。',
      staged: 'ステージ済み',
      noChanges: '変更なし',
      notRepo: 'Git リポジトリではありません',
      noDiff: '表示する差分がありません',
      scopeUncommitted: '未コミット',
      scopeBranch: 'ブランチ',
      scopeLastTurn: '前のターン',
      commit: 'コミット',
      commitAndPush: 'コミットしてプッシュ',
      commitPlaceholder: 'メッセージ（⌘↵ でコミット）',
      generateCommitMessage: 'コミットメッセージを生成',
      stopGenerating: '生成を停止',
      createPr: 'PR を作成',
      openPr: 'PR を開く',
      ghMissing: 'PR を開くには GitHub CLI (gh) をインストールしてサインインしてください',
      agentShip: 'Hermes にコミットと PR を任せる',
      agentShipPrompt:
        '現在の変更を確認し、分かりやすい Conventional Commits 形式でコミットし、ブランチをプッシュして、プルリクエストを作成してください。',
      newBranch: '新しいブランチ',
      branchOffFrom: base => `${base} から新しいブランチ`,
      switchTo: branch => `${branch} に切り替え`,
      switchFailed: branch => `${branch} に切り替えできませんでした`,
      worktrees: 'ワークツリー'
    }
  },

  updates: {
    stages: {
      idle: '準備中…',
      prepare: '準備中…',
      fetch: 'ダウンロード中…',
      pull: 'もうすぐ完了…',
      pydeps: '仕上げ中…',
      update: 'Hermes を更新中…',
      rebuild: 'デスクトップアプリを再ビルド中…',
      restart: 'Hermes を再起動中…',
      done: '更新が完了しました',
      manual: 'ターミナルから更新',
      guiSkew: 'デスクトップアプリを更新してください',
      error: '更新が一時停止中'
    },
    checking: '更新を確認中…',
    checkFailedTitle: '更新を確認できませんでした',
    tryAgain: '再試行',
    notAvailableTitle: '更新は利用できません',
    unsupportedMessage: 'このバージョンの Hermes はアプリ内から自分を更新できません。',
    connectionRetry: '接続を確認してもう一度試してください。',
    latestBody: '最新バージョンを実行しています。',
    latestBodyBackend: 'バックエンドは最新バージョンを実行しています。',
    allSetTitle: '準備完了',
    availableTitle: '新しい更新が利用可能',
    availableBody: '新しいバージョンの Hermes をインストールする準備ができています。',
    availableTitleBackend: 'バックエンドの更新があります',
    availableBodyBackend: '接続中の Hermes バックエンドの新しいバージョンをインストールできます。',
    availableBodyNoChangelog:
      '新しいバージョンを利用できます。このインストール形式ではリリースノートは表示できません。',
    updateNow: '今すぐ更新',
    maybeLater: '後で',
    moreChanges: count => `さらに ${count} 件の変更が含まれています。`,
    manualTitle: 'ターミナルから更新',
    manualBody:
      'Hermes をコマンドラインからインストールしたため、更新もそこで実行されます。これをターミナルに貼り付けてください:',
    manualPickedUp: 'Hermes は次回起動時に新しいバージョンを読み込みます。',
    guiSkewTitle: 'デスクトップアプリを更新してください',
    guiSkewBody:
      'バックエンドは更新されましたが、このデスクトップアプリのパッケージは変更されていません。一致させるために Hermes デスクトップアプリ（AppImage / .deb / .rpm）を更新または再インストールしてください。',
    copy: 'コピー',
    copied: 'コピーしました',
    done: '完了',
    applyingBody:
      'Hermes アップデーターが独自のウィンドウで引き継ぎ、完了後に自動的に Hermes を再度開きます。更新中はご自分で Hermes を開き直さないでください。',
    applyingBodyBackend: 'リモートバックエンドが更新を適用して再起動します。復帰すると Hermes が自動的に再接続します。',
    applyingClose: 'このウィンドウは更新中に閉じ、その後 Hermes が自動的に再度開きます。',
    errorTitle: '更新が完了しませんでした',
    errorBody: 'ご安心ください。何も失われていません。今すぐ再試行できます。',
    notNow: '今は後で',
    applyStatus: {
      preparing: 'バックエンドを更新しています…',
      pulling: 'バックエンドを更新中…',
      restarting: 'バックエンドが更新を読み込むため再起動しています…',
      notAvailable: 'このバックエンドでは更新を利用できません。',
      failed: 'バックエンドの更新に失敗しました。',
      noReturn:
        'バックエンドがオンラインに戻りませんでした。更新が完了していない可能性があります。バックエンドホストを確認してください。'
    }
  },

  install: {
    stageStates: {
      pending: '待機中',
      running: 'インストール中',
      succeeded: '完了',
      skipped: 'スキップ',
      failed: '失敗'
    },
    oneTimeTitle: 'Hermes には一度限りのインストールが必要です',
    unsupportedDesc: platform =>
      `${platform} では自動の初回インストールはまだ利用できません。ターミナルを開いて以下のコマンドを実行し、このアプリを再起動してください。以降の起動ではこの手順はスキップされます。`,
    installCommand: 'インストールコマンド',
    copyCommand: 'コマンドをコピー',
    viewDocs: 'インストールドキュメントを見る',
    installTo: 'インストール先',
    retryAfterRun: '実行しました — 再試行',
    failedTitle: 'インストールに失敗しました',
    settingUpTitle: 'Hermes Agent を設定中',
    finishingTitle: '仕上げ中',
    failedDesc:
      'インストール手順のいずれかが失敗しました。Windows では、別の Hermes CLI またはデスクトップインスタンスが実行中の場合に発生することがあります。実行中の Hermes インスタンスをすべて停止してから再試行してください。詳細は以下またはデスクトップログで確認できます。',
    activeDesc:
      'これは一回限りのセットアップです。Hermes インストーラーが依存関係をダウンロードしてマシンを設定しています。以降の起動ではこの手順はスキップされます。',
    progress: (completed, total) => `${total} ステップ中 ${completed} 完了`,
    currentStage: stage => ` — 現在: ${stage}`,
    fetchingManifest: 'インストーラーマニフェストを取得中...',
    error: 'エラー',
    hideOutput: 'インストーラーの出力を非表示',
    showOutput: 'インストーラーの出力を表示',
    lines: count => `${count} 行`,
    noOutput: 'まだ出力がありません。',
    cancelling: 'キャンセル中...',
    cancelInstall: 'インストールをキャンセル',
    transcriptSaved: 'フルトランスクリプトを保存しました:',
    copiedOutput: 'コピーしました！',
    copyOutput: '出力をコピー',
    reloadRetry: '再読み込みして再試行'
  },

  onboarding: {
    headerTitle: 'Hermes Agent のセットアップをしましょう',
    headerDesc: 'チャットを始めるにはモデルプロバイダーを接続してください。ほとんどのオプションはワンクリックです。',
    providerTitles: {
      anthropic: 'Anthropic API キー',
      'claude-code': 'Anthropic OAuth: サブスクリプション利用には追加使用クレジットが必要'
    },
    preparingInstall: 'Hermes はインストールを完了中です。初回実行では通常 1 分以内に完了します。',
    starting: 'Hermes を起動中…',
    lookingUpProviders: 'プロバイダーを検索中...',
    collapse: '折りたたむ',
    otherProviders: 'その他のプロバイダー',
    haveApiKey: 'API キーをお持ちです',
    chooseLater: '後でプロバイダーを選択します',
    recommended: '推奨',
    connected: '接続済み',
    featuredPitch: '1 つのサブスクリプションで 300 以上の最先端モデル — Hermes を実行するための推奨方法',
    fireworksPitch: '直接モデル API — Fireworks がホストする最先端モデル',
    openRouterPitch: '1 つのキーで数百のモデル — 堅実なデフォルト',
    apiKeyOptions: {
      fireworks: {
        short: 'モデル API に直接接続',
        description: 'Fireworks AI がホストするモデルに直接アクセスします。'
      },
      openrouter: {
        short: '1 つのキーで多くのモデル',
        description: '1 つのキーで数百のモデルをホスト。新規インストールのデフォルトとして最適。'
      },
      openai: { short: 'GPT クラスのモデル', description: 'OpenAI モデルへの直接アクセス。' },
      gemini: { short: 'Gemini モデル', description: 'Google Gemini モデルへの直接アクセス。' },
      xai: { short: 'Grok モデル', description: 'xAI Grok モデルへの直接アクセス。' },
      local: {
        short: 'セルフホスト',
        description:
          'ローカルまたはセルフホストの OpenAI 互換エンドポイント（vLLM、llama.cpp、Ollama など）に Hermes を接続。'
      }
    },
    backToSignIn: 'サインインに戻る',
    getKey: 'キーを取得',
    replaceCurrent: '現在の値を置き換え',
    pasteApiKey: 'API キーを貼り付け',
    couldNotSave: '認証情報を保存できませんでした。',
    connecting: '接続中',
    update: '更新',
    flowSubtitles: {
      pkce: 'ブラウザーを開いてサインインし、ここに戻ります',
      device_code: 'ブラウザーで確認ページを開きます — Hermes が自動接続します',
      external: 'ターミナルで一度サインインして、チャットに戻ります'
    },
    startingSignIn: provider => `${provider} のサインインを開始中...`,
    verifyingCode: provider => `${provider} でコードを確認中...`,
    connectedProvider: provider => `${provider} が接続されました`,
    connectedPicking: provider => `${provider} が接続されました。デフォルトモデルを選択中...`,
    signInFailed: 'サインインに失敗しました。再試行してください。',
    pickDifferentProvider: '別のプロバイダーを選択',
    signInWith: provider => `${provider} でサインイン`,
    openedBrowser: provider => `${provider} をブラウザーで開きました。`,
    authorizeThere: 'そこで Hermes を承認してください。',
    copyAuthCode: '認証コードをコピーして以下に貼り付けてください。',
    pasteAuthCode: '認証コードを貼り付け',
    reopenAuthPage: '認証ページを再度開く',
    autoBrowser: provider =>
      `${provider} をブラウザーで開きました。Hermes をそこで承認すれば自動接続されます。コピーや貼り付けは不要です。`,
    reopenSignInPage: 'サインインページを再度開く',
    waitingAuthorize: '承認を待っています...',
    externalPending: provider =>
      `${provider} は独自の CLI からサインインします。ターミナルでこのコマンドを実行してから、戻って「サインインしました」を選択してください:`,
    signedIn: 'サインインしました',
    deviceCodeOpened: provider => `${provider} をブラウザーで開きました。そこにこのコードを入力してください:`,
    reopenVerification: '確認ページを再度開く',
    copy: 'コピー',
    defaultModel: 'デフォルトモデル',
    freeTier: '無料プラン',
    pro: 'Pro',
    free: '無料',
    price: (input, output) => `${input} 入力 / ${output} 出力 per Mtok`,
    change: '変更',
    startChatting: '始める',
    docs: provider => `${provider} ドキュメント`
  },

  modelPicker: {
    title: 'モデルを切り替え',
    current: '現在:',
    unknown: '(不明)',
    search: 'プロバイダーとモデルをフィルター...',
    noModels: 'モデルが見つかりません。',
    addProvider: 'プロバイダーを追加',
    loadFailed: 'モデルを読み込めませんでした',
    noAuthenticatedProviders: '認証済みプロバイダーがありません。',
    moaWarning: 'アグリゲーターが選択されたモデルとして応答します。各呼び出しの前に、リファレンスモデルが分析を提供します。',
    pro: 'Pro',
    proNeedsSubscription: 'Pro モデルには有料の Nous サブスクリプションが必要です。',
    free: '無料',
    freeTier: '無料プラン',
    priceTitle: '100 万トークンあたりの入力/出力価格',
    wasPrice: '旧価格'
  },

  modelVisibility: {
    title: 'モデル',
    search: 'モデルを検索',
    noAuthenticatedProviders: '認証済みプロバイダーがありません。',
    addProvider: 'プロバイダーを追加…'
  },

  shell: {
    windowControls: 'ウィンドウコントロール',
    paneControls: 'ペインコントロール',
    appControls: 'アプリコントロール',
    modelMenu: {
      search: 'モデルを検索',
      noModels: 'モデルが見つかりません',
      editModels: 'モデルを編集…',
      refreshModels: 'モデルを更新',
      fast: '高速',
      medium: '中',
      moaPresets: 'MOA プリセット'
    },
    modelOptions: {
      noOptions: 'このモデルにはオプションがありません',
      options: 'オプション',
      thinking: '思考',
      fast: '高速',
      effort: '努力度',
      minimal: '最小',
      low: '低',
      medium: '中',
      high: '高',
      xhigh: '特高',
      max: '最大',
      ultra: 'ウルトラ',
      updateFailed: 'モデルオプションの更新に失敗しました',
      fastFailed: '高速モードの更新に失敗しました'
    },
    gatewayMenu: {
      gateway: 'ゲートウェイ',
      connected: '接続済み',
      connecting: '接続中',
      offline: 'オフライン',
      inferenceReady: '推論準備完了',
      inferenceNotReady: '推論準備未完了',
      checkingInference: '推論を確認中',
      disconnected: '切断済み',
      openSystem: 'システムパネルを開く',
      connection: label => `接続: ${label}`,
      recentActivity: '最近のアクティビティ',
      viewAllLogs: 'すべてのログを見る →',
      messagingPlatforms: 'メッセージングプラットフォーム'
    },
    approvalMode: {
      title: '承認モード',
      ariaLabel: mode => `承認モード: ${mode}`,
      manual: '手動',
      manualDescription: '承認が必要な操作の前に確認します',
      smart: 'スマート',
      smartDescription: '必要な場合にのみ確認します',
      off: 'オフ',
      offDescription: '承認プロンプトなしで実行します'
    },
    statusbar: {
      unknown: '不明',
      restart: '再起動',
      update: '更新',
      updateInProgress: '更新中',
      commitsBehind: (count, branch) => `${branch} より ${count} コミット遅れています`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `バックエンド v${version}`,
      clientLabel: version => `クライアント v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `リモート: ${host}`,
      connectionCloud: host => `クラウド: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud ${host} に接続中 · クリックして管理`,
      connectionSshTooltip: host => `SSH 経由で ${host} に接続中 · クリックして管理`,
      connectionRemoteTooltip: host => `リモートバックエンド ${host} に接続中 · クリックして管理`,
      backendLabel: version => `バックエンド v${version}`,
      commit: sha => `コミット ${sha}`,
      branch: branch => `ブランチ ${branch}`,
      closeCommandCenter: 'コマンドセンターを閉じる',
      openCommandCenter: 'コマンドセンターを開く',
      showTerminal: 'ターミナルを表示',
      hideTerminal: 'ターミナルを非表示',
      gateway: 'ゲートウェイ',
      gatewayReady: '準備完了',
      gatewayNeedsSetup: '設定が必要',
      gatewayChecking: '確認中',
      gatewayConnecting: '接続中',
      gatewayOffline: 'オフライン',
      gatewayRestarting: '再起動中…',
      gatewayTitle: 'Hermes 推論ゲートウェイのステータス',
      agents: 'エージェント',
      closeAgents: 'エージェントを閉じる',
      openAgents: 'エージェントを開く',
      subagents: count => `${count} サブエージェント`,
      failed: count => `${count} 失敗`,
      running: count => `${count} 実行中`,
      cron: 'Cron',
      openCron: 'Cron ジョブを開く',
      starmap: 'メモリグラフ',
      openStarmap: 'メモリグラフを開く',
      turnRunning: '実行中',
      currentTurnElapsed: '現在のターン経過時間',
      contextUsage: 'コンテキスト使用状況',
      contextUsagePanel: {
        categories: {
          conversation: '会話',
          mcp: 'MCP',
          memory: 'メモリ',
          rules: 'ルール',
          skills: 'スキル',
          subagent_definitions: 'サブエージェント定義',
          system_prompt: 'システムプロンプト',
          tool_definitions: 'ツール定義'
        },
        empty: 'コンテキストデータはまだありません',
        loading: '内訳を読み込み中…',
        percentFull: percent => `${percent}% 使用中`,
        title: 'コンテキスト使用状況',
        tokenSummary: (used, max) => `${used} / ${max} Tokens`
      },
      openContextUsage: 'コンテキスト使用状況の内訳を開く',
      session: 'セッション',
      runtimeSessionElapsed: 'ランタイムセッション経過時間',
      yoloOn: 'YOLO オン — 危険なコマンドを自動承認中。クリックでオフに。Shift+クリックで全体に切り替え。',
      yoloOff: 'YOLO オフ — クリックで危険なコマンドを自動承認。Shift+クリックで全体に切り替え。',
      modelNone: 'なし',
      noModel: 'モデルなし',
      switchModel: 'モデルを切り替え',
      openModelPicker: 'モデルピッカーを開く',
      modelPinned: '手動で固定中 — 新しいチャットは設定のデフォルトではなくこのモデルを使用します',
      modelTitle: (provider, model) => `モデル · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },

  rightSidebar: {
    aria: '右サイドバー',
    panelsAria: '右サイドバーパネル',
    files: 'ファイルシステム',
    terminal: 'ターミナル',
    noFolderSelected: 'フォルダーが選択されていません',
    changeCwdTitle: '作業ディレクトリを変更',
    remotePickerTitle: 'リモートフォルダーを選択',
    remotePickerDescription: '接続中のバックエンド上のフォルダーを参照します。',
    remotePickerSelect: 'フォルダーを選択',
    folderTip: cwd => `${cwd} — クリックしてフォルダーを変更`,
    openFolder: 'フォルダーを開く',
    refreshTree: 'ツリーを更新',
    collapseAll: 'すべてのフォルダーを折りたたむ',
    previewUnavailable: 'プレビューは利用できません',
    couldNotPreview: path => `${path} をプレビューできませんでした`,
    noProjectTitle: 'プロジェクトなし',
    noProjectBody: 'プロジェクトを開くと、ファイルの閲覧と変更の確認ができます。',
    noProjectOpen: 'プロジェクト未選択',
    noDiffs: '差分なし',
    unreadableTitle: '読み取り不可',
    unreadableBody: error => `このフォルダーを読み取れませんでした (${error})。`,
    emptyTitle: '空',
    emptyBody: 'このフォルダーは空です。',
    treeErrorTitle: 'ツリーエラー',
    treeErrorBody: 'ファイルツリーがこのフォルダーのレンダリング中にエラーが発生しました。',
    tryAgain: '再試行',
    loadingTree: 'ファイルツリーを読み込み中',
    loadingFiles: 'ファイルを読み込み中',
    terminalHide: 'ターミナルを非表示',
    terminalsAria: 'ターミナル',
    terminalNew: '新しいターミナル',
    terminalCloseOthers: '他を閉じる',
    terminalCloseAll: 'すべて閉じる',
    addToChat: 'チャットに追加'
  },

  preview: {
    tab: 'プレビュー',
    closeTab: label => `${label} を閉じる`,
    closeOthers: '他を閉じる',
    closeToRight: '右側を閉じる',
    closeAll: 'すべて閉じる',
    closePane: 'プレビューペインを閉じる',
    loading: 'プレビューを読み込み中',
    unavailable: 'プレビューは利用できません',
    opening: '開いています...',
    hide: '非表示',
    openPreview: 'プレビューを開く',
    openInBrowser: 'ブラウザで開く',
    linkHint: '⌘/Ctrl+クリックでプレビューペイン',
    sourceLineTitle: 'クリックして選択 · Shift クリックで拡張 · コンポーザーにドラッグ',
    source: 'ソース',
    renderedPreview: 'プレビュー',
    diff: '差分',
    unknownSize: 'サイズ不明',
    binaryTitle: 'これはバイナリファイルのようです',
    binaryBody: label => `${label} をプレビューすると読み取り不能なテキストが表示される場合があります。`,
    largeTitle: 'このファイルは大きいです',
    largeBody: (label, size) => `${label} は ${size} です。Hermes は最初の 512 KB のみを表示します。`,
    previewAnyway: 'とにかくプレビュー',
    truncated: '最初の 512 KB を表示しています。',
    noInlineTitle: 'インラインプレビューなし',
    noInlineBody: mimeType => `${mimeType || 'このファイルタイプ'} はコンテキストとして添付できます。`,
    edit: '編集',
    editing: '編集中',
    unsavedChanges: '未保存の変更',
    saveFailed: message => `保存できませんでした：${message}`,
    diskChangedTitle: 'ファイルがディスク上で変更されました',
    diskChangedBody:
      'このファイルは開いてから変更されています。あなたの版で上書きするか、編集を破棄して再読み込みしますか？',
    overwrite: '上書き',
    discardReload: '破棄して再読み込み',
    console: {
      deselect: 'エントリーの選択を解除',
      select: 'エントリーを選択',
      copyFailed: 'コンソール出力をコピーできませんでした',
      copyEntry: 'このエントリーをコピー',
      sendEntry: 'このエントリーをチャットに送信',
      messages: count => `${count} 件のコンソールメッセージ`,
      resize: 'プレビューコンソールのサイズ変更',
      title: 'プレビューコンソール',
      selected: count => `${count} 件選択`,
      sendToChat: 'チャットに送信',
      copySelected: '選択をクリップボードにコピー',
      copyAll: 'すべてをクリップボードにコピー',
      copy: 'コピー',
      clear: 'クリア',
      empty: 'コンソールメッセージはまだありません。',
      promptHeader: 'プレビューコンソール:',
      sentTitle: 'チャットに送信しました',
      sentMessage: count => `${count} 件のログエントリーがコンポーザーに追加されました`
    },
    web: {
      appFailedToBoot: 'プレビューアプリの起動に失敗しました',
      serverNotFound: 'サーバーが見つかりません',
      failedToLoad: 'プレビューの読み込みに失敗しました',
      tryAgain: '再試行',
      restarting: 'Hermes を再起動中...',
      askRestart: 'Hermes にサーバーの再起動を依頼',
      lookingRestart: taskId => `Hermes は再起動するプレビューサーバーを検索中です (${taskId})`,
      restartingTitle: 'プレビューサーバーを再起動中',
      restartingMessage: 'Hermes はバックグラウンドで作業中です。進捗はプレビューコンソールで確認してください。',
      startRestartFailed: message => `サーバー再起動を開始できませんでした: ${message}`,
      restartFailed: 'サーバーの再起動に失敗しました',
      hideConsole: 'プレビューコンソールを非表示',
      showConsole: 'プレビューコンソールを表示',
      hideDevTools: 'プレビュー DevTools を非表示',
      openDevTools: 'プレビュー DevTools を開く',
      finishedRestarting: message =>
        `Hermes がプレビューサーバーの再起動を完了しました${message ? `: ${message}` : ''}`,
      failedRestarting: message => `サーバーの再起動に失敗しました: ${message}`,
      unknownError: '不明なエラー',
      restartedTitle: 'プレビューサーバーが再起動しました',
      reloadingNow: 'プレビューを再読み込み中です。',
      restartFailedTitle: 'プレビューの再起動に失敗しました',
      restartFailedMessage: 'Hermes がサーバーを再起動できませんでした。',
      stillWorking:
        'Hermes はまだ作業中ですが、再起動の結果がまだ届いていません。サーバーコマンドがフォアグラウンドで実行されている可能性があります。',
      workspaceReloading: 'ワークスペースが変更され、プレビューを再読み込み中',
      fileChanged: url => `ファイルが変更され、プレビューを再読み込み中: ${url}`,
      filesChanged: (count, url) => `${count} 件のファイルが変更され、プレビューを再読み込み中: ${url}`,
      watchFailed: message => `プレビューファイルを監視できませんでした: ${message}`,
      moduleMimeDescription:
        'モジュールスクリプトが間違った MIME タイプで提供されています。通常、静的ファイルサーバーがプロジェクトの開発サーバーの代わりに Vite/React アプリを提供していることを意味します。',
      loadFailedConsole: (code, message) => `読み込みに失敗しました${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'プレビューページに到達できませんでした。',
      openTarget: url => `${url} を開く`,
      fallbackTitle: 'プレビュー'
    }
  },

  zones: {
    showHeader: 'ヘッダーを表示',
    hideHeader: 'ヘッダーを隠す',
    minimize: '最小化',
    restore: '復元',
    closeOthers: '他を閉じる',
    closeToRight: '右側を閉じる',
    closeAll: 'すべて閉じる',
    newSessionTab: '新しいセッションタブ',
    split: dir => `${dir}に分割`,
    move: dir => `${dir}へ移動`,
    dirUp: '上',
    dirDown: '下',
    dirLeft: '左',
    dirRight: '右',
    pluginDisabled: pluginId => `プラグイン「${pluginId}」を無効化しました`,
    pluginDisabledBody: '設定 → プラグイン で再有効化するとペインが戻ります。',
    missingPane: paneId => `ペインが見つかりません: ${paneId}`,
    editTitle: 'レイアウト',
    editHint: 'レイアウトを選ぶか、ペインをゾーン間へドラッグ。ゾーンを右クリックで分割。',
    reset: 'リセット',
    templates: 'テンプレート',
    custom: 'カスタム',
    newGridLayout: '新しいグリッドレイアウト',
    saveCurrentAs: '現在の配置をテンプレートとして保存',
    nameLayoutPlaceholder: 'レイアウト名を入力…',
    deletePreset: name => `${name} を削除`,
    zoneEditorTitle: 'ゾーンエディター',
    editorHintPre: 'クリックで分割 · ',
    editorHintPost: ' で線の向きを反転 · ゾーンをまたいでドラッグで結合 · 共有辺をドラッグでリサイズ',
    templateColumns: '列',
    templateRows: '行',
    templateGrid: 'グリッド',
    templatePriority: '優先',
    zoneTag: index => `ゾーン ${index}`,
    mergeZones: count => `${count} 個のゾーンを結合`,
    customZoneName: count => `カスタム ${count} ゾーン`,
    layoutNamePlaceholder: fallback => `レイアウト名（${fallback}）`,
    saveApply: '保存して適用',
    notExpressible: 'この配置は互いに噛み合っています（風車型）— 入れ子の分割では表現できません',
    zoneCount: count => `${count} ゾーン`
  },

  assistant: {
    thread: {
      loadingSession: 'セッションを読み込み中',
      showEarlier: '以前のメッセージを表示',
      loadingResponse: 'Hermes が応答を読み込み中',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'バックグラウンドタスクの完了後に再開します'
          : `${count} 件のバックグラウンドタスクの完了後に再開します`,
      thinking: '考え中',
      today: time => `今日 ${time}`,
      yesterday: time => `昨日 ${time}`,
      copy: 'コピー',
      refresh: '更新',
      moreActions: 'その他のアクション',
      branchNewChat: '新しいチャットでブランチ',
      dismissError: 'エラーを閉じる',
      readAloudFailed: '読み上げに失敗しました',
      preparingAudio: '音声を準備中...',
      stopReading: '読み上げを停止',
      readAloud: '読み上げ',
      editMessage: 'メッセージを編集',
      stop: '停止',
      restorePrevious: '前のチェックポイントに戻す',
      restoreCheckpoint: 'チェックポイントを復元',
      restoreFromHere: 'チェックポイントを復元 — このプロンプトから再実行',
      restoreTitle: 'このチェックポイントに復元しますか？',
      restoreBody: 'このプロンプト以降のメッセージは会話から削除され、ここからプロンプトが再実行されます。',
      restoreConfirm: '復元して再実行',
      restoreNext: '次のチェックポイントに戻す',
      goForward: '進む',
      sendEdited: '編集済みメッセージを送信',
      attachingFile: '添付中…'
    },
    approval: {
      gatewayDisconnected: 'Hermes ゲートウェイが接続されていません',
      sendFailed: '承認応答を送信できませんでした',
      run: '実行',
      command: 'コマンド',
      moreOptions: 'その他の承認オプション',
      allowSession: 'このセッションで許可',
      alwaysAllowMenu: '常に許可…',
      jumpToApproval: '承認が必要',
      reject: '拒否',
      alwaysTitle: 'このコマンドを常に許可しますか？',
      alwaysDescription: pattern =>
        `これにより "${pattern}" パターンが永続的な許可リスト (~/.hermes/config.yaml) に追加されます。Hermes はこのセッションや将来のセッションで、このようなコマンドについて再度尋ねません。`,
      alwaysAllow: '常に許可'
    },
    clarify: {
      notReady: '明確化リクエストはまだ準備できていません',
      gatewayDisconnected: 'Hermes ゲートウェイが接続されていません',
      sendFailed: '明確化応答を送信できませんでした',
      loadingQuestion: '質問を読み込み中…',
      other: 'その他（回答を入力）',
      placeholder: '回答を入力…',
      skip: 'スキップ',
      skipped: 'スキップ済み',
      continueLabel: '続行',
      lateAnswer: (question, choice) => `「${question}」について — 私の回答: ${choice}`,
      lateAnswerTip: 'この回答をフォローアップメッセージとして下書きします',
      lateAnswerHint: 'この質問はもう回答を待っていません。選択肢を選ぶとフォローアップメッセージとして下書きされます。'
    },
    tool: {
      code: 'コード',
      copyCode: 'コードをコピー',
      renderingImage: '画像をレンダリング中',
      copyOutput: '出力をコピー',
      copyCommand: 'コマンドをコピー',
      copyContent: 'コンテンツをコピー',
      copyUrl: 'URL をコピー',
      copyResults: '結果をコピー',
      copyQuery: 'クエリをコピー',
      copyFile: 'ファイルをコピー',
      copyPath: 'パスをコピー',
      outputAlt: 'ツール出力',
      rawResponse: '生の応答',
      copyActivity: 'アクティビティをコピー',
      recoveredOne: '1 つの失敗したステップの後に回復しました',
      recoveredMany: count => `${count} つの失敗したステップの後に回復しました`,
      failedOne: '1 つのステップが失敗しました',
      failedMany: count => `${count} つのステップが失敗しました`,
      statusRunning: '実行中',
      statusError: 'エラー',
      statusRecovered: '回復しました',
      statusDone: '完了',
      actions: {
        read: '読み取り完了',
        reading: '読み取り中',
        opened: 'オープン済み',
        opening: 'オープン中',
        failedToOpen: 'オープン失敗',
        searched: '検索完了',
        searching: '検索中',
        ran: '実行完了',
        running: '実行中',
        ranCode: 'コード実行完了',
        runningCode: 'スクリプト作成中'
      },
      prefixes: {
        browser: 'ブラウザー',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `「${value}」を${action}`,
        actionTarget: (action, target) => `${target} を${action}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `${prefix} ${action}を実行中`,
        runningTool: action => `${action}を実行中`
      },
      titles: {
        browser_click: {
          done: 'ページ要素をクリックしました',
          pending: 'ページ要素をクリック中',
          pendingAction: 'クリック中'
        },
        browser_fill: { done: 'フォーム欄に入力しました', pending: 'フォーム欄に入力中', pendingAction: '入力中' },
        browser_navigate: { done: 'ページを開きました', pending: 'ページをオープン中', pendingAction: 'オープン中' },
        browser_snapshot: {
          done: 'ページスナップショットを取得しました',
          pending: 'ページスナップショットを取得中',
          pendingAction: '取得中'
        },
        browser_take_screenshot: {
          done: 'スクリーンショットを取得しました',
          pending: 'スクリーンショットを取得中',
          pendingAction: '取得中'
        },
        browser_type: { done: 'ページに入力しました', pending: 'ページに入力中', pendingAction: '入力中' },
        clarify: { done: '質問しました', pending: '質問中', pendingAction: '質問中' },
        cronjob: { done: 'Cron ジョブ', pending: 'Cron ジョブをスケジュール中', pendingAction: 'スケジュール中' },
        edit_file: { done: 'ファイルを編集しました', pending: 'ファイルを編集中', pendingAction: '編集中' },
        execute_code: { done: 'コードを実行しました', pending: 'スクリプト作成中', pendingAction: 'スクリプト作成中' },
        image_generate: { done: '画像を生成しました', pending: '画像を生成中', pendingAction: '生成中' },
        list_files: {
          done: 'ファイルを一覧表示しました',
          pending: 'ファイルを一覧表示中',
          pendingAction: '一覧表示中'
        },
        patch: {
          done: 'ファイルにパッチを適用しました',
          pending: 'ファイルにパッチ適用中',
          pendingAction: 'パッチ適用中'
        },
        read_file: { done: 'ファイルを読み取りました', pending: 'ファイルを読み取り中', pendingAction: '読み取り中' },
        search_files: { done: 'ファイルを検索しました', pending: 'ファイルを検索中', pendingAction: '検索中' },
        session_search_recall: {
          done: 'セッション履歴を検索しました',
          pending: 'セッション履歴を検索中',
          pendingAction: '検索中'
        },
        terminal: { done: 'コマンドを実行しました', pending: 'コマンドを実行中', pendingAction: '実行中' },
        todo: { done: 'Todo を更新しました', pending: 'Todo を更新中', pendingAction: '更新中' },
        vision_analyze: { done: '画像を分析しました', pending: '画像を分析中', pendingAction: '分析中' },
        web_extract: {
          done: 'Web ページを読み取りました',
          pending: 'Web ページを読み取り中',
          pendingAction: '読み取り中'
        },
        web_search: { done: 'Web を検索しました', pending: 'Web を検索中', pendingAction: '検索中' },
        write_file: { done: 'ファイルを編集しました', pending: 'ファイルを編集中', pendingAction: '編集中' }
      }
    }
  },

  prompts: {
    gatewayDisconnected: 'Hermes ゲートウェイが接続されていません',
    sudoSendFailed: 'sudo パスワードを送信できませんでした',
    secretSendFailed: 'シークレットを送信できませんでした',
    sudoTitle: '管理者パスワード',
    sudoDesc:
      'Hermes は特権コマンドを実行するために sudo パスワードが必要です。ローカルエージェントにのみ送信されます。',
    sudoPlaceholder: 'sudo パスワード',
    secretTitle: 'シークレットが必要です',
    secretDesc: 'Hermes は続行するための認証情報が必要です。',
    secretPlaceholder: 'シークレット値'
  },

  desktop: {
    audioReadFailed: '録音した音声を読み取れませんでした',
    sessionUnavailable: 'セッションが利用できません',
    createSessionFailed: '新しいセッションを作成できませんでした',
    promptFailed: 'プロンプトに失敗しました',
    providerCredentialRequired: '最初のメッセージを送信する前にプロバイダー認証情報を追加してください。',
    emptySlashCommand: '空のスラッシュコマンド',
    desktopCommands: 'デスクトップコマンド',
    skillCommandsAvailable: count => `${count} 件のスキルコマンドが利用可能です。`,
    warningLine: message => `警告: ${message}`,
    yoloArmed: 'このチャットでは YOLO が有効になっています',
    yoloOff: 'YOLO オフ',
    yoloSystem: active => `このセッションの YOLO ${active ? 'オン' : 'オフ'}`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'YOLO を切り替えられませんでした',
    profileStatus: current =>
      `プロファイル: ${current}。/profile <name> または「新しいセッション」ピッカーを使って別のプロファイルでチャットを始めてください。`,
    unknownProfile: '不明なプロファイル',
    noProfileNamed: (target, available) => `"${target}" という名前のプロファイルはありません。利用可能: ${available}`,
    newChatsProfile: name => `新しいチャットはプロファイル ${name} を使用します。`,
    setProfileFailed: 'プロファイルの設定に失敗しました',
    sttDisabled: '音声認識は設定で無効になっています。',
    stopFailed: '停止に失敗しました',
    regenerateFailed: '再生成に失敗しました',
    editFailed: '編集に失敗しました',
    resumeFailed: '再開に失敗しました',
    resumeStrandedTitle: 'このセッションを読み込めませんでした',
    resumeStrandedBody:
      'このセッションへの接続に失敗し、自動再試行も停止しました。ゲートウェイが実行中か確認してから、もう一度お試しください。',
    resumeRetry: '再試行',
    nothingToBranch: 'ブランチするものがありません',
    branchNeedsChat: 'ブランチする前にチャットを開始または再開してください。',
    sessionBusy: 'セッションが使用中',
    branchStopCurrent: 'このチャットをブランチする前に現在のターンを停止してください。',
    branchNoText: 'このメッセージにはブランチするテキストがありません。',
    branchTitle: n => `下書き: ブランチ #${n}`,
    branchFailed: 'ブランチに失敗しました',
    deleteFailed: '削除に失敗しました',
    archived: 'アーカイブしました',
    archiveFailed: 'アーカイブに失敗しました',
    cwdChangeFailed: '作業ディレクトリの変更に失敗しました',
    cwdStagedTitle: '作業ディレクトリがステージングされました',
    cwdStagedMessage:
      'このアクティブなセッションへの cwd の変更を適用するにはデスクトップバックエンドを再起動してください。',
    modelSwitchFailed: 'モデルの切り替えに失敗しました',
    sessionExported: 'セッションをエクスポートしました',
    sessionExportFailed: 'セッションをエクスポートできませんでした',
    imageSaved: '画像を保存しました',
    downloadStarted: 'ダウンロードを開始しました',
    restartToUseSaveImage: '画像を保存するには Hermes Desktop を再起動してください。',
    restartToSaveImages: '画像を保存するには Hermes Desktop を再起動してください',
    imageDownloadFailed: '画像のダウンロードに失敗しました',
    openImage: '画像を開く',
    downloadImage: '画像をダウンロード',
    savingImage: '画像を保存中',
    imagePreviewFailed: '画像のプレビューに失敗しました',
    imageAttach: '画像を添付',
    imageWriteFailed: '画像のディスクへの書き込みに失敗しました。',
    imageAttachFailed: '画像の添付に失敗しました',
    attachImages: '画像を添付',
    clipboard: 'クリップボード',
    noClipboardImage: 'クリップボードに画像が見つかりません',
    clipboardPasteFailed: 'クリップボードからの貼り付けに失敗しました',
    dropFiles: 'ファイルをドロップ',
    handoff: {
      pickPlatform: '送信先を選択',
      success: platform => `${platform} に引き継ぎました。いつでもここで再開できます。`,
      systemNote: platform => `↻ ${platform} に引き継ぎました — いつでもここで再開できます。`,
      failed: error => `引き継ぎに失敗しました: ${error}`,
      timedOut: 'ゲートウェイの待機がタイムアウトしました。`hermes gateway` は起動していますか？'
    }
  },

  errors: {
    genericFailure: '問題が発生しました',
    boundaryTitle: 'インターフェイスで問題が発生しました',
    boundaryDesc: 'ビューで予期しないエラーが発生しました。チャットと設定は安全です。',
    reloadWindow: 'ウィンドウを再読み込み',
    openLogs: 'ログを開く'
  },

  ui: {
    search: {
      clear: '検索をクリア'
    },
    pagination: {
      label: 'ページング',
      previous: '前へ',
      previousAria: '前のページへ',
      next: '次へ',
      nextAria: '次のページへ'
    },
    sidebar: {
      title: 'サイドバー',
      description: 'モバイルサイドバーを表示します。',
      toggle: 'サイドバーを切り替え'
    }
  }
})
