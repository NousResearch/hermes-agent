import { defineFieldCopy } from '@/app/settings/field-copy'

import type { TranslationOverrides } from './define-locale'

export const jaSettings = {
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
    plugins: {
      installModal: {
        installFromGit: 'Git からインストール',
        reviewRepository: 'リポジトリを確認',
        repoPlaceholder: 'https://github.com/owner/repo'
      }
    },
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
      providerLocalModels: 'ローカルモデル',
      gateway: 'ゲートウェイ',
      apiKeys: 'ツールとキー',
      keybinds: 'キーボードショートカット',
      keysTools: 'ツール',
      keysSettings: '設定',
      mcp: 'MCP',
      archivedChats: 'アーカイブ済みチャット',
      about: '情報',
      billing: '請求',
      notifications: '通知',
      vault: 'パスワードとログイン'
    },
    vault: {
      title: 'パスワードとログイン',
      blurb:
        '「GitHub にログインして」と言えば、エージェントが代わりにサインインします。初めてサインインページに出会ったときにその場でログイン情報を尋ね、以降は自動で処理します。パスワードはこのマシン上で暗号化され、ページに直接入力されます。モデルは一切見ません。',
      count: n => `${n} 件保存済み`,
      loadFailed: 'ボールト項目を読み込めませんでした',
      empty: 'まだ何も保存されていません',
      emptyDesc:
        'ここで何かを追加する必要はありません。エージェントにサイトへのサインインを頼むと、その場で一度だけログイン情報を尋ねます。事前に登録したい場合は「追加」を使ってください。',
      add: '追加',
      addTitle: 'ログイン情報・カード・住所を追加',
      addDescription: 'このマシン上に暗号化して保存されます。エージェントがパスワードを見ることはありません。',
      added: '保存しました。',
      adding: '保存中…',
      addConfirm: '保存',
      kindField: '種類',
      kinds: { login: 'ログイン', payment: '支払いカード', address: '住所' },
      labelField: 'ラベル',
      labelPlaceholder: '例: GitHub 仕事用アカウント',
      labelRequired: 'ラベルは必須です。',
      originField: 'サイトのオリジン',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: 'https://example.com のような有効な URL を入力してください。',
      identifierTypeField: '識別子の種類',
      identifierTypes: { email: 'メール', phone: '電話番号', username: 'ユーザー名' },
      identifierField: '識別子',
      identifierShown: identifier => identifier,
      passwordField: 'パスワード',
      loginFieldsRequired: '識別子とパスワードは必須です。',
      cardNumberField: 'カード番号',
      cardNameField: 'カード名義',
      expMonthField: '有効期限（月）',
      expYearField: '有効期限（年）',
      cvcField: 'CVC',
      postalField: '郵便番号',
      addressLine1Field: '住所 1 行目',
      addressLine2Field: '住所 2 行目',
      cityField: '市区町村',
      stateField: '都道府県 / 地域',
      countryField: '国',
      optional: '（任意）',
      createdOn: date => `追加日 ${date}`,
      deleteAction: '保存済み項目を削除',
      otpField: '認証キー',
      otpPlaceholder: 'Base32 シークレットまたは otpauth:// リンク',
      otpHint: '2FA を有効にするときにサイトが表示する「セットアップキー」。保存すると Hermes がコードを生成します。',
      twoFactorBadge: '2FA 自動',
      deleteTitle: 'この項目を削除しますか？',
      deleteDescription: label => `「${label}」は暗号化ボールトから削除されます。元に戻せません。`,
      deleteConfirm: '削除',
      sources: {
        title: 'パスワードマネージャー',
        blurb:
          'インストール済みのパスワードマネージャーは自動的に検出されます。エージェントがそこからログイン情報を初めて必要とするときにロック解除を求めます（セッションごとに一度）。メモリに残るのはセッショントークンのみで、エージェントはマスターパスワードやログイン情報を一切見ません。',
        toggleFailed: 'パスワードマネージャーの設定を更新できませんでした',
        notInstalled: name =>
          `未検出です。${name} のコマンドラインツールをインストールしてサインインすると、Hermes が自動的に検出します。`,
        disabledDesc: '検出済みですが、Hermes では無効になっています。',
        lockedDesc:
          '検出済み。エージェントがログイン情報を必要とするときにロック解除を求めます。今すぐ解除することもできます。',
        unlockedDesc: 'このセッションでロック解除済み。30分間操作がないか Hermes を閉じると自動的にロックされます。',
        statusLocked: 'ロック中',
        statusNotDetected: '未検出',
        statusOff: 'オフ',
        statusUnlocked: 'ロック解除済み',
        unlock: 'ロック解除',
        unlocking: 'ロック解除中…',
        lock: 'ロック',
        unlocked: name => `${name} をこのセッションでロック解除しました。`,
        unlockTitle: name => `${name} のロックを解除`,
        unlockDescription:
          'マスターパスワードを入力してください。このマシン上のパスワードマネージャーに渡された後に破棄され、保存・記録されることも、エージェントに表示されることもありません。',
        masterPasswordPlaceholder: 'マスターパスワード'
      }
    },
    notifications: {
      title: '通知',
      intro: 'アプリ内トーストとは別の、ネイティブのデスクトップ通知です。設定は端末ごとに保存されます。',
      enableAll: '通知を有効にする',
      enableAllDesc: 'オフで以下の通知をすべて無効にします。',
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
          description: 'バックグラウンドのターンエラー。'
        },
        backgroundDone: {
          label: 'バックグラウンドタスク完了',
          description: 'バックグラウンドのターミナルコマンドが完了しました。'
        },
        credits: {
          label: 'クレジット通知',
          description: 'クレジットの利用が停止または復旧しました。'
        },
        plugin: {
          label: 'プラグイン通知',
          description: 'Hermes がバックグラウンドの間に、デスクトッププラグインが通知を送信しました。'
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
      reasoningCollapsedTitle: '思考ブロックをデフォルトで折りたたむ',
      reasoningCollapsedDesc: 'ストリーミング中の推論を、開くまで折りたたんだまま利用できるようにします。',
      uiScaleTitle: 'UI スケール',
      uiScaleDesc: (percent: number) =>
        `アプリ全体の文字と UI を拡大縮小します。Cmd/Ctrl と +、-、0 でも変更できます。現在: ${percent}%`,
      sessionDensityTitle: 'セッションリストの密度',
      sessionDensityDesc: 'サイドバーのセッションタイトルの下に表示する情報量を選びます。',
      sessionDensityCompact: 'コンパクト',
      sessionDensityComfortable: '標準',
      sessionDensityDetailed: '詳細',
      tabStripTitle: 'タブバー',
      tabStripDesc: 'ゾーンの上にタブを表示します。自動では、他にチャットやタイルのゾーンがない限り、ペインが1つのときに隠します。',
      tabStripAuto: '自動',
      tabStripAlways: '常に表示',
      tabStripNever: '表示しない',
      appActionsTitle: 'アプリ操作',
      appActionsDesc:
        '設定・レイアウト・HUD をタイトルバーの左右どちらに置くか。右にするとタブ用のスペースが左に残ります。',
      appActionsLeft: '左',
      appActionsRight: '右',
      terminalFontTitle: 'ターミナルフォント',
      terminalFontDesc:
        'Desktop のターミナルで使用するインストール済みフォントを選びます。Nerd Font は Powerlevel10k とシェルアイコンを表示できます。空欄では内蔵の JetBrains Mono を使用します。',
      terminalFontPlaceholder: 'MesloLGS NF または CSS フォントスタック',
      terminalFontPreview: 'グリフのプレビュー',
      terminalFontReset: '既定値を使用',
      chatFontTitle: 'チャットフォント',
      chatFontDesc:
        'チャットとアプリ全体に使うインストール済みフォントを選択します。OpenDyslexic などの読みやすいフォントに便利です。空欄ならテーマのフォントを使います。',
      chatFontPlaceholder: 'OpenDyslexic または CSS フォントスタック',
      chatFontPreview: 'プレビュー',
      chatFontSample: 'いろはにほへと ちりぬるを 0123456789',
      chatFontReset: 'テーマのフォントを使用',
      translucencyTitle: 'ウィンドウの透過',
      translucencyDesc: 'テキストも含めウィンドウ全体を透過させてデスクトップを表示します。',
      translucencyGlassDesc: 'マットガラス: デスクトップが滑らかなぼかしとして透け、テキストは鮮明なまま。',
      translucencyModeClear: 'クリア',
      translucencyModeGlass: 'ガラス',
      translucencyTintTitle: '色味',
      translucencyFadeTitle: 'フェード',
      translucencyFrostTitle: 'くもりの質感',
      translucencyFrost: {
        'under-window': '深い',
        popover: 'やわらか',
        titlebar: '明るい',
        header: 'まぶしい'
      },
      translucencyScopeTitle: '適用範囲',
      translucencyScope: {
        window: 'ウィンドウ全体',
        sidebar: 'サイドバーのみ'
      },
      backdropTitle: 'チャット背景',
      backdropDesc: '会話の背後に表示される淡い彫像の画像。',
      userBubbleTitle: 'メッセージの吹き出し',
      userBubbleDesc: '自分のメッセージの透け具合。0 で不透明、100 で枠線だけが残ります。',
      introSplashTitle: 'イントロ表示',
      introSplashDesc: '空のチャットに表示されるワードマークとプロンプト。',
      reactionsTitle: 'メッセージリアクション',
      reactionsDesc:
        'iMessage風の絵文字タップバック — メッセージにリアクションでき、Hermesもあなたのメッセージにリアクションします。',
      tipsTitle: 'アプリ内ヒント',
      tipsDesc:
        'アプリや Hermes からのヒントをときどき表示します。各ヒントは一度だけ表示されます。利用開始から30日後に自動でオフになりますが、再びオンにできます。',
      tipsReset: (count: number) => `${count}件のヒントをもう一度表示`,
      toursTitle: 'ガイドツアー',
      toursDesc:
        '各ステップを強調しながら、Hermes がアプリを案内します。利用開始から30日後に自動でオフになりますが、再びオンにできます。',
      composerPopoutTitle: 'フローティング入力欄',
      composerPopoutDesc: '入力欄をドックからドラッグして外せるようにします。オフにすると画面下部に固定されます。',
      vibeHeartsTitle: 'バイブハート',
      vibeHeartsDesc:
        'ありがとう・愛してる・good bot・ハート絵文字のときに浮かぶハート。上のメッセージリアクションとは別です。',
      embedsTitle: 'インライン埋め込み',
      embedsDesc:
        'リッチプレビューは第三者サイト（YouTube、X など）から読み込まれます。確認は許可するまでプレースホルダーを表示し、常には自動で読み込み、オフはリンクのままにします。',
      embedsAsk: '確認',
      embedsAlways: '常に',
      embedsOff: 'オフ',
      embedsReset: (count: number) => `許可した${count}件のサービスをリセット`,
      resumeLastSessionTitle: '起動時に前回のチャットを再開',
      resumeLastSessionDesc:
        'オンの場合、コールドスタート時に直近のチャットを再び開きます。オフにすると常に新しいチャットから始まります。',
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
      modelContextLength: 'メインのチャットモデルのみ、検出されたコンテキストウィンドウを上書きします（トークン数）。0 のままにすると、選択したモデルから検出された値を使用します。補助モデル/MoA モデルには影響しません。',
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
        codexGpt55Autoraise: 'Codex 圧縮の自動引き上げ',
        targetRatio: '圧縮目標',
        protectLastN: '保護する直近メッセージ'
      },
      auxiliary: {
        compression: {
          timeout: '圧縮モデルのタイムアウト（秒）'
        }
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
        enabled: '会話が大きくなったとき、古いコンテキストを要約します。',
        codexGpt55Autoraise: '対応する ChatGPT Codex OAuth モデルの圧縮しきい値を 85% に引き上げます。'
      },
      auxiliary: {
        compression: {
          timeout: '補助圧縮モデルの呼び出しごとに待機する秒数（既定 120）。遅いローカルモデルでは値を上げてください。'
        }
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
      bundleOutOfSync: 'アプリのビルドが古くなっています',
      bundleOutOfSyncDesc:
        'Hermes ランタイムは更新されましたが、デスクトップアプリ自体は古いビルドのままです。アプリを更新するまで、新しいインターフェース機能(Bot Mode など)は表示されません。下の更新を実行してアプリを再ビルドしてください。それでもこの警告が消えない場合は、最新のデスクトップインストーラーから再インストールしてください。',
      bundleOutOfSyncAction: 'インストーラーを入手',
      bundleSwapPending: '再起動して更新を完了',
      bundleSwapPendingDesc:
        '更新されたアプリはすでにインストール済みです。Hermes を再起動するだけで新しいビルドが読み込まれます。チャットや設定はそのまま保持されます。',
      bundleSwapPendingAction: 'Hermes を再起動',
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
      updateReadyUnknown: '新しい更新の準備ができました。',
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
      searchPlaceholder: '検索…',
      noResults: '結果が見つかりません',
      systemDefault: 'システムのデフォルト',
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
    screenshot: {
      enabledTitle: 'スクリーンショットのショートカット',
      enabledDesc:
        'どのアプリからでも左右の Command キーを同時に押すと、最前面のウインドウを撮影し、Hermes の現在の下書きに添付します。自動送信はしません。初期設定はオフで、この Mac にのみ適用されます。機密情報が写る可能性があるため、送信前に添付画像を確認してください。',
      statusTitle: 'スクリーンショットのショートカットの状態',
      checking: 'スクリーンショットのショートカットを確認中…',
      disabled: 'スクリーンショットのショートカットはオフです。',
      starting: 'ショートカットの検出を開始しています。まだ使用できません。',
      ready: 'ショートカットを使用できます。撮影した画像は現在の下書きに添付され、送信はされません。',
      inputPermission:
        '入力監視の許可により、他のアプリがアクティブな間も両方の Command キーを検出できます。システム設定 → プライバシーとセキュリティ → 入力監視で Hermes を許可し、ここに戻って再試行してください。',
      screenPermission:
        '画面収録の許可により、このショートカットを使ったときに最前面のアプリのウインドウを撮影できます。システム設定 → プライバシーとセキュリティ → 画面収録で Hermes を許可し、ここに戻って再試行してください。macOS に求められた場合は Hermes を再起動してください。',
      openSettings: 'システム設定を開く',
      retry: '再試行',
      unavailable: 'スクリーンショットのショートカットは使用できません。再試行するか、オフにしてください。',
      errorTitle: 'スクリーンショットのショートカットのエラー',
      loadFailed: 'ショートカットの状態を読み取れませんでした。再試行して現在の設定を確認してください。',
      saveFailed: 'ショートカットの変更を確認できませんでした。再試行して現在の設定を確認してください。',
      permissionFailed: 'システム設定を開けませんでした。プライバシーとセキュリティを手動で開き、再試行してください。',
      captureFailed: '最前面のウインドウを撮影できませんでした。添付も送信もされていません。',
      contextChanged: '撮影中に現在の下書きが変わりました。画像は添付も送信もされていません。'
    },
    quickEntry: {
      enabledTitle: 'クイック入力',
      enabledDesc:
        'グローバルショートカットで小さな入力欄をどこからでも呼び出し、Hermes を開かずにプロンプトを送信します。',
      shortcutTitle: 'クイック入力のショートカット',
      shortcutDesc: '修飾キーが 1 つ以上必要です（例: CommandOrControl+Shift+Space）。',
      active: 'ショートカットは有効です。',
      takenBy: 'このショートカットは他のアプリが使用しています。別のものを選んでください。',
      invalidShortcut: '有効なショートカットではありません。修飾キーを 1 つ以上含めてください。'
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
      actions: 'アクション',

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
        'Hermes Desktop はデフォルトで独自のローカルゲートウェイを起動します。別のマシンや信頼できるプロキシの背後で既に動作している Hermes バックエンドをこのアプリで制御する場合は、リモートゲートウェイを使用してください。ゲートウェイ接続はマシン単位の設定で、プロファイルは接続したゲートウェイから検出されます。',
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
      plainTextConfirmTitle: 'ゲートウェイトークンを平文で保存しますか？',
      plainTextConfirmDesc:
        'このマシンで OS のキーリングサービスが見つからなかったため、トークンはアプリの接続設定ファイルに暗号化されずに保存され、このユーザーとして実行される任意のプロセスから読み取れる状態になります。暗号化して保存するには、GNOME Keyring または KWallet をインストールまたは有効化してください。',
      plainTextConfirmAction: '平文で保存',
      plainTextStoredTitle: 'トークンは平文で保存されています',
      plainTextStoredDesc:
        'セキュアストレージが利用できないため、保存済みのトークンはこのマシンのアプリの接続設定ファイルに暗号化されずに保存されています。暗号化するには GNOME Keyring または KWallet をインストールまたは有効化してください。',
      keychainEncryptionTitle: 'OS キーチェーンで保存済みのシークレットを暗号化',
      keychainEncryptionDesc:
        'デフォルトはオフです。オンにすると、ゲートウェイのトークンとサインイン資格情報がシステムのキーチェーン（Keychain Access、GNOME Keyring、Windows DPAPI）で暗号化されます。システムから許可やパスワードを求められる場合があります。オフの場合は、現在のユーザーのみが読める通常ファイルとして保存されます。',
      keychainEncryptionFailed: 'シークレット暗号化の設定を変更できませんでした',
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
    search: {
      placeholder: 'すべての設定を検索...',
      pill: '検索'
    },
    profileScope: {
      appliesTo: '適用対象',
      editsProfile: profile => `このページの変更は「${profile}」プロファイルに適用されます。`
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
      costTokens: tokens => `1 呼び出しあたり約 ${tokens} トークン`,
      usage30d: uses => `過去 30 日で ${uses} 回使用`,
      unusedPill: '未使用',
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
      noOutput: 'まだ出力がありません。',
      deepLinkTitle: 'MCP サーバーを追加しますか？',
      deepLinkDescription:
        'リンクがこの MCP サーバーを Hermes に追加するよう要求しました。下の設定はリンク側から来たものです。内容を必ず確認してください。',
      deepLinkStdioWarning:
        'このサーバーは下記のコマンドでローカルプロセスを実行します。提供元を信頼できる場合のみ続行してください。',
      deepLinkConfirm: 'サーバーを追加',
      deepLinkNameInvalid: '名前は 1〜64 文字の英数字、ドット、ハイフン、アンダースコアです。',
      deepLinkNameConflict: name =>
        `${name} という名前のサーバーは既に存在します。別の名前にするかキャンセルしてください。`,
      deepLinkErrorTitle: 'MCP インストールリンクを拒否しました',
      deepLinkErrorName: 'リンクのサーバー名が欠落しているか無効です。',
      deepLinkErrorConfig: 'リンクの設定が有効な base64 エンコード JSON ではありません。',
      deepLinkErrorShape:
        '設定は文字列の `url` または `command` フィールドを持つ JSON オブジェクトである必要があります。',
      deepLinkErrorUrl: 'サーバー URL は http:// と https:// のみ許可されます。',
      deepLinkErrorTooLarge: '設定ペイロードが 32KB の上限を超えています。',
      importButton: 'インポート',
      importPlaceholder: 'mcp.json スニペット、npx/docker コマンド、claude mcp add 行、URL、Cursor リンクを貼り付け…',
      importNoMatch: '貼り付けたテキストからサーバー設定を認識できませんでした。',
      importConfirm: 'mcp.json に追加',
      importConfirmMany: count => `${count} 件のサーバーを mcp.json に追加`
    },
    model: {
      loading: 'モデル設定を読み込み中...',
      appliesDesc:
        '新しいセッションに適用されます。コンポーザーのモデルピッカーを使ってアクティブなチャットをホットスワップできます。',
      provider: 'プロバイダー',
      model: 'モデル',
      applying: '適用中...',
      loadFailed: 'モデルを読み込めませんでした',
      restartRequired:
        'アップデート後、このバックエンドは古いコードのままです。再起動して新しいコードを読み込んでください。',
      restartBackend: 'バックエンドを再起動',
      restartingBackend: 'バックエンドを再起動中...',
      restartFailed: 'バックエンドを再起動できませんでした',
      auxiliaryTitle: '補助モデル',
      resetAllToMain: 'すべてメインにリセット',
      auxiliaryDesc:
        'ヘルパータスクはデフォルトでメインモデルで実行されます。タスクに専用モデルを割り当てることでオーバーライドできます。',
      setToMain: 'メインに設定',
      change: '変更',
      autoUseMain: '自動 · メインモデルを使用',
      inheritMainEffort: '継承 · メインモデルの推論強度',
      providerDefault: '(プロバイダーのデフォルト)',
      tasks: {
        vision: { label: 'ビジョン', hint: '画像分析' },
        compression: { label: '圧縮', hint: 'コンテキストの圧縮' },
        skills_hub: { label: 'スキルハブ', hint: 'スキル検索' },
        approval: { label: '承認', hint: 'スマート自動承認' },
        mcp: { label: 'MCP', hint: 'MCP ツールルーティング' },
        title_generation: { label: 'タイトル生成', hint: 'セッションタイトル' },
        review: { label: 'レビュー', hint: '/review レビューサブエージェント' },
        triage_specifier: { label: 'トリアージ指定', hint: 'カンバン仕様の具体化' },
        kanban_decomposer: { label: 'カンバン分解', hint: 'タスク分解' },
        profile_describer: { label: 'プロファイル記述', hint: 'プロファイル概要の自動生成' },
        curator: { label: 'キュレーター', hint: 'スキル使用レビュー' }
      }
    },
    localModels: {
      title: 'ローカルモデル',
      runtimeTitle: 'ローカルランタイム',
      runtimeReady: backend => `準備完了 · ${backend}`,
      serverRunning: '実行中',
      runtimeInstalled: 'llama.cpp ランタイムをインストール済み',
      runtimeInstalledDetail: (tag, backend) =>
        `ビルド ${tag}、${backend} バックエンド。サーバーは Hermes が起動・管理します。`,
      installTitle: 'ローカルランタイムをインストール',
      installDetail:
        'llama.cpp 推論エンジン（数百 MB）をダウンロードします。ダウンロードしたモデルはすべてこのマシン上で動作します——アカウント不要、データが外部に送られることはありません。',
      installAction: 'ランタイムをインストール',
      installing: 'ランタイムをインストール中…',
      installFailed: 'ランタイムのインストールに失敗しました',
      hardwareTitle: 'このマシン',
      hardwareLoading: 'ハードウェアを確認中…',
      vram: label => `GPU メモリ ${label}`,
      ram: label => `RAM ${label}`,
      unifiedMemory: 'ユニファイドメモリ',
      modelsTitle: 'モデル',
      recommended: 'おすすめ',
      recommendedReason: {
        'best-quality-resident':
          'GPU に完全に載り、フルスピードで動くモデルの中で最高品質です。おすすめは品質とこのハードウェアでの予測速度を両立させて選ばれます。',
        'speed-gated-quality':
          'より高品質なモデルもこのマシンに載りますが、メモリ帯域の制約で応答が遅くなります — これは速度を保てる最良のモデルです。',
        'fastest-resident':
          'このハードウェアでフルスピードに達するモデルはありません。GPU メモリ内で動くものの中で最速です。'
      } as Record<string, string>,
      noRecommendationTitle: 'このマシン向けの自動推奨モデルはありません',
      noRecommendationDetail:
        '自動セットアップには、GPU メモリまたはユニファイドメモリに完全に収まる厳選モデルが必要です。下の一覧から選ぶか、ほかのモデルを探すこともできます。',
      noRecommendationAction: 'モデルを探す',
      quickstartConfigure: '自分で選ぶ',
      downloaded: 'ダウンロード済み',
      downloadAction: size => `ダウンロード · ${size}`,
      downloadProgress: (done, total) => `ダウンロード中 ${done} / ${total}`,
      downloadDoneToast: model => `${model} の準備ができました。`,
      installDoneToast: 'ローカルランタイムのインストールが完了しました。',
      useAction: '使用する',
      activePill: 'デフォルト',
      updateTitle: 'エンジンの更新があります',
      updateDetail: (next, current) =>
        `新しい llama.cpp ビルド（${next}）をインストールできます——現在は ${current} です。ダウンロード中もモデルは引き続き使えます。`,
      updateAction: 'エンジンを更新',
      updating: 'エンジンを更新中…',
      upToDateTitle: 'エンジンは最新です',
      upToDateDetail: (tag, backend) => `llama.cpp ${tag}（${backend}）で動作中——設定されたビルドです。`,
      activeDetail: '新しいチャットはこのモデルを使用——最初のメッセージ送信時に読み込みます',
      activeNotLoaded: '最初のメッセージで読み込みます',
      loadedPill: '読み込み済み',
      placementResident: 'すべて GPU 上',
      placementSpilled: '一部 RAM 上',
      placementResidentTip: 'このコンテキストウィンドウで GPU メモリ内で完全に動作しています — フルスピード。',
      placementSpilledTip:
        'モデルの一部がシステム RAM から動作しています — 動作しますが遅くなります。よりコンパクトなビルドか小さいコンテキストなら完全に収まります。',
      loadingPill: '読み込み中…',
      ejectTip: 'GPU メモリを解放（必要時に再読み込み）',
      ejected: 'モデルをアンロードしました——GPU メモリを解放しました。',
      ejectFailed: 'モデルをアンロードできませんでした',
      stopServer: 'オフにする',
      startServer: 'オンにする',
      runtimeRunningDetail:
        'ローカルサーバーが実行中です。オフにすると GPU メモリを全て解放し、再度オンにするまで新しいチャットはローカルモデルを使用しません。',
      serverStopped: 'ローカルサーバーを停止しました——GPU メモリを解放しました。',
      serverStarted: 'ローカルサーバー実行中。',
      serverStopFailed: 'ローカルサーバーを停止できませんでした',
      serverStartFailed: 'ローカルサーバーを起動できませんでした',
      activating: '起動中…',
      activateFailed: model => `${model} への切り替えに失敗しました`,
      activateDoneToast: model => `新しいチャットは ${model} を使用します。`,
      downloadFailed: model => `${model} のダウンロードに失敗しました`,
      pillFitsGpu: 'GPU に完全に収まります',
      pillUsesRam: 'システム RAM を使用',
      pillTooBig: 'このマシンには大きすぎます',
      browseTitle: 'さらにモデルを探す',
      browseHint:
        'Hugging Face 全体を検索できます。ここでダウンロードしたモデルは自動でマシンに合わせて動作しますが、当方でのテストは行われていません。',
      browsePlaceholder: 'モデル名または作者で検索…',
      browseSearching: 'Hugging Face を検索中',
      browseListing: 'モデルファイルを読み込み中',
      browseShowFiles: 'ファイルを表示',
      browseRefresh: '更新',
      browseDownloads: 'ダウンロード',
      browseLikes: 'いいね',
      browseGated: 'Hugging Face へのサインインが必要',
      browseNoGguf: '互換性のあるモデルファイルが見つかりません。',
      browseFitUnknown: '適合状況は不明',
      browseAlreadyDownloaded: 'ダウンロード済みです。',
      addedByYou: 'あなたが追加',
      browseDownloadStarted: '{name} をダウンロード中',
      browseDownloadAria: '{name} をダウンロード',
      sideloadButton: 'モデルファイルを追加',
      sideloadTitle: 'GGUF モデルファイルを選択',
      sideloadDone: '{name} を追加しました。',
      sideloadAlreadyPresent: '既にライブラリにあります。',
      pillFullContext: max => `フル ${max} コンテキスト`,
      pillFullContextTip: '最初からモデルの完全なコンテキストウィンドウで動作します',
      pillUpTo: max => `最大 ${max} コンテキスト`,
      pillGrowsTip: '会話が必要とするにつれて自動的に拡張します',
      pillVision: '画像対応',
      deleteAction: 'モデルを削除',
      deleteConfirm: model => `${model} をディスクから削除しますか？`,
      deleted: model => `${model} を削除しました。`,
      deleteFailed: '削除に失敗しました'
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
      autoArchiveTitle: '古いチャットを自動アーカイブ',
      autoArchiveDesc:
        'しばらく操作していないチャットを自動的にアーカイブします。ピン留めしたチャットはアーカイブされず、削除もされません。アーカイブされたチャットはここに移動します。',
      autoArchiveDaysLabel: 'アーカイブまでの日数',
      autoArchiveDaysUnit: '日間操作なし',
      autoArchiveFailed: '自動アーカイブを更新できませんでした',
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
      activeBackend: '使用中',
      activeBackendHint: 'これが現在アクティブなバックエンドです',
      useBackend: 'このバックエンドを使う',
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
        needsSetupHint:
          'このバックエンドは選択されていますが、セットアップが完了するまでコマンドは失敗します。',
        needsSetupConfirmTitle: backend => `それでも ${backend} を選択しますか？`,
        needsSetupConfirmDescription: detail =>
          `${detail} この変更後に開始されるセッションは、セットアップが完了するまでターミナルとファイルツールを使用できません。`,
        needsSetupConfirmDescriptionGeneric:
          'このバックエンドはまだセットアップされていません。この変更後に開始されるセッションは、セットアップが完了するまでターミナルとファイルツールを使用できません。',
        needsSetupConfirmAction: 'それでも選択する'
      },
      browserRealProfile: {
        label: '実際のブラウザプロファイルを使用',
        description:
          '既定ブラウザのログイン情報と Cookie を管理されたスナップショットにコピーし、エージェントはそれを使ってブラウジングします。実際のプロファイルが直接開かれることはありません。新しいセッションに適用されます。',
        enabledTitle: '実プロファイルブラウジング：オン',
        enabledMessage: '新しいセッションは既定ブラウザプロファイルのスナップショットでブラウジングします。',
        disabledTitle: '実プロファイルブラウジング：オフ',
        disabledMessage: 'プロファイルのスナップショットは削除され、新しいセッションはクリーンなブラウザを使用します。',
        failedSave: '実プロファイル設定を保存できませんでした',
        prompt: {
          title: 'サイトにログインしたまま利用',
          body: 'Hermes が既定ブラウザプロファイルのスナップショットでブラウジングできるようにすると、サイトはログイン済みの状態で開きます。',
          bulletSnapshot: 'Cookie とログイン情報は管理されたスナップショットにコピーされます。',
          bulletLiveProfile: '実際のブラウザプロファイルが直接開かれることはありません。',
          bulletLocal: 'データがこのコンピュータの外に出ることはありません。',
          dontShowAgain: '今後表示しない',
          notNow: '今はしない',
          enable: 'プロファイルを使用'
        }
      }
    }
  },

  modelPicker: {
    title: 'モデルを切り替え',
    current: '現在:',
    unknown: '(不明)',
    search: 'プロバイダーとモデルをフィルター...',
    noModels: 'モデルが見つかりません。',
    addProvider: 'プロバイダーを追加',
    loadFailed: 'モデルを読み込めませんでした',
    downloading: 'ダウンロード中',
    localDownloadsHeading: 'ローカル',
    noAuthenticatedProviders: '認証済みプロバイダーがありません。',
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
  }
} satisfies Pick<TranslationOverrides, 'language' | 'settings' | 'modelPicker' | 'modelVisibility'>
