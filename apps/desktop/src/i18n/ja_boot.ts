import type { TranslationOverrides } from './define-locale'

export const jaBoot = {
  boot: {
    ready: 'Hermes Desktop の準備ができました',
    desktopBootFailedWithMessage: message => `デスクトップの起動に失敗しました: ${message}`,
    steps: {
      connectingGateway: 'ライブデスクトップゲートウェイに接続中',
      loadingSettings: 'Hermes の設定を読み込み中',
      loadingSessions: '最近のセッションを読み込み中',
      retryingRemoteBackend: 'リモート Hermes バックエンドに再接続中…',
      startingDesktopConnection: 'デスクトップ接続を開始中',
      startingHermesDesktop: 'Hermes Desktop を起動中…'
    },
    errors: {
      backgroundExited: 'Hermes バックグラウンドプロセスが終了しました。',
      backgroundExitedDuringStartup: '起動中に Hermes バックグラウンドプロセスが終了しました。',
      backendStopped: 'バックエンドが停止しました',
      desktopBootFailed: 'デスクトップの起動に失敗しました',
      gatewayConnectionLost: 'ゲートウェイへの接続が切断されました',
      gatewayConnectionLostDetail:
        'Still retrying in the background. You can keep reading and drafting — open Gateway settings if this persists.',
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
      cloudDownTitle: 'Nous Cloud エージェントが停止しています',
      cloudDownDescription:
        'このゲートウェイが接続している Nous 管理のクラウドエージェントがサーバーエラーを返しています。ここから再起動することはできません。ステータスを確認するか、ローカルゲートウェイに切り替えるか、サポートに連絡してください。',
      cloudDownHint:
        '下のボタンから Nous Portal（インスタンスの状態と操作）を開くか、Discord でサポートを受けられます。',
      cloudDownCheckPortal: 'Portal のステータスを確認',
      cloudDownDiscord: 'Discord でサポートを受ける',
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

  remoteDisplayBanner: {
    message: reason =>
      `ソフトウェアレンダリングが有効です — リモートディスプレイを検出しました（${reason}）。ちらつきを防ぐため GPU アクセラレーションは無効化されています。`
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
    gitUnusable: 'このコンピューターで Git を実行できなかったため、更新を確認できませんでした。',
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
    blockerTitle: 'Hermes を更新するためにローカルプレビューを閉じますか？',
    blockerBody:
      '更新する前に、これらのローカルプレビューを停止する必要があります。ファイルが変更または削除されることはありません。',
    foreignBlockerTitle: '他のプロセスを閉じて Hermes を更新',
    foreignBlockerBody:
      'Hermes はこれらのプロセスを安全に自動終了できません。各プロセスを所有するアプリ、ターミナル、またはサービスを閉じてから、もう一度更新してください。',
    mixedBlockerBody:
      'Hermes は以下のローカルプレビューを閉じることができます。更新を続けるには、他のプロセスを手動で閉じる必要があります。',
    closePreviewsAndUpdate: 'プレビューを閉じて更新',
    closePreviewsAndCheckAgain: 'プレビューを閉じて再確認',
    localPreview: 'ローカルプレビュー',
    portLabel: port => `ポート ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: '技術的な詳細',
    notNow: '今は後で',
    clientAlsoBehindTitle: 'デスクトップアプリが古くなっています',
    clientAlsoBehindMessage:
      'バックエンドは最新ですが、このデスクトップアプリはまだ古いバージョンです。最新の修正を反映するには更新してください。',
    clientAlsoBehindAction: 'デスクトップアプリを更新',
    everythingDispatched: '更新を開始しました',
    everythingSkipped: 'スキップ',
    everythingRowFailed: '更新に失敗しました',
    everythingFanoutFailedTitle: '他のインスタンスを更新できませんでした',
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

  guidedGreeting: {
    line: 'やあ、どうぞ。Hermes です。二分だけください、あなたに合わせて整えます。それから、本当にやりたいことに取りかかりましょう。\n\nまずは、何とお呼びすればいいですか。',
    nameSuggestion: (name: string) => `（よければ、${name} さんとお呼びします。）`
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
    setupChoiceTitle: 'Hermes Desktop をセットアップ',
    setupChoiceDesc:
      'すでに実行している Hermes ゲートウェイに接続するか、このコンピューターに Hermes をローカルインストールします。',
    connectExistingTitle: '既存の Hermes に接続',
    connectExistingShort: '既存環境に接続',
    connectExistingDesc:
      'セッショントークンまたはブラウザーサインインでリモートバックエンドを使用します。ローカルインストールは開始されません。',
    installLocalTitle: 'Hermes をローカルにインストール',
    installLocalDesc: 'Hermes をダウンロードし、Python 環境を作成して、このコンピューターでバックエンドを実行します。',
    localStartUnavailable:
      'ローカルインストールを開始できません。Hermes Desktop を再起動して、もう一度お試しください。',
    remoteSetupTitle: '既存の Hermes に接続',
    remoteSetupDesc:
      'ゲートウェイ URL を入力してください。Hermes Desktop がトークンとブラウザーサインインのどちらが必要かを検出します。',
    remoteUrlTitle: 'ゲートウェイ URL',
    remoteUrlDesc: 'Hermes ゲートウェイのベース URL を使用します。リモートの場合は https:// を含めてください。',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'ゲートウェイ認証方式を検出中...',
    probeError: 'その Hermes ゲートウェイに到達できませんでした。',
    identityProvider: 'ID プロバイダー',
    authTitle: '認証',
    authNeedsOauth: provider => `このゲートウェイをテストする前に ${provider} でサインインしてください。`,
    authSignedIn: 'ブラウザーサインインが完了しました。',
    connected: '接続済み',
    signIn: 'サインイン',
    signInWith: provider => `${provider} でサインイン`,
    enterUrlFirst: '先にゲートウェイ URL を入力してください。',
    signInIncomplete: '認証が完了する前にサインインウィンドウが閉じられました。',
    tokenTitle: 'セッショントークン',
    tokenDesc: 'リモートゲートウェイの .env ファイルからセッショントークンを貼り付けます。',
    pasteSessionToken: 'セッショントークンを貼り付け',
    incompleteSignInTest: 'OAuth で保護されたこのゲートウェイをテストする前にサインインしてください。',
    incompleteTokenTest: 'このゲートウェイをテストする前にセッショントークンを入力してください。',
    testConnection: '接続をテスト',
    testSucceeded: (baseUrl, version) => `${baseUrl}${version ? ` (${version})` : ''} に接続しました。`,
    applyRemote: '適用して再接続',
    backToSetup: '戻る',
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
    localModelsTitle: 'モデルをローカルで実行',
    localModelsPitch: 'アカウント不要——モデルをダウンロードしてこのマシンで実行',
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
    signInExpired:
      '承認待ちでタイムアウトしました。多くの場合、開いたタブのサインインページが止まっている（サーバー側の問題）ためです。そのページでサインインを完了してから再試行してください。解決しない場合は API キーまたは CLI を利用してください。',
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
  }
} satisfies Pick<
  TranslationOverrides,
  'boot' | 'remoteDisplayBanner' | 'updates' | 'guidedGreeting' | 'install' | 'onboarding'
>
