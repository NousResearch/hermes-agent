import type { TranslationOverrides } from './define-locale'

export const jaDiagnostics = {
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
    updateReadyMessageUnknown: '新しい更新が利用可能です。',
    seeWhatsNew: '新機能を見る',
    mcp: {
      needsAuthTitle: 'MCP サーバーの再認証が必要です',
      needsAuthMessage: name => `${name} MCP の再認証が必要です。`,
      errorTitle: 'MCP サーバーに接続できません',
      errorMessage: name => `${name} MCP のヘルスチェックに失敗しました。`,
      signIn: 'サインイン',
      view: '表示',
      disable: '無効化',
      disabledMessage: name => `${name} MCP を無効にしました。機能 → MCP からいつでも再有効化できます。`,
      disableFailed: name => `${name} MCP を無効にできませんでした。`
    },
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT には ELEVENLABS_API_KEY が必要です。',
      elevenLabsRejectedKey: 'ElevenLabs が API キーを拒否しました (401)。',
      diskFull: 'ディスク容量不足です — 空きを作ってからもう一度お試しください。',
      gatewayAuthFailed: 'ゲートウェイ認証に失敗しました — API_SERVER_KEY を確認してください。',
      methodNotAllowed:
        'デスクトップバックエンドがそのリクエストを拒否しました (405 Method Not Allowed)。Hermes Desktop を再起動してください。',
      microphonePermission: 'マイクのアクセス許可が拒否されました。',
      openaiRejectedApiKey: 'OpenAI が API キーを拒否しました。',
      openaiTtsNeedsKey: 'OpenAI TTS には VOICE_TOOLS_OPENAI_KEY または OPENAI_API_KEY が必要です。',
      codeSkewRestartRequired:
        'アップデート後、このバックエンドは古いコードのままです。再起動して新しいコードを読み込んでください。'
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
      sayStopToEnd: phrase => `「${phrase}」と言うと音声チャットを終了できます。`,
      transcriptionFailed: '音声文字起こしに失敗しました',
      transcriptionUnavailable: '音声文字起こしはまだ利用できません。',
      tryRecordingAgain: 'もう一度録音してください。',
      unavailable: '音声は利用できません'
    },
    native: {
      approvalTitle: '承認が必要です',
      approvalTitleNamed: session => `承認が必要です — ${session}`,
      approveAction: '承認',
      rejectAction: '拒否',
      inputTitle: '入力が必要です',
      inputTitleNamed: session => `入力が必要です — ${session}`,
      inputBody: 'Hermes が応答を待っています。',
      turnDoneTitle: 'Hermes が完了しました',
      turnDoneBody: '',
      turnErrorTitle: 'ターンが失敗しました',
      backgroundDoneTitle: 'バックグラウンドタスクが完了しました',
      backgroundFailedTitle: 'バックグラウンドタスクが失敗しました',
      creditsTitle: 'クレジット'
    }
  },

  sendDiagnostics: {
    title: 'Nous に診断情報を送信',
    privacyNotice:
      'デバッグバンドルを Nous 内部ストレージにアップロードします（公開ペーストではありません）。システム情報（OS、バージョン、プロバイダー、設定済み API キーの種類 — キー自体は含まれません）と、エージェント/ゲートウェイ/デスクトップの完全なログ（各最大 512 KB。会話内容、ツール出力、ファイルパスを含む可能性が高い）が含まれます。シークレットはアップロード前にマスクされます。閲覧できるのは Nous スタッフと許可された Discord モデレーターのみで、14 日後に自動削除されます。',
    upload: 'アップロード',
    uploading: 'アップロード中…',
    cancel: 'キャンセル',
    close: '閉じる',
    copyLink: 'リンクをコピー',
    uploadIdFallback: id => `表示リンクが返されませんでした — サポートにアップロード ID ${id} をお伝えください`,
    doneTitle: '診断情報を送信しました',
    doneDescription:
      'バンドルは非公開でアップロードされました。サポートスレッドで以下のリンクを共有すると、チームがログを確認できます。',
    failedTitle: 'アップロードに失敗しました',
    failedHint:
      'ターミナルから `hermes debug share --nous` を実行するか、`hermes debug share --local` でアップロードせずにレポートを表示することもできます。',
    handoffLead: '続きは次の場所で:',
    links: {
      github: 'GitHub Issues',
      portal: 'Nous Portal サポート',
      discord: 'Discord'
    }
  },

  errors: {
    genericFailure: '問題が発生しました',
    boundaryTitle: 'インターフェイスで問題が発生しました',
    boundaryDesc: 'ビューで予期しないエラーが発生しました。チャットと設定は安全です。',
    reloadWindow: 'ウィンドウを再読み込み',
    openLogs: 'ログを開く'
  }
} satisfies Pick<TranslationOverrides, 'notifications' | 'sendDiagnostics' | 'errors'>
