import type { TranslationOverrides } from './define-locale'

// The notifications copy (shared profile warning, backend/app version drift, voice errors),
// composed by ja.ts.
export const jaNotifications: TranslationOverrides['notifications'] = {
    sharedProfileWarning:
      '別の Hermes インストールがこのプロファイルを使用しています。両方が設定とデータを共有しているため、変更が競合する可能性があります。このまま続けるか、変更する前にもう一方を終了してください。',
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
    desktopOutOfDateTitle: 'アプリが古いです',
    desktopOutOfDateMessage:
      'この Hermes アプリは接続先のバックエンドより古く、正常に動作しない場合があります。アプリを更新して揃えてください。',
    updateDesktopApp: 'アプリを更新',
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
      microphoneSecureContextRequired:
        'マイク録音には HTTPS、localhost、またはネイティブデスクトップアプリが必要です。',
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
      turnDoneBody: 'メッセージが完了しました。',
      turnErrorTitle: 'ターンが失敗しました',
      backgroundDoneTitle: 'バックグラウンドタスクが完了しました',
      backgroundFailedTitle: 'バックグラウンドタスクが失敗しました',
      creditsTitle: 'クレジット'
    }
}
