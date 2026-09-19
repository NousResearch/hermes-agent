import type { TranslationOverrides } from './define-locale'

export const jaAssistant = {
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
      thought: '思考済み',
      thoughtBriefly: '少し思考',
      thoughtFor: duration => `${duration} 思考`,
      turnDuration: duration => `このターンの所要時間: ${duration}`,
      today: time => `今日 ${time}`,
      yesterday: time => `昨日 ${time}`,
      copy: 'コピー',
      refresh: '更新',
      moreActions: 'その他のアクション',
      branchNewChat: '新しいチャットでブランチ',
      react: 'リアクション',
      dismissError: 'エラーを閉じる',
      errorLayers: {
        auth: '認証エラー',
        billing: 'クレジット不足',
        disk: 'ディスク容量不足',
        endpoint: 'カスタムエンドポイントのエラー',
        gateway: 'ゲートウェイのエラー',
        generic: 'ターンが失敗しました',
        provider: 'プロバイダーのエラー',
        runtime: 'ローカルランタイムのエラー',
        streaming: 'ストリーミング接続のエラー'
      },
      errorRetry: '再試行',
      errorLimitResets: time => `制限は ${time} にリセットされます`,
      errorStartNewSession: '新しいセッションを開始',
      errorSwitchProvider: 'プロバイダーを切り替え',
      errorSignInAgain: provider => `${provider} に再度サインイン`,
      errorOauthExpired: provider =>
        `${provider} のサインインが期限切れか取り消されました。続けるには再度サインインしてください。`,
      errorOpenLogs: 'ログを開く',
      errorOpenLogsFailed: 'ログフォルダを開けませんでした',
      errorOpenDesktopLogs: 'デスクトップのログを開く',
      errorCopyDiagnostics: 'エラー詳細をコピー',
      errorSendDiagnostics: '診断情報を送信',
      filesChanged: count => `${count} 件のファイルを変更`,
      reviewChanges: 'レビュー',
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
      confirmAndContinueLabel: '確定して続行',
      answeredBadge: '回答済み',
      questionProgress: (answered, total) => `${total}問中${answered}問回答済み`,
      lateAnswer: (question, choice) => `「${question}」について — 私の回答: ${choice}`,
      lateAnswerTip: 'この回答をフォローアップメッセージとして下書きします',
      lateAnswerHint: 'この質問はもう回答を待っていません。選択肢を選ぶとフォローアップメッセージとして下書きされます。'
    },
    tool: {
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
      failedCalls: (count: number) => `失敗したツール呼び出し: ${count}`,
      skillActivity: {
        loading: 'スキルを読み込み中',
        loaded: 'スキルを読み込みました',
        loadFailed: 'スキルの読み込みに失敗しました',
        readingResource: 'スキルのリソースを読み込み中',
        readResource: 'スキルのリソースを読み込みました',
        resourceFailed: 'スキルのリソースの読み込みに失敗しました',
        listing: 'スキル一覧を取得中',
        listed: 'スキル一覧を取得しました',
        listFailed: 'スキル一覧の取得に失敗しました',
        unavailable: 'スキルの結果を取得できません'
      },
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
      resultUnavailable: '結果を取得できません',
      memoryWriteNoted: 'メモリへの書き込みを記録',
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
        memory: {
          done: 'メモリに保存しました',
          pending: 'メモリに保存中',
          pendingAction: '保存中'
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
  }
} satisfies Pick<TranslationOverrides, 'assistant'>
