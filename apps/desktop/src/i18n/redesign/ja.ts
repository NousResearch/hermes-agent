import type { RedesignOverlay } from './types'

// Japanese wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const jaRedesign: RedesignOverlay = {
  sharedMetrics: {
    collectedMilestones: 'セットアップ：完了したステップ、プロバイダーの接続、スキル・プラグイン・ルーティンの数',
    sending:
      '共有すると、Hermes はこのエージェント用のランダムな ID とともに 1 日 1 回 Nous に統計を送信します。Hermes がインストールされたことを示す 1 回限りの記録を除き、オプトイン前の統計が送信されることはありません。この設定はいつでも設定画面で変更できます。',
    sendDesc:
      '件数のみを、このエージェント用のランダムな ID を付けて 1 日 1 回送信します。メッセージ、ファイル、パス、エラーテキストは一切含みません。',
    consentBody:
      'Hermes は使われ方を集計できます：チャットの長さ、実行されたモデルとツール、何かが失敗したタイミング。メッセージ、ファイル、パス、エラーテキストは一切記録しません。',
    collectedActivity: 'チャット：長さ、結果、エラーの種類、1 日あたりのアクティブ時間'
  },
  common: {
    bots: 'エージェント'
  },
  boot: {
    steps: {
      connectingGateway: 'マシンに接続中',
      loadingSessions: '最近のチャットを読み込み中'
    },
    errors: {
      gatewayConnectionLost: 'マシンへの接続が切断されました',
      gatewayConnectionLostDetail:
        'バックグラウンドで再接続を続けています。閲覧や下書きは続けられます。問題が続く場合はマシン設定を開いてください。',
      gatewaySignInRequired: 'マシンへのサインインが必要です'
    },
    failure: {
      description:
        'バックグラウンドの Hermes が起動しませんでした。以下の回復手順をお試しください。チャットや設定は削除されません。',
      remoteTitle: 'リモートマシンへのサインインが必要です',
      remoteDescription:
        'リモートマシンのセッションが期限切れです。再接続するにはもう一度サインインしてください。チャットや設定は削除されません。',
      useLocalGateway: 'ローカルマシンを使用',
      gatewaySettings: 'マシン設定',
      remoteSignInHint: signInLabel =>
        `保存済みのリモートブラウザセッションからサインアウトし、${signInLabel}を開きます。代わりにバンドルされたバックエンドに切り替えるには「ローカルマシンを使用」を選択してください。`,
      remoteFailureHint: '「マシン設定」でマシンの URL とサインインを確認するか、ローカルマシンに切り替えてください。',
      cloudDownDescription:
        '接続先の Nous 管理のクラウドエージェントがサーバーエラーを返しています。ここから再起動することはできません。ステータスを確認するか、ローカルマシンに切り替えるか、サポートに連絡してください。',
      signedInMessage: 'リモートマシンに再接続中…',
      signInToRemoteGateway: 'リモートマシンにサインイン'
    }
  },
  notifications: {
    sharedProfileWarning:
      '別の Hermes インストールがこのエージェントを使用しています。両方が設定とデータを共有しているため、変更が競合する可能性があります。このまま続けるか、変更する前にもう一方を終了してください。',
    errors: {
      gatewayAuthFailed:
        'この Hermes は保存済みのサインインを受け付けなくなりました。「マシン」を開いてもう一度サインインするか（または新しいアクセストークンを貼り付けて）、再試行してください。'
    },
    voice: {
      couldNotStartSession: '音声チャットを開始できませんでした'
    },
    native: {
      inputBody: 'エージェントが応答を待っています。',
      turnDoneTitle: 'エージェントが完了しました'
    }
  },
  promptNotices: {
    desktopCommands: {
      configFooter:
        '設定画面で変更するか、接続先のマシンで `hermes config edit` を使って config.yaml を編集してください。',
      saveLegacyRefused:
        '保存されませんでした: このマシンは秘匿化・ファイル名指定・JSON 以外のコピーを書き出せません。Hermes を更新するか、オプションなしで /save を実行して接続先のマシンに生の JSON を書き込んでください。',
      savedOnHost: (path: string) => `接続先のマシンに保存しました: ${path}`,
      exportRemote:
        '/export は接続先のマシンからエージェントをパッケージ化しますが、そのマシンはこのコンピューターではありません。そのマシンで `hermes profile export` を実行してください。',
      importRemote:
        '/import は接続先のマシンにエージェントをインストールしますが、そのマシンはこのコンピューターではありません。そのマシンで `hermes profile import <archive>` を実行してください。',
      historyHermes: 'エージェント',
      checkpointsDisabled:
        'このチャットではチェックポイントが有効になっていません（config.yaml で checkpoints.enabled を設定してください）。'
    },
    sessionCommands: {
      busy: (command: string) =>
        `チャットが使用中です — エージェントはまだ応答中です。先に現在の応答を停止してから（停止ボタン、またはターミナルでは Ctrl+C）/${command} を実行してください。`,
      busyCompressing: (command: string) =>
        `チャットが使用中です — /compress を実行中です。完了したら /${command} を実行してください。`,
      titleSet: (title, queued) =>
        `チャットのタイトルを設定しました: ${title}${queued ? '（チャットの開始後に反映）' : ''}`,
      titleCleared: 'チャットのタイトルを消去しました。'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      'デバッグバンドルを Nous 内部ストレージにアップロードします（公開ペーストではありません）。システム情報（OS、バージョン、プロバイダー、設定済み API キーの種類 — キー自体は含まれません）と、エージェント/Hermes サービス/デスクトップの完全なログ（各最大 512 KB。会話内容、ツール出力、ファイルパスを含む可能性が高い）が含まれます。シークレットはアップロード前にマスクされます。閲覧できるのは Nous スタッフと許可された Discord モデレーターのみで、14 日後に自動削除されます。'
  },
  settings: {
    subpages: {
      gatewayConnection: 'プライマリマシン'
    },
    nav: {
      gateway: 'マシン',
      sessions: 'チャット'
    },
    searchPlaceholder: {
      gateway: 'マシン接続…',
      sessions: 'アーカイブ済みチャットを検索…'
    },
    appearance: {
      vibeHeartsDesc:
        '感謝や愛情を伝える言葉、ハート絵文字に反応して浮かぶハート。上のメッセージリアクションとは別です。',
      themeProfileNote: profile =>
        `エージェント「${profile}」に保存されます。エージェントごとに個別のテーマを保持します。`,
      sessionDensityTitle: 'チャットリストの密度',
      sessionDensityDesc: 'サイドバーのチャットタイトルの下に表示する情報量を選びます。',
      reactionsDesc:
        'iMessage風の絵文字タップバック — メッセージにリアクションでき、エージェントもあなたのメッセージにリアクションします。',
      tipsDesc:
        'アプリやエージェントからのヒントをときどき表示します。各ヒントは一度だけ表示されます。利用開始から30日後に自動でオフになりますが、再びオンにできます。',
      toursDesc:
        '各ステップを強調しながら、エージェントがアプリを案内します。利用開始から30日後に自動でオフになりますが、再びオンにできます。',
      pet: {
        intro:
          'アプリ上に浮かぶ petdex のアニメーションマスコットを採用しましょう。ツール実行中は走り、成功すると喜び、エラーでしょんぼりと、エージェントの状態に反応します。'
      }
    },
    uninstallSection: {
      options: {
        full: {
          description:
            'アプリ、エージェント、すべてのユーザーデータ（設定、チャット、ルーティン、シークレット、ログ）を削除します。'
        }
      }
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `このビルドは安定版 ${v} より新しく、チャットはすでに ${v} では開けない Hermes ランタイムを使っています。新しい安定版が出るまで、Hermes はこのビルドのままです。`
      }
    },
    gateway: {
      loading: 'マシン設定を読み込み中...',
      unavailableTitle: 'マシン設定は利用できません',
      unavailableDesc: 'デスクトップ IPC ブリッジはマシン設定を公開していません。',
      webappHostDesc:
        'Webapp は常にそれを配信する Hermes マシン上で動作します。下の保存済み接続はこのマシンの一覧で、Hermes Desktop アプリと共有されます。他のマシンにはこのマシン経由で接続します。',
      webappOauthPaste: {
        description:
          'サインインのタブが読み込めないアドレスで止まった場合（このブラウザが接続先のマシン上にない場合）、そのアドレスをここに貼り付けてください。'
      },
      webAccess: {
        pendingOn: '保存しました。Hermes の再起動後に Web アプリが提供されます。',
        pendingOff: 'オフにしましたが、Hermes は再起動するまで Web アプリを提供し続けます。',
        unavailableDetail: reason => `Hermes が Web アプリを提供できませんでした: ${reason}`,
        restartBusy:
          '保存しました。チャットが実行中のため、Hermes は再起動しませんでした。終了したら「Hermes を再起動」を選んでください。',
        restartFailed:
          '保存しましたが、Hermes を再起動できませんでした。「Hermes を再起動」を選んで再試行してください。',
        restartFailedBecause: reason => `Hermes を再起動できませんでした: ${reason}`,
        startTimedOut: 'Hermes が 90 秒以内に Web アプリを提供しませんでした。',
        stopTimedOut: 'Hermes が 90 秒以内に Web アプリの提供を停止しませんでした。',
        startStages: {
          restarting: 'Web アプリを提供するために Hermes を再起動しています。'
        },
        restartGateway: 'Hermes を再起動'
      },
      title: 'マシン接続',
      intro:
        'Hermes Desktop はデフォルトでこのマシン上の Hermes に接続します。別のマシンや信頼できるプロキシの背後で動作する Hermes を使用する場合は、リモートマシンに接続してください。マシンへの接続はこの端末の設定として保存され、エージェントは接続先のマシンから検出されます。',
      localTitle: 'ローカルマシン',
      remoteTitle: 'リモートマシン',
      remoteDesc:
        'このデスクトップシェルをリモートの Hermes バックエンドに接続します。ホスト型マシンは OAuth またはユーザー名とパスワードを使用します。自己ホスト型はセッショントークンを使用する場合があります。',
      probing: 'このマシンの認証方法を確認中…',
      probeError: 'このマシンにまだ到達できません。URL を確認してください。応答後に認証方法が表示されます。',
      authSignedInPassword:
        'このマシンはユーザー名とパスワードを使用します。サインイン済みです。セッションは自動的に更新されます。',
      authSignedInOauth: 'このマシンは OAuth を使用します。サインイン済みです。セッションは自動的に更新されます。',
      authNeedsPassword:
        'このマシンはユーザー名とパスワードを使用します。このデスクトップアプリを承認するにはサインインしてください。',
      authNeedsOauth: provider =>
        `このマシンは OAuth を使用します。このデスクトップアプリを承認するには ${provider} でサインインしてください。`,
      plainTextConfirmTitle: 'マシントークンを平文で保存しますか？',
      keychainEncryptionDesc:
        'デフォルトはオフです。オンにすると、マシンのトークンとサインイン資格情報がシステムのキーチェーン（Keychain Access、GNOME Keyring、Windows DPAPI）で暗号化されます。システムから許可やパスワードを求められる場合があります。オフの場合は、現在のユーザーのみが読める通常ファイルとして保存されます。',
      diagnosticsDesc: 'ファイルマネージャーで desktop.log を表示します。Hermes の起動に失敗した際に役立ちます。',
      incompleteTitle: 'リモートマシンの設定が不完全です',
      restartingTitle: 'マシンに再接続中',
      savedTitle: 'マシン設定を保存しました',
      reachableTitle: 'リモートマシンに到達可能',
      signedOutMessage: 'リモートマシンセッションをクリアしました。',
      failedLoad: 'マシン設定の読み込みに失敗しました',
      testFailed: 'リモートマシンのテストに失敗しました',
      applyFailed: 'マシン設定を適用できませんでした',
      saveFailed: 'マシン設定を保存できませんでした',
      webappHostTitle: 'Hermes マシン',
      envOverrideTitle: 'この接続は Hermes の起動方法によって固定されています。'
    },
    profileScope: {
      editsProfile: profile => `このページの変更はエージェント「${profile}」に適用されます。`
    },
    model: {
      tasks: {
        profile_describer: {
          label: 'エージェントの説明',
          hint: 'エージェントの概要を自動生成'
        },
        title_generation: {
          hint: 'チャットタイトル'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: 'エージェントルート',
              description: 'サブエージェントが選択したエージェントのモデル、スキル、キーを使って動作します。'
            }
          },
          pluginMissing: 'モデルルートとエージェントルートは Advanced Delegation プラグインが提供します。',
          pluginDisabled: 'Advanced Delegation はインストール済みですが、このエージェントではオフです。',
          blockedDisabled: 'このエージェントで Advanced Delegation をオンにするまで、サブエージェントは起動しません。',
          profileRoutesTitle: 'サブエージェントとして使えるエージェント',
          profileRoutesHint: '選択したエージェントのモデル、スキル、指示、キーを使います。',
          profileEmpty: 'ほかのエージェントはまだありません。エージェントバーから作成してください。',
          profileMissing: name => `${name}（エージェントは存在しません）`,
          inactiveProfiles: profile =>
            `エージェントルートが各サブエージェントの設定元を選びます。エージェントを選ばなかったサブエージェントは「${profile}」として動作します。`,
          inactiveProfilesNoDefault:
            'エージェントルートが各サブエージェントの設定元を選びます。エージェントを選ばなかったサブエージェントはメインエージェントのモデルを使います。'
        }
      },
      appliesDesc:
        '新しいチャットに適用されます。コンポーザーのモデルピッカーを使ってアクティブなチャットをホットスワップできます。'
    },
    vault: {
      otpHint:
        '2FA を有効にするときにサイトが表示する「セットアップキー」。保存するとエージェントがコードを生成します。'
    },
    notifications: {
      kinds: {
        input: {
          description: 'エージェントが質問したか、パスワードやシークレットを必要としています。'
        }
      }
    },
    fieldDescriptions: {
      'display.personality': '新しいチャットのデフォルトのアシスタントスタイルです。',
      'memory.memoryEnabled': '将来のチャットに役立つ永続メモリを保存します。'
    },
    memoryLayers: {
      selectedDesc: current =>
        `まだ有効ではありません。この選択が保存されるまで、チャットは ${current} を使い続けます。`,
      noProviderDesc:
        'エージェントは下の組み込みレイヤーのみを使用します。追加するには、設定 › メモリとコンテキストで外部プロバイダーを選択してください。',
      providerDesc: name => `${name} は組み込みレイヤーに加えて、すべてのチャットに独自の想起を追加します。`,
      builtinDesc:
        'MEMORY.md と USER.md は外部プロバイダーとは別の、エージェント自身のメモリです。どちらも既定でチャットに読み込まれ、設定 › メモリとコンテキストのメモリとユーザープロファイルの切り替えでそれぞれオフにできます。プロバイダーはこのレイヤーに追加されるもので、置き換えることはありません。'
    },
    sessions: {
      loading: 'アーカイブ済みチャットを読み込み中…',
      archivedTitle: 'アーカイブ済みチャット',
      defaultDirDesc:
        '別のフォルダーを選択しない限り、新しいチャットはこのフォルダーで開始します。未設定の場合はホームディレクトリが使用されます。',
      failedLoad: 'アーカイブ済みチャットを読み込めませんでした'
    },
    toolsets: {
      terminalBackend: {
        selectedMessage: backend => `ターミナルコマンドは ${backend} で実行されます。新しいチャットに適用されます。`,
        needsSetupConfirmDescription: detail =>
          `${detail} この変更後に開始されるチャットは、セットアップが完了するまでターミナルとファイルツールを使用できません。`,
        needsSetupConfirmDescriptionGeneric:
          'このバックエンドはまだセットアップされていません。この変更後に開始されるチャットは、セットアップが完了するまでターミナルとファイルツールを使用できません。'
      },
      browserRealProfile: {
        description:
          '既定ブラウザのログイン情報と Cookie を管理されたスナップショットにコピーし、エージェントはそれを使ってブラウジングします。アプリ内ブラウザにもそれでログインします。実際のプロファイルが直接開かれることはありません。新しいチャットに適用されます。',
        enabledMessage: '新しいチャットは既定ブラウザプロファイルのスナップショットでブラウジングします。',
        disabledMessage:
          'プロファイルのスナップショットは削除され、アプリ内ブラウザもそのログインから外れます。新しいチャットはクリーンなブラウザを使用します。',
        prompt: {
          body: 'エージェントが既定ブラウザプロファイルのスナップショットでブラウジングできるようにすると、エージェントでもアプリ内ブラウザでも、サイトはログイン済みの状態で開きます。'
        }
      }
    }
  },
  skills: {
    plugins: {
      autoUpdateTip: (hours: number) =>
        `Hermes の確認時（${hours} 時間ごと）にプラグインの更新を自動で適用します。更新されたコードは新しいチャットで読み込まれます。`
    },
    remotePluginHint:
      'プラグインを管理するには、このエージェントをそのマシンで開いてください。プラグインのインストールには接続中のマシンを使用します。',
    openAgentSettings: 'エージェント設定',
    appliesToNewSessions: name => `${name} は新しいチャットに適用されます。`,
    bulkUpdated: count => `${count} 件を新しいチャット向けに更新しました。`,
    changesApplyNewSessions: '変更は新しいチャットに適用されます。',
    openToolLibrary: 'ツールカタログを見る'
  },
  gatewayGroupOpen: {
    lookupFailed: 'マシンでこのグループを検索できませんでした。もう一度お試しください。'
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: 'マシンのステータス、ログ、再起動/更新'
      },
      sessions: {
        title: 'チャットパネル',
        detail: 'チャットの検索、ピン留め、管理'
      }
    },
    gatewayRunning: 'メッセージングが実行中',
    gatewayStopped: 'メッセージングが停止中',
    restartGateway: 'Hermes を再起動',
    gatewayRestartFailed: 'Hermes の再起動に失敗しました。',
    sharedGatewayRestartTitle: 'すべてのエージェントの Hermes を再起動しますか？',
    sharedGatewayRestartDescription: bots => `このデバイス上のすべてのエージェントが再接続します: ${bots}`,
    sharedGatewayRestarted: count => `Hermes を再起動しました（エージェント ${count} 件）`,
    searchPlaceholder: 'チャット、ビュー、アクションを検索',
    goToSession: 'チャットへ移動',
    sections: {
      sessions: 'チャット'
    },
    nav: {
      newChat: {
        title: '新しいチャット',
        detail: '新しいチャットを開始'
      },
      artifacts: {
        title: 'ライブラリ'
      }
    },
    providerSessions: 'チャット',
    pinSession: 'チャットをピン留め',
    unpinSession: 'チャットのピン留めを解除',
    exportSession: 'チャットをエクスポート',
    deleteSession: 'チャットを削除',
    noSessions: 'チャットはまだありません。',
    hermesActiveSessions: (version, count) => `Hermes ${version} · アクティブチャット ${count}`,
    statSessions: 'チャット'
  },
  messaging: {
    states: {
      gateway_stopped: 'メッセージングが停止中'
    },
    hintPendingRestart: 'この変更を適用するにはステータスバーから Hermes を再起動してください。',
    sharedListenerUrl: '利用可能なアドレス:',
    hintGatewayStopped: 'ステータスバーから Hermes を起動して接続してください。',
    gatewayStopped: 'メッセージングが停止中',
    restartToApply: 'この変更は Hermes の再起動後に有効になります。',
    restartToReconnect: '新しい認証情報は Hermes の再起動後に有効になります。',
    appliedLive: 'Hermes を再起動せずに適用しました。',
    connectingLive: '新しい認証情報で接続しています。',
    restartNeeded: '保存しました。新しい設定を反映するには Hermes を再起動してください。',
    restartFailedManual: 'Hermes の再起動に失敗しました。手動で再起動し、Hermes のログを確認してください。',
    relayLink: {
      managedNote: (platform: string) =>
        `このインスタンスではメッセージを Team Gateway が扱うため、Hermes は起動のたびに ${platform} の直接接続をオフにします。リレーと併用するには、Hermes の環境に GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true を設定してください（直接接続があるとインスタンスがゼロまでスケールダウンできず、メッセージが二重に届くことがあります）。`
    },
    telegramQr: {
      savedRestarting: 'Telegram を保存しました。Hermes を再起動中…',
      savedRestartFailed: detail => `Telegram を保存しましたが、Hermes の再起動に失敗しました${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'ボットがプロアクティブなメッセージを送信するチャンネル（ルーティンの実行結果、リマインダー）。'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'ルーティンの実行結果を配信するデフォルトのチャンネルまたはグループ。'
      }
    }
  },
  profiles: {
    close: 'エージェントを閉じる',
    title: 'エージェント',
    count: count => `エージェント ${count} 件`,
    search: 'エージェントを検索...',
    loading: 'エージェントを読み込み中...',
    newProfile: '新しいエージェント',
    importProfile: 'エージェントをインポート…',
    exportProfile: 'エージェントをエクスポート…',
    imported: 'エージェントをインポートしました',
    exported: 'エージェントをエクスポートしました',
    failedImport: 'エージェントのインポートに失敗しました',
    failedExport: 'エージェントのエクスポートに失敗しました',
    exportNeedsHost: name =>
      `ブラウザーではエージェントのアーカイブを保存できません。Hermes を実行しているマシンで次を実行してください: hermes profile export ${name}`,
    allProfiles: 'すべてのエージェント',
    showAllProfiles: 'すべてのエージェントを表示',
    manageProfiles: 'エージェントを管理…',
    remoteOverride: {
      description: 'このエージェントのチャットは、このパソコンではなく指定したリモートの Hermes で実行されます。',
      collisionWarning: (label: string) =>
        `「${label}」という名前のマシンが設定に既に存在します。このエージェントの接続は別物であり、それを変更しません。`,
      confirmTitle: 'このエージェントをリモートマシンに接続しますか？',
      savedTitle: 'エージェントを接続しました',
      menuItem: 'リモートマシンに接続…',
      title: (profile: string) => `${profile} をリモートマシンに接続`,
      confirmNote: (profile: string, host: string) =>
        `${profile} の新しいチャットは ${host} で実行されます。コマンドの実行やファイルの読み取りはこのパソコンではなくそのコンピュータで行われます。信頼できるマシンにのみ接続してください。`,
      authFailedTitle: 'リモートマシンが保存済みトークンを拒否しました'
    },
    defaultProfile: 'デフォルトのエージェント',
    defaultDescription: 'Hermes の起動時と新しいチャットに使用します。既存のチャットのエージェントは変わりません。',
    failedSetDefault: 'デフォルトのエージェントを設定できませんでした',
    noProfiles: 'エージェントはまだありません。',
    selectPrompt: '詳細を表示するにはエージェントを選択してください。',
    refresh: 'エージェントを更新',
    refreshing: 'エージェントを更新中',
    soulDesc: 'このエージェントへの指示や振る舞いを設定します。',
    soulMissing:
      'このエージェントにはまだ SOUL.md がありません。下に指示を入力して保存すると作成できます。config.yaml のペルソナ設定は別途管理されます。',
    soulPlaceholder: mode =>
      `このエージェントへの指示や振る舞い。\n空欄のままにすると ${mode} のデフォルトを使用します。`,
    deleteTitle: 'エージェントを削除しますか？',
    createDesc: 'エージェントごとに独自の指示、設定、スキル、チャットを持ちます。',
    cloneFromDesc: '選択したエージェントから設定、スキル、SOUL.md をコピーします。',
    cloneFromDefault: 'デフォルトのエージェントから設定を複製',
    cloneFromDefaultDesc: 'デフォルトのエージェントから設定、スキル、SOUL.md をコピーします。',
    invalidName: hint => `無効なエージェント名。${hint}`,
    createAction: 'エージェントを作成',
    renameTitle: 'エージェントの名前を変更',
    renameDescPrefix: '名前を変更するとエージェントのディレクトリと ',
    failedLoad: 'エージェントの読み込みに失敗しました',
    failedDelete: 'エージェントの削除に失敗しました',
    failedCreate: 'エージェントの作成に失敗しました',
    failedRename: 'エージェントの名前変更に失敗しました',
    fleet: {
      localDevice:
        'このデバイス（ローカルバックエンド — Hermes が無ければインストールし、あれば新しいチャットを開きます）',
      installDeviceDesc:
        'Hermes をローカルにインストールしてから、このコンピュータで新しいチャットを開きます。確認するまでインストールは始まりません。'
    },
    created: 'エージェントを作成しました',
    renamed: 'エージェントの名前を変更しました',
    deleted: 'エージェントを削除しました'
  },
  cron: {
    control: {
      nothingToStop: '実行中のルーティンはありません'
    },
    close: 'ルーティンを閉じる',
    title: 'ルーティン',
    count: count => `ルーティン ${count} 件`,
    search: 'ルーティンを検索...',
    loading: 'ルーティンを読み込み中...',
    newCron: '新しいルーティン',
    emptyDescNew: 'ルーティンを作成すると、エージェントがプロンプトを自動実行し、選択した宛先に結果を送信します。',
    emptyTitleNew: 'ルーティンがまだありません',
    actionsTitle: 'ルーティンの操作',
    edit: 'ルーティンを編集',
    deleteTitle: 'ルーティンを削除しますか？',
    resumed: 'ルーティンを再開しました',
    paused: 'ルーティンを一時停止しました',
    triggered: 'ルーティンを実行しました',
    deleted: 'ルーティンを削除しました',
    created: 'ルーティンを作成しました',
    updated: 'ルーティンを更新しました',
    failedLoad: 'ルーティンの読み込みに失敗しました',
    failedUpdate: 'ルーティンの更新に失敗しました',
    failedTrigger: 'ルーティンの実行に失敗しました',
    failedDelete: 'ルーティンの削除に失敗しました',
    failedSave: 'ルーティンの保存に失敗しました',
    editTitle: 'ルーティンを編集',
    createTitle: '新しいルーティン',
    createDesc:
      'プロンプトを自動実行するルーティンを作成します。Cron 構文または「15 分ごと」などのフレーズを使用します。',
    scriptOnlyEditHint: 'スクリプトのみを実行するルーティン（AI プロンプトなし）。ルーティン ID:',
    createAction: 'ルーティンを作成',
    tabs: {
      jobs: 'ルーティン'
    },
    blueprints: {
      subtitle: 'すぐに使えるルーティン',
      dialogDesc: '詳細を入力して実行タイミングを設定します。',
      scheduled: 'ブループリントからルーティンを作成しました',
      emptyDesc: 'このバックエンドで利用できるルーティンのブループリントはありません。'
    },
    resume: 'ルーティンを再開',
    pause: 'ルーティンを一時停止'
  },
  sidebar: {
    filter: {
      profile: 'エージェント'
    },
    gatewayGroups: {
      grouping: 'マシンとエージェント',
      aliasHint: '表示名のみ変更します。マシン名とエージェント名は変わりません。'
    },
    profileRail: 'エージェントバー',
    nav: {
      cron: 'ルーティン',
      'new-session': '新しいチャット',
      artifacts: 'ライブラリ'
    },
    cronJobs: 'ルーティン',
    projects: {
      createdInPreviousContext:
        'プロジェクトは以前の接続またはエージェントで作成されました。そこに戻ってください。IDEA.md は書き込まれていません。',
      showAllSessions: 'すべてのチャットを表示',
      showAllCount: count => `${count} 件のチャットをすべて表示`
    },
    row: {
      ownedByProfile: profile => `エージェント: ${profile}`,
      copyIdFailed: 'チャット ID をコピーできませんでした',
      sessionActions: 'チャットの操作',
      sessionRunning: 'チャット実行中',
      renameTitle: 'チャットの名前を変更',
      untitledPlaceholder: '無題のチャット',
      deleteTitle: 'チャットを削除しますか？',
      deleted: 'チャットを削除しました',
      untitledChat: id => `チャット ${id}`
    },
    searchAria: 'チャットを検索',
    searchPlaceholder: 'チャットを検索…',
    noMatch: query => `"${query}" に一致するチャットがありません。`,
    sessions: 'チャット',
    groupAriaGrouped: 'チャットを単一リストとして表示',
    groupAriaUngrouped: 'ワークスペースごとにチャットをグループ化',
    showSessions: 'チャットを表示',
    groupTitleGrouped: 'チャットのグループ化を解除',
    projectEmpty: 'チャットはまだありません',
    projectLoadFailed: 'チャットの読み込みに失敗しました',
    noSessions: 'チャットはまだありません',
    noFilterMatches: 'このフィルターに一致するチャットはありません',
    newSessionIn: label => `${label} で新しいチャット`
  },
  composer: {
    commandDescs: {
      '/export': 'エージェントの設定、スキル、テーマを共有用アーカイブに書き出す',
      '/import': '共有アーカイブを新しいエージェントとして読み込む',
      '/profile': 'アクティブなエージェントを切り替え',
      '/suggestions': '提案されたルーティンを確認し、採用または却下',
      '/blueprint': 'ブループリントからルーティンを設定',
      '/platform': '問題のあるメッセージングプラットフォームを一時停止、再開、一覧表示',
      '/clear': '新しいチャットを開始',
      '/resume': '保存済みのチャットを再開',
      '/title': '現在のチャット名を変更',
      '/handoff': 'このチャットをメッセージプラットフォームへ引き継ぐ',
      '/bg': '別のバックグラウンドチャットでプロンプトを実行',
      '/agents': 'アクティブなエージェントと実行中のタスクを表示',
      '/goal': 'エージェントが達成するまでターンをまたいで取り組む継続目標を設定',
      '/heartbeat': 'アイドル時にこのチャットで繰り返すプロンプトを設定',
      '/loop': 'このチャットで一定間隔でプロンプトを繰り返す',
      '/status': '現在のチャットの状態を表示',
      '/personality': '定義済みのパーソナリティを設定',
      '/usage': 'トークン使用量とレート制限を表示。`reset` で保存済みの Codex 制限リセットを利用',
      '/model': 'このチャットのモデルを切り替え'
    },
    queueLostDiscardTip:
      'ターン中に Hermes が再起動したため完了できません。破棄すると後続のキュー済みターンが実行されます。',
    newSessionPlaceholders: [
      '何を作りますか？',
      'エージェントにタスクを与える',
      '何か考えていることはありますか？',
      '必要なことを説明してください',
      '何に取り組みますか？',
      '何でも聞いてください',
      '目標から始める'
    ],
    attachUrlDesc: 'エージェントがページを取得し、このターンのコンテキストとして含めます。',
    queueDroppedBody:
      'このバックグラウンドのエントリは、チャットを繰り返し再開できなかったため破棄されました。キューの他のエントリには影響しません。',
    browseAllSessions: 'すべてのチャットを表示…'
  },
  updates: {
    bundleOutOfSyncDesc:
      'Hermes ランタイムは更新されましたが、デスクトップアプリ自体は古いビルドのままです。アプリを更新するまで、新しいインターフェース機能（エージェントモードなど）は表示されません。下の更新を実行してアプリを再ビルドしてください。それでもこの警告が消えない場合は、最新のデスクトップインストーラーから再インストールしてください。',
    applyStatus: {
      noReturn:
        'バックエンドがオンラインに戻りませんでした。更新が完了していない可能性があります。Hermes を実行しているマシンを確認してください。'
    }
  },
  install: {
    setupChoiceDesc:
      'Hermes がすでに動作しているマシンに接続するか、このコンピューターに Hermes をローカルインストールします。',
    remoteSetupDesc:
      'マシン URL を入力してください。Hermes Desktop がトークンとブラウザーサインインのどちらが必要かを検出します。',
    remoteUrlTitle: 'マシン URL',
    remoteUrlDesc: 'Hermes マシンのベース URL を使用します。リモートの場合は https:// を含めてください。',
    probing: 'マシン認証方式を検出中...',
    probeError: 'その Hermes マシンに到達できませんでした。',
    authNeedsOauth: provider => `このマシンをテストする前に ${provider} でサインインしてください。`,
    enterUrlFirst: '先にマシン URL を入力してください。',
    tokenDesc: 'リモートマシンの .env ファイルからセッショントークンを貼り付けます。',
    incompleteSignInTest: 'OAuth で保護されたこのマシンをテストする前にサインインしてください。',
    incompleteTokenTest: 'このマシンをテストする前にセッショントークンを入力してください。',
    connectExistingDesc:
      'セッショントークンまたはブラウザーサインインでリモートマシンを使用します。ローカルインストールは開始されません。'
  },
  shell: {
    gatewayMenu: {
      gateway: 'マシン',
      reconnectGateway: 'マシンに再接続',
      machineSettings: 'マシン',
      restartTitle: 'このマシンで Hermes を再起動しますか？'
    },
    statusbar: {
      gateway: 'マシン',
      gatewayTitle: 'マシン',
      cron: 'ルーティン',
      openCron: 'ルーティンを開く'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: '削除されたマシン',
    terminalGatewayRemovedBody:
      'このターミナルが動作していたマシンは削除されたため、再度開けません。このタブを閉じてください。',
    terminalGatewayNoShellBody:
      'このマシンにはシェルアクセスがないため、このターミナルは再度開けません。このタブを閉じてください。'
  },
  preview: {
    saveScopeChanged: 'この下書きを保存するには、元の接続とエージェントに戻ってください。',
    web: {
      restarting: 'エージェントがサーバーを再起動中...',
      askRestart: 'エージェントにサーバーの再起動を依頼',
      lookingRestart: taskId => `エージェントは再起動するプレビューサーバーを検索中です (${taskId})`,
      restartingMessage: 'エージェントはバックグラウンドで作業中です。進捗はプレビューコンソールで確認してください。',
      blankPageBody: '上のアドレス欄に入力するか、エージェントにページを開くよう頼んでください。',
      finishedRestarting: message =>
        `エージェントがプレビューサーバーの再起動を完了しました${message ? `: ${message}` : ''}`,
      restartFailedMessage: 'エージェントがサーバーを再起動できませんでした。',
      stillWorking:
        'エージェントはまだ作業中ですが、再起動の結果がまだ届いていません。サーバーコマンドがフォアグラウンドで実行されている可能性があります。'
    }
  },
  assistant: {
    thread: {
      errorCodes: {
        cron_session_read_only: {
          title: 'このチャットはルーティンの実行用です',
          body: 'このチャットはルーティンの実行専用のため、メッセージは実行されませんでした。送信するには新しいチャットを開始してください。'
        },
        no_reply: {
          body: 'エージェントはこのターンを完了しましたが、返信がこのウィンドウに届きませんでした。再試行してもう一度送信してください。'
        },
        loop_error: {
          title: 'エージェントはこのステップを完了できませんでした'
        },
        launch_credentials_unavailable: {
          body: 'このチャットの開始時の API キーがその後変更または削除されたため、メッセージは送信されませんでした。開いているチャットが自動で別のキーに切り替わることはありません。「新しいチャットを開始」で現在のキーを使ってください。'
        }
      },
      loadingSession: 'チャットを読み込み中',
      loadingResponse: 'エージェントが応答を読み込み中',
      errorLayers: {
        generic: 'エージェントはこの返信を完了できませんでした',
        runtime: 'エージェントで問題が発生しました'
      },
      errorLayerBodies: {
        generic:
          'エージェントの返信中に問題が発生しました。再試行してください。問題が続く場合はエラー詳細をコピーしてください。'
      },
      errorStartNewSession: '新しいチャットを開始'
    },
    approval: {
      gatewayDisconnected: 'マシンに接続されていません',
      allowSession: 'このチャットで許可',
      alwaysDescription: pattern =>
        `これにより "${pattern}" パターンが永続的な許可リスト (~/.hermes/config.yaml) に追加されます。エージェントはこのチャットや将来のチャットで、このようなコマンドについて再度尋ねません。`
    },
    clarify: {
      gatewayDisconnected: 'マシンに接続されていません'
    },
    catalogInstall: {
      targetProfile: profile => `エージェント「${profile}」にインストールします`
    },
    tool: {
      titles: {
        cronjob: {
          done: 'ルーティン',
          pending: 'ルーティンを設定中',
          pendingAction: '設定中'
        },
        session_search_recall: {
          done: 'チャット履歴を検索しました',
          pending: 'チャット履歴を検索中'
        }
      }
    },
    themePicker: {
      basicDetail: 'エージェントと話すために。'
    }
  },
  prompts: {
    gatewayDisconnected: 'マシンに接続されていません',
    sudoInstallDesc:
      'エージェント画面のパッケージ（TigerVNC + Xfce）を接続先のマシンにインストールするため、sudo パスワードが必要です。そのマシンにのみ送信されます。',
    sudoDesc:
      'sudo パスワードを入力する前にコマンドを確認してください。パスワードは実行するエージェントに送信され、このチャット中キャッシュされます。',
    secretDesc: 'エージェントは続行するための認証情報が必要です。',
    vaultSaveDesc: origin =>
      `エージェントは ${origin} のサインインページに到達しましたが、保存されたログイン情報がありません。ここで一度入力すると、このマシン上で暗号化して保存され、ページに直接入力されます。モデルはパスワードを一切見ません。`,
    vaultCodeDesc: site =>
      `${site} がワンタイムコード（SMS、メール、または認証アプリ）を求めています。ここに入力するとエージェントがページに入力します。モデルはコードを一切見ません。`,
    vaultCodeFootnote:
      'ヒント：「設定 → パスワードとログイン」でこのログインに認証キーを保存すると、エージェントがコードを自動入力します。'
  },
  desktop: {
    profileStatus: current =>
      `エージェント: ${current}。/profile <name> または「新しいチャット」ピッカーを使って別のエージェントでチャットを始めてください。`,
    unknownProfile: '不明なエージェント',
    noProfileNamed: (target, available) => `"${target}" という名前のエージェントはありません。利用可能: ${available}`,
    newChatsProfile: name => `新しいチャットはエージェント「${name}」を使用します。`,
    setProfileFailed: 'エージェントの設定に失敗しました',
    resumeStrandedBody:
      'このチャットへの接続に失敗し、自動再試行も停止しました。マシン上で Hermes が実行中か確認してから、もう一度お試しください。',
    poolSlotTimeoutBody:
      'すべてのローカルエージェントのバックエンドスロットが使用中です。「設定」→「詳細設定」で待機状態に保つエージェントのバックエンド数を増やすか、アイドル状態のバックエンドが解放された後に再試行してください。',
    handoff: {
      timedOut: 'Hermes の待機がタイムアウトしました。`hermes gateway` は起動していますか？'
    },
    sessionUnavailable: 'チャットが利用できません',
    createSessionFailed: '新しいチャットを作成できませんでした',
    yoloSystem: active => `このチャットの YOLO ${active ? 'オン' : 'オフ'}`,
    resumeStrandedTitle: 'このチャットを読み込めませんでした',
    sessionBusy: 'チャットが使用中',
    cwdStagedMessage:
      'このアクティブなチャットへの cwd の変更を適用するにはデスクトップバックエンドを再起動してください。',
    sessionExported: 'チャットをエクスポートしました',
    sessionExportFailed: 'チャットをエクスポートできませんでした',
    sessionOpenFailed: 'このチャットを開けませんでした'
  },
  tips: {
    items: {
      cron: {
        title: 'ルーティンで自動実行',
        text: 'プロンプトを毎時・毎晩などに自動実行できます。Cron 式で実行日時を指定することもできます。'
      },
      profiles: {
        title: 'エージェントごとに独立',
        text: 'それぞれのエージェントが独自の指示、設定、メモリ、チャットを持ちます。'
      },
      skills: {
        text: 'スキルは手順書のフォルダで、必要な場面でエージェントが自分で読み込みます。'
      },
      messaging: {
        title: 'デスクを離れてもエージェントと'
      },
      artifacts: {
        title: 'エージェントが作ったものすべて',
        text: '全チャットの画像・ファイル・リンクをライブラリに集めています。'
      },
      'command-palette': {
        text: 'チャット、設定、スキル、コマンドはすべてパレットから呼び出せます。'
      }
    }
  },
  keybinds: {
    actions: {
      'agents.new': '新しいエージェント（エージェントページ）'
    }
  },
  firstRun: {
    guide: {
      upgradeRequired:
        'チャットなしのセットアップには、このマシン上の Hermes の更新が必要です。そのマシンで hermes update を実行し、Hermes を再起動してから再試行してください。チャットは変更されません。'
    },
    intro: {
      greeting: 'こんにちは、あなたのエージェントです。'
    }
  },
  catalog: {
    installDescription: 'このスキルは新しいチャットで利用できます。信頼できる提供元からのみインストールしてください。',
    installHint: 'インストール前にソースを確認してください。変更は新しいチャットに適用されます。'
  },
  sessionImport: {
    action: 'チャットを取り込む',
    search: '読み込み済みのチャットを検索',
    scanError: 'チャットを取得できません',
    emptyHelp: 'このマシン上の Claude Code と Codex のチャットがここに表示されます。',
    searchHelp: '別のタイトルやフォルダーを検索するか、チャットを追加で読み込んでください。',
    more: 'さらにチャットを読み込む',
    chooseHelp: 'チャットを選び、取り込む前に履歴を確認できます。'
  },
  titlebar: {
    searchTitle: 'チャット、ビュー、アクションを検索',
    unreadSessions: count => (count === 1 ? '未読チャット 1 件' : `未読チャット ${count} 件`)
  },
  skillDeepLink: {
    installDescription: 'このスキルは新しいチャットで利用できます。信頼できる提供元からのみインストールしてください。'
  },
  starmap: {
    emptyDesc: 'エージェントがスキルやメモリを蓄積すると、ここに表示されます。'
  },
  artifacts: {
    search: 'ライブラリを検索...',
    refresh: 'ライブラリを更新',
    refreshing: 'ライブラリを更新中',
    indexing: '最近のチャットの出力をインデックス中',
    noArtifactsTitle: 'ライブラリにはまだ何もありません',
    noArtifactsDesc: 'チャットで生成された画像やファイルの出力がここに表示されます。',
    failedLoad: 'ライブラリの読み込みに失敗しました',
    colSession: 'チャット',
    partialIndex: (failed, total) =>
      `ライブラリのインデックス作成中に、最近のチャット ${total} 件中 ${failed} 件をスキップしました。`
  },
  artifactPreview: {
    missingTitle: 'アイテムを利用できません',
    missingBody: 'このアイテムはライブラリにありません。'
  },
  statusStack: {
    control: {
      waitSession: target => `チャット ${target} を待機中`,
      loopDeferredNotice: '現在アクティブなゴールがチャットを制御しています。',
      continuationBusy:
        'ゴールを再開しました — チャットが実行中です。続行するには、まず現在の応答を停止してください（Stop ボタンまたは Esc）',
      controlUnavailable: msg => `チャットの制御を利用できません: ${msg}`
    },
    coding: {
      agentShip: 'エージェントにコミットと PR を任せる'
    }
  },
  onboarding: {
    headerTitle: 'エージェントのセットアップをしましょう'
  },
  interfaceMode: {
    hint: '表示される内容が変わるだけで、エージェントにできることは変わりません。',
    simple: {
      description:
        'エージェントと話すための表示。サイドバーとチャットのみ。ターミナル、ファイル、差分のペインは表示しません。'
    }
  },
  zones: {
    newSessionTab: '新しいチャットタブ'
  }
}
