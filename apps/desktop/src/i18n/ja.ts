import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale } from './define-locale'

export const ja = defineLocale({
  common: {
    save: '保存',
    saving: '保存中…',
    cancel: 'キャンセル',
    close: '閉じる',
    confirm: '確認',
    delete: '削除',
    refresh: '更新',
    retry: '再試行',
    on: 'オン',
    off: 'オフ'
  },

  language: {
    label: '言語',
    description: 'デスクトップインターフェイスの言語を選択します。',
    saving: '言語を保存中…',
    saveError: '言語の更新に失敗しました',
    switchTo: '言語を切り替え'
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
      gateway: 'ゲートウェイ',
      apiKeys: 'ツールとキー',
      keysTools: 'ツール',
      keysSettings: '設定',
      mcp: 'MCP',
      archivedChats: 'アーカイブ済みチャット',
      about: '情報'
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
      product: 'プロダクト',
      productDesc: '読みやすいツール活動と簡潔な要約を表示します。',
      technical: 'テクニカル',
      technicalDesc: '生のツール引数、結果、低レベルの詳細を含めます。',
      themeTitle: 'テーマ',
      themeDesc: 'デスクトップ専用のパレットです。選択したモードの上に適用されます。'
    },
    fieldLabels: defineFieldCopy({
      model: 'デフォルトモデル',
      model_context_length: 'コンテキストウィンドウ',
      fallback_providers: 'フォールバックモデル',
      toolsets: '有効なツールセット',
      timezone: 'タイムゾーン',
      display: {
        personality: '人格',
        show_reasoning: '推論ブロック'
      },
      agent: {
        max_turns: '最大エージェントステップ',
        image_input_mode: '画像添付',
        api_max_retries: 'API 再試行回数',
        service_tier: 'サービス階層',
        tool_use_enforcement: 'ツール使用の強制'
      },
      terminal: {
        cwd: '作業ディレクトリ',
        backend: '実行バックエンド',
        timeout: 'コマンドタイムアウト',
        persistent_shell: '永続シェル',
        env_passthrough: '環境変数の引き継ぎ'
      },
      file_read_max_chars: 'ファイル読み取り上限',
      tool_output: {
        max_bytes: 'ターミナル出力上限',
        max_lines: 'ファイルページ上限',
        max_line_length: '行長上限'
      },
      code_execution: {
        mode: 'コード実行モード'
      },
      approvals: {
        mode: '承認モード',
        timeout: '承認タイムアウト',
        mcp_reload_confirm: 'MCP 再読み込みの確認'
      },
      command_allowlist: 'コマンド許可リスト',
      security: {
        redact_secrets: 'シークレットを伏せる',
        allow_private_urls: 'プライベート URL を許可'
      },
      browser: {
        allow_private_urls: 'ブラウザーのプライベート URL',
        auto_local_for_private_urls: 'プライベート URL にはローカルブラウザーを使用'
      },
      checkpoints: {
        enabled: 'ファイルチェックポイント',
        max_snapshots: 'チェックポイント上限'
      },
      voice: {
        record_key: '音声ショートカット',
        max_recording_seconds: '最大録音時間',
        auto_tts: '応答を読み上げる'
      },
      stt: {
        enabled: '音声認識',
        provider: '音声認識プロバイダー',
        local: {
          model: 'ローカル文字起こしモデル',
          language: '文字起こし言語'
        },
        elevenlabs: {
          model_id: 'ElevenLabs STT モデル',
          language_code: 'ElevenLabs 言語',
          tag_audio_events: '音声イベントをタグ付け',
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
          voice_id: 'ElevenLabs 音声',
          model_id: 'ElevenLabs モデル'
        }
      },
      memory: {
        memory_enabled: '永続メモリ',
        user_profile_enabled: 'ユーザープロファイル',
        memory_char_limit: 'メモリ予算',
        user_char_limit: 'プロファイル予算',
        provider: 'メモリプロバイダー'
      },
      context: {
        engine: 'コンテキストエンジン'
      },
      compression: {
        enabled: '自動圧縮',
        threshold: '圧縮しきい値',
        target_ratio: '圧縮目標',
        protect_last_n: '保護する直近メッセージ'
      },
      delegation: {
        model: 'サブエージェントモデル',
        provider: 'サブエージェントプロバイダー',
        max_iterations: 'サブエージェントターン上限',
        max_concurrent_children: '並列サブエージェント',
        child_timeout_seconds: 'サブエージェントタイムアウト',
        reasoning_effort: 'サブエージェント推論強度'
      },
      updates: {
        non_interactive_local_changes: 'アプリ内更新時のローカル変更'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'コンポーザーで別のモデルを選ばない限り、新しいチャットで使用されます。',
      model_context_length: '0 のままにすると、選択したモデルから検出されたコンテキストウィンドウを使用します。',
      fallback_providers: 'デフォルトモデルが失敗したときに試す provider:model 形式のバックアップです。',
      display: {
        personality: '新しいセッションのデフォルトのアシスタントスタイルです。',
        show_reasoning: 'バックエンドが推論内容を提供したときに表示します。'
      },
      timezone: 'Hermes がローカル時刻のコンテキストを必要とするときに使用します。空欄ならシステムのタイムゾーンを使います。',
      agent: {
        image_input_mode: '画像添付をモデルへ送る方法を制御します。',
        max_turns: 'Hermes が 1 回の実行を停止するまでのツール呼び出しターン上限です。'
      },
      terminal: {
        cwd: 'ツールとターミナル作業のデフォルトプロジェクトフォルダーです。',
        persistent_shell: 'バックエンドが対応している場合、コマンド間でシェル状態を保持します。',
        env_passthrough: 'ツール実行へ渡す環境変数です。'
      },
      code_execution: {
        mode: 'コード実行を現在のプロジェクトにどれだけ厳密に制限するかを設定します。'
      },
      file_read_max_chars: 'Hermes が 1 回のファイル読み取りで取得できる最大文字数です。',
      approvals: {
        mode: '明示的な承認が必要なコマンドを Hermes がどう扱うかを設定します。',
        timeout: '承認プロンプトがタイムアウトするまで待つ時間です。'
      },
      security: {
        redact_secrets: '検出したシークレットを、可能な限りモデルから見える内容から隠します。'
      },
      checkpoints: {
        enabled: 'ファイル編集前にロールバック用スナップショットを作成します。'
      },
      memory: {
        memory_enabled: '将来のセッションに役立つ永続メモリを保存します。',
        user_profile_enabled: 'ユーザーの好みをまとめた簡潔なプロファイルを維持します。'
      },
      context: {
        engine: '長い会話がコンテキスト上限に近づいたときの管理戦略です。'
      },
      compression: {
        enabled: '会話が大きくなったとき、古いコンテキストを要約します。'
      },
      voice: {
        auto_tts: 'アシスタントの応答を自動で読み上げます。'
      },
      stt: {
        enabled: 'ローカルまたはプロバイダーによる音声文字起こしを有効にします。',
        elevenlabs: {
          language_code: '任意の ISO-639-3 言語コードです。空欄なら ElevenLabs が自動検出します。'
        }
      },
      updates: {
        non_interactive_local_changes:
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
    }
  },

  skills: {
    all: 'すべて',
    noDescription: '説明はありません。'
  },

  profiles: {
    newProfile: '新しいプロファイル',
    noProfiles: 'プロファイルが見つかりません。',
    skills: count => `${count} スキル`,
    defaultBadge: 'デフォルト',
    rename: '名前を変更',
    saveSoul: 'SOUL を保存',
    cloneFromDefault: 'デフォルトプロファイルから設定を複製',
    invalidName: hint => `無効なプロファイル名。${hint}`,
    nameRequired: '名前は必須です',
    created: '作成しました',
    renamed: '名前を変更しました',
    deleted: '削除しました',
    soulSaved: 'SOUL.md を保存しました'
  },

  cron: {
    last: '前回',
    next: '次回',
    resume: '再開',
    pause: '一時停止',
    triggerNow: '今すぐ実行',
    namePlaceholder: '例: 日次サマリー',
    promptPlaceholder: '実行ごとにエージェントが行う内容は？'
  }
})
