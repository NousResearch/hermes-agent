// Notification settings copy used when the desktop locale loads.
export const jaNotifications = {
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
  testSent: 'テストを送信しました。表示されない場合は、OS の通知許可と集中モード／おやすみモードを確認してください。',
  testUnsupported: 'このシステムはネイティブ通知に対応していません。',
  muteSoundsTitle: 'インターフェース音をミュート',
  muteSoundsDesc:
    '完了・ウェイクワード・思考中の自動通知音、触覚フィードバック、OS 通知の音を無効にします。サウンドのプレビューと音声による応答は引き続き再生されます。',
  completionSoundTitle: '完了サウンド',
  completionSoundDesc: 'エージェントのターン終了時に再生されます。プリセットを選んでここで試聴できます。',
  completionSoundPreview: '試聴'
}
