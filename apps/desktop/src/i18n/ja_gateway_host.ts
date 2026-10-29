import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Gateways copy where the connection cannot change, spread into ja.ts.
export const jaGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'ゲートウェイ設定は利用できません',
  unavailableDesc: 'デスクトップ IPC ブリッジはゲートウェイ設定を公開していません。',
  webappHostTitle: 'Hermes ホスト',
  webappHostDesc:
    'Webapp は常に配信元の Hermes ホストを使用します。ゲートウェイの切り替え、Hermes Cloud へのサインイン、保存済み接続の管理は Hermes Desktop アプリで行ってください。',
  webAccess: {
    title: 'Web アクセス',
    intro:
      'スマートフォンや別のコンピューターから、Nous アカウントでサインインしてこのコンピューターの Hermes を開きます。サインインした人はこのコンピューターでコマンドを実行できます。',
    failed: 'Web アクセスを変更できませんでした',
    running: '実行中',
    stopped: '停止中',
    starting: '起動中',
    stopping: '停止処理中',
    restartNeeded: '再起動が必要',
    unavailable: '利用できません',
    notRunningDetail: '起動すると、別のデバイスから Hermes を開けます。',
    needsNous: '先に「プロバイダー」で Nous アカウントにサインインしてください。',
    runningLan: '同じネットワーク上のデバイスでこのアドレスを開いてください。',
    runningPublic: 'あなたの URL 経由で提供中です。',
    runningLocal:
      'このコンピューターからのみ開けます。他のデバイスから開くには、停止してからもう一度起動してください。',
    pendingOn: '保存しました。ゲートウェイの再起動後に Web アプリが提供されます。',
    pendingOff: 'オフにしましたが、ゲートウェイは再起動するまで Web アプリを提供し続けます。',
    unavailableDetail: reason => `ゲートウェイが Web アプリを提供できませんでした: ${reason}`,
    restartBusy:
      '保存しました。チャットが実行中のため、ゲートウェイは再起動しませんでした。終了したら「ゲートウェイを再起動」を選んでください。',
    restartFailed:
      '保存しましたが、ゲートウェイを再起動できませんでした。「ゲートウェイを再起動」を選んで再試行してください。',
    restartFailedBecause: reason => `ゲートウェイを再起動できませんでした: ${reason}`,
    startTimedOut: 'ゲートウェイが 90 秒以内に Web アプリを提供しませんでした。',
    stopTimedOut: 'ゲートウェイが 90 秒以内に Web アプリの提供を停止しませんでした。',
    start: '開始',
    stop: '停止',
    restartGateway: 'ゲートウェイを再起動',
    modeLan: 'このネットワーク',
    modeLanDetail: url => `Wi-Fi 上のデバイスで ${url} を開きます。信頼できるネットワークでのみ使用してください。`,
    lanUnencrypted:
      '接続は暗号化されていません。同じネットワーク上の他の人が送受信の内容を読み取れる可能性があります。',
    noLan: 'このコンピューターには、他のデバイスから到達できるネットワークアドレスがありません。',
    modePublic: '独自の URL',
    modePublicDetail: port =>
      `Tailscale Serve やトンネルなど、すでに運用している HTTPS アドレスで、http://127.0.0.1:${port} に転送するもの。`
  }
}
