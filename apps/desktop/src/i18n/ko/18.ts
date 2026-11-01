// ko/18.ts — Korean translation of the `messaging` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko18: TranslationOverrides = {
messaging: {
    search: '메시징 검색...',
    loading: '메시징 플랫폼 불러오는 중...',
    loadFailed: '메시징 플랫폼을 불러오지 못했어요',
    states: {
      connected: '연결됨',
      connecting: '연결 중',
      disabled: '사용 안 함',
      fatal: '오류',
      gateway_stopped: '메시징 게이트웨이 중지됨',
      not_configured: '설정 필요',
      pending_restart: '재시작 필요',
      retrying: '다시 시도 중',
      startup_failed: '시작 실패'
    },
    unknown: '알 수 없음',
    hintPendingRestart: '이 변경 사항을 적용하려면 상태 표시줄에서 게이트웨이를 다시 시작해 주세요.',
    sharedListenerUrl: '공유 게이트웨이 리스너에서 제공:',
    hintGatewayStopped: '연결하려면 상태 표시줄에서 게이트웨이를 시작해 주세요.',
    credentialsSet: '자격 증명 설정됨',
    needsSetup: '설정 필요',
    gatewayStopped: '메시징 게이트웨이 중지됨',
    getCredentials: '자격 증명 발급받기',
    openSetupGuide: '설정 가이드 열기',
    required: '필수',
    recommended: '권장',
    advanced: count => `고급 (${count})`,
    noTokenNeeded: '이 플랫폼은 여기서 토큰이 필요하지 않아요. 위의 설정 가이드를 사용한 뒤 아래에서 활성화해 주세요.',
    enabled: '사용 중',
    disabled: '사용 안 함',
    unsavedChanges: '저장되지 않은 변경 사항',
    saving: '저장 중...',
    saveChanges: '변경 사항 저장',
    saved: '저장됨',
    replaceValue: '현재 값 바꾸기',
    openDocs: '문서 열기',
    clearField: key => `${key} 지우기`,
    enableAria: name => `${name} 사용`,
    disableAria: name => `${name} 사용 안 함`,
    platformEnabled: name => `${name} 사용 설정됨`,
    platformDisabled: name => `${name} 사용 안 함`,
    restartToApply: '이 변경 사항은 게이트웨이를 다시 시작한 후 적용돼요.',
    setupSaved: name => `${name} 설정 저장됨`,
    restartToReconnect: '새 자격 증명은 게이트웨이를 다시 시작한 후 적용돼요.',
    appliedLive: '실행 중인 게이트웨이에 적용했어요.',
    connectingLive: '실행 중인 게이트웨이가 새 자격 증명으로 연결 중이에요.',
    keyCleared: key => `${key} 지워짐`,
    setupUpdated: name => `${name} 설정을 업데이트했어요.`,
    failedUpdate: name => `${name} 업데이트 실패`,
    failedSave: name => `${name} 저장 실패`,
    failedClear: key => `${key} 지우기 실패`,
    pendingRequests: count => `대기 중인 요청 (${count})`,
    pendingAria: count => `${count}개의 대기 중인 페어링 ${count === 1 ? '요청' : '요청'}`,
    approvedUsers: count => `승인된 사용자 (${count})`,
    approve: '승인',
    approving: '승인 중...',
    revoke: '권한 해제',
    revoking: '해제 중...',
    revokeAria: name => `${name} 권한 해제`,
    revokeTitle: '액세스 권한 해제',
    revokeDesc: (name: string) => `${name} 님은 액세스 권한을 잃고 다음 메시지부터 인식되지 않아요.`,
    approvedUser: name => `${name} 승인됨`,
    approvedHint: '다음 메시지부터 자동으로 인식돼요.',
    revokedUser: name => `${name} 권한 해제됨`,
    failedApprove: name => `${name} 승인 실패`,
    failedRevoke: name => `${name} 권한 해제 실패`,
    pairingLockedOut: '승인 실패가 너무 많아 이 플랫폼이 잠겼어요. 나중에 다시 시도해 주세요.',
    waitingSince: minutes => (minutes < 1 ? '방금' : `${minutes}분 전`),
    restartNeeded: '저장했어요. 새 설정을 적용하려면 메시징 게이트웨이를 다시 시작해 주세요.',
    restartNow: '지금 재시작',
    restarting: '재시작 중…',
    restartFailedManual: "Hermes가 메시징 설정을 적용하려고 다시 시작하지 못했어요",
    restartFailedManualDetail: '다시 시도를 눌러 보세요. 계속 실패하면 로그를 열어 진단 정보를 보내 주세요.',
    restartAgain: '다시 재시작',
    openLogs: '로그 열기',
    telegramQr: {
      title: 'Telegram 봇 연결 방법 선택',
      subtitle: '두 방법 모두 직접 관리하는 봇을 연결하고, 자격 증명은 이 Hermes 설치에만 저장해요.',
      quickSetup: '빠른 설정',
      recommended: '권장',
      quickHelp:
        'QR 코드를 스캔하고 Telegram에서 확인하세요. Hermes가 봇을 만들고 Telegram 사용자 ID를 자동으로 감지해요.',
      createWithQr: 'QR로 만들기',
      starting: '시작 중…',
      replaceWarning:
        'Telegram 자격 증명이 이미 설정되어 있어요. 저장하면 새 QR 설정이나 봇 토큰이 현재 봇을 대체해요.',
      scanHint: '휴대폰의 Telegram 앱으로 스캔하거나 이 컴퓨터에서 링크를 여세요.',
      waiting: 'Telegram 응답 대기 중…',
      expiresIn: remaining => `만료까지 ${remaining}`,
      expired: '만료됨',
      openTelegram: 'Telegram 열기',
      ready: '봇 생성됨',
      allowedUsers: '허용된 사용자',
      ownerDetected: '소유자 감지됨',
      addAtLeastOne: 'Telegram 사용자 ID를 하나 이상 추가하세요.',
      userIdPlaceholder: 'Telegram 사용자 ID',
      add: '추가',
      numericOnly: '허용할 Telegram 사용자 ID는 숫자여야 해요.',
      saveAndRestart: '저장 후 재시작',
      applying: '저장 중…',
      pairingExpired: 'Telegram 페어링이 만료됐어요. 다시 시도하려면 새 QR 설정을 시작하세요.',
      stillWaiting: detail => `Telegram 응답을 계속 기다리는 중이에요. 다음 재시도: ${detail}`,
      savedRestarting: 'Telegram 저장됨. 게이트웨이 재시작 중…',
      savedRestartFailed: detail => `Telegram 저장됨. 게이트웨이 재시작 실패${detail}`
    },
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: '봇 토큰',
        help: '@BotFather로 봇을 만든 뒤 받은 토큰을 붙여 넣으세요.',
        placeholder: 'Telegram 봇 토큰 붙여 넣기'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: '허용할 Telegram 사용자 ID',
        help: '권장. @userinfobot에서 확인한 숫자 ID를 쉼표로 구분해 입력하세요. 이 설정이 없으면 누구나 봇에 DM을 보낼 수 있어요.'
      },
      TELEGRAM_PROXY: { label: '프록시 URL', help: 'Telegram이 차단된 네트워크에서만 필요해요.' },
      DISCORD_BOT_TOKEN: {
        label: '봇 토큰',
        help: 'Discord Developer Portal에서 애플리케이션을 만들고 봇을 추가한 뒤 토큰을 붙여 넣으세요.'
      },
      DISCORD_ALLOWED_USERS: {
        label: '허용할 Discord 사용자 ID',
        help: '권장. Discord 사용자 ID를 쉼표로 구분해 입력하세요.'
      },
      DISCORD_REPLY_TO_MODE: { label: '답장 방식', help: 'first, all, off 중 하나.' },
      DISCORD_ALLOW_ALL_USERS: {
        label: '모든 Discord 사용자 허용',
        help: '개발용으로만 사용하세요. true이면 허용 목록 없이 누구나 봇에 DM을 보낼 수 있어요.'
      },
      DISCORD_HOME_CHANNEL: {
        label: '홈 채널 ID',
        help: '봇이 먼저 보내는 메시지(크론 출력, 리마인더)를 전송하는 채널이에요.'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: '홈 채널 이름',
        help: '로그와 상태 출력에 표시할 홈 채널 이름이에요.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: '모든 iMessage 사용자 허용',
        help: 'true이면 BlueBubbles 허용 목록을 건너뛰어요.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: '모든 Mattermost 사용자 허용' },
      MATTERMOST_HOME_CHANNEL: { label: '홈 채널' },
      QQ_ALLOW_ALL_USERS: { label: '모든 QQ 사용자 허용' },
      QQBOT_HOME_CHANNEL: { label: 'QQ 홈 채널', help: '크론 전달에 사용할 기본 채널 또는 그룹이에요.' },
      QQBOT_HOME_CHANNEL_NAME: { label: 'QQ 홈 채널 이름' },
      SLACK_BOT_TOKEN: {
        label: 'Slack 봇 토큰',
        help: 'Slack 앱을 설치한 뒤 OAuth & Permissions에서 확인한 봇 토큰을 사용하세요.',
        placeholder: 'Slack 봇 토큰 붙여 넣기'
      },
      SLACK_APP_TOKEN: {
        label: 'Slack 앱 토큰',
        help: 'Socket Mode에 필요한 앱 레벨 토큰을 사용하세요.',
        placeholder: 'Slack 앱 토큰 붙여 넣기'
      },
      SLACK_ALLOWED_USERS: { label: '허용할 Slack 사용자 ID', help: '권장. Slack 사용자 ID를 쉼표로 구분해 입력하세요.' },
      MATTERMOST_URL: { label: '서버 URL', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: '봇 토큰' },
      MATTERMOST_ALLOWED_USERS: {
        label: '허용할 사용자 ID',
        help: '권장. Mattermost 사용자 ID를 쉼표로 구분해 입력하세요.'
      },
      MATRIX_HOMESERVER: { label: '홈서버 URL', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: '액세스 토큰' },
      MATRIX_USER_ID: { label: '봇 사용자 ID', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: '허용할 Matrix 사용자 ID',
        help: '권장. @user:server 형식의 사용자 ID를 쉼표로 구분해 입력하세요.'
      },
      SIGNAL_HTTP_URL: {
        label: 'Signal 브리지 URL',
        placeholder: 'http://127.0.0.1:8080',
        help: '실행 중인 signal-cli REST 브리지의 URL이에요.'
      },
      SIGNAL_ACCOUNT: { label: '전화번호', help: 'signal-cli 브리지에 등록한 번호예요.' },
      SIGNAL_ALLOWED_USERS: { label: '허용할 Signal 사용자', help: '권장. Signal 식별자를 쉼표로 구분해 입력하세요.' },
      WHATSAPP_ENABLED: {
        label: 'WhatsApp 브리지 사용',
        help: '아래 토글이 자동으로 설정해요. 필요하다고 확신하지 않으면 그대로 두세요.'
      },
      WHATSAPP_MODE: { label: '브리지 모드' },
      WHATSAPP_ALLOWED_USERS: {
        label: '허용할 WhatsApp 사용자',
        help: '권장. 전화번호나 WhatsApp ID를 쉼표로 구분해 입력하세요.'
      }
    },
    platformIntro: {}
  },
}
