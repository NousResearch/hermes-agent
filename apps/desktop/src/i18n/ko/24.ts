// ko/24.ts — Korean translation of the `updates, handoffTour, guidedGreeting` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko24: TranslationOverrides = {
  updates: {
    discontinuedTitle: '이 Hermes 빌드는 더 이상 지원되지 않아요',
    discontinuedBody:
      '이 Hermes 빌드는 더 이상 지원되지 않으며 오류가 발생할 수 있어요. 앱을 제거해 주세요. 데이터는 디스크에 유지돼요.',
    channels: { stable: 'Stable', canary: 'Canary' },
    bundleSwapPending: '업데이트를 완료하려면 다시 시작하세요',
    bundleSwapPendingDesc:
      '업데이트된 앱이 이미 설치되었어요. 적용하려면 Hermes를 다시 시작하기만 하면 돼요. 대화와 설정은 그대로 유지돼요.',
    bundleSwapPendingAction: 'Hermes 다시 시작',
    stages: {
      idle: '준비 중…',
      prepare: '준비 중…',
      fetch: '다운로드 중…',
      pull: '거의 다 되었어요…',
      pydeps: '마무리 중…',
      update: 'Hermes 업데이트 중…',
      rebuild: '데스크톱 앱 다시 빌드 중…',
      restart: 'Hermes 다시 시작 중…',
      done: '업데이트 완료',
      manual: '터미널에서 업데이트',
      guiSkew: '데스크톱 앱 업데이트',
      error: '업데이트 일시 중지됨'
    },
    checking: '업데이트 확인 중…',
    checkFailedTitle: '업데이트를 확인할 수 없어요',
    tryAgain: '다시 시도',
    notAvailableTitle: '업데이트를 사용할 수 없어요',
    unsupportedMessage: '이 버전의 Hermes는 앱 내에서 자체 업데이트할 수 없어요.',
    connectionRetry:
      'Hermes가 업데이트 서버에 연결할 수 없었어요. 인터넷 연결을 확인하고 다시 시도해 주세요. 원격 Hermes를 사용하는 경우 온라인 상태인지 확인해 주세요.',
    gitUnusable: '이 컴퓨터에서 Git을 실행할 수 없어 업데이트를 확인할 수 없었어요.',
    connectionSettings: '연결 설정',
    openDownloadPage: '다운로드 페이지 열기',
    latestBody: '최신 버전을 사용하고 있어요.',
    latestBodyBackend: '백엔드가 최신 버전을 실행 중이에요.',
    allSetTitle: '모두 최신 상태예요',
    availableTitle: '새로운 업데이트가 있어요',
    availableBody: '새로운 버전의 Hermes를 설치할 준비가 되었어요.',
    availableTitleBackend: '백엔드 업데이트가 있어요',
    availableBodyBackend: '연결된 Hermes 백엔드의 최신 버전을 설치할 준비가 되었어요.',
    availableBodyNoChangelog: '최신 버전이 준비되었어요. 이 설치 유형에서는 릴리스 노트를 사용할 수 없어요.',
    availableBodyAppInstaller:
      '새로운 버전의 Hermes가 준비되었어요. Hermes가 닫히고 Windows에서 업데이트를 완료한 후 자동으로 다시 열려요.',
    updateNow: '지금 업데이트',
    maybeLater: '나중에',
    moreChanges: count => `+ ${count}개의 변경 사항이 더 포함되어 있어요.`,
    manualTitle: '터미널에서 업데이트',
    manualBody: '명령줄에서 Hermes를 설치했으므로 업데이트도 터미널에서 실행돼요. 다음 명령어를 터미널에 붙여넣으세요:',
    manualPickedUp: '다음에 실행할 때 새 버전이 적용돼요.',
    guiSkewTitle: '데스크톱 앱 업데이트',
    guiSkewBody:
      '백엔드는 업데이트되었지만 데스크톱 앱 패키지는 변경되지 않았어요. 일치하도록 Hermes 데스크톱 앱(AppImage / .deb / .rpm)을 업데이트하거나 다시 설치해 주세요.',
    copy: '복사',
    copied: '복사됨',
    done: '완료',
    applyingBody:
      '별도 창에서 Hermes 업데이트가 진행되며, 완료되면 자동으로 다시 열려요. 업데이트 중에는 직접 Hermes를 다시 열지 마세요.',
    applyingBodyBackend:
      '원격 백엔드에서 업데이트를 적용 중이며 곧 다시 시작돼요. 준비되면 Hermes가 자동으로 다시 연결돼요.',
    applyingClose: '업데이트가 진행되는 동안 이 창이 닫히며, 완료되면 Hermes가 자동으로 다시 열려요.',
    applyingBodyAppInstaller:
      'Hermes가 닫히고 Windows에서 업데이트를 완료해요. 완료되면 자동으로 다시 열리므로 아무것도 하지 않으셔도 돼요.',
    applyingCloseAppInstaller: '이 창이 닫히고 Windows에서 업데이트를 완료한 후 Hermes가 자동으로 다시 열려요.',
    checkUnknownTitleAppInstaller: '업데이트를 확인할 수 없어요',
    checkUnknownBodyAppInstaller:
      '지금은 Windows에서 업데이트를 확인할 수 없어요. Hermes를 다시 시작할 때도 업데이트가 자동으로 설치돼요.',
    errorTitle: '업데이트가 완료되지 않았어요',
    errorBody: '걱정하지 마세요. 손실된 데이터는 없어요. 지금 다시 시도할 수 있어요.',
    blockerTitle: 'Hermes를 업데이트하기 위해 로컬 미리보기를 닫을까요?',
    blockerBody:
      '업데이트하기 전에 Hermes가 이 로컬 미리보기를 중지해야 해요. 파일이 수정되거나 삭제되지는 않아요.',
    foreignBlockerTitle: 'Hermes를 업데이트하기 위해 다른 프로세스를 닫아주세요',
    foreignBlockerBody:
      'Hermes가 이러한 프로세스를 자동으로 안전하게 닫을 수 없어요. 각 프로세스를 사용하는 앱, 터미널 또는 서비스를 닫은 후 업데이트를 다시 시도해 주세요.',
    mixedBlockerBody:
      'Hermes가 아래 목록의 로컬 미리보기를 닫을 수 있어요. 업데이트를 계속하려면 다른 프로세스를 수동으로 닫아야 해요.',
    closePreviewsAndUpdate: '미리보기 닫고 업데이트',
    closePreviewsAndCheckAgain: '미리보기 닫고 다시 확인',
    localPreview: '로컬 미리보기',
    portLabel: port => `포트 ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: '기술 세부사항',
    notNow: '지금 안 함',
    clientAlsoBehindTitle: '데스크톱 앱이 이전 버전이에요',
    clientAlsoBehindMessage:
      '백엔드는 최신 상태이지만 데스크톱 앱은 아직 이전 버전이에요. 최신 수정 사항을 적용하려면 업데이트해 주세요.',
    clientAlsoBehindAction: '데스크톱 앱 업데이트',
    everythingDispatched: '업데이트 요청됨',
    everythingSkipped: '건너뜀',
    everythingRowFailed: '업데이트 실패',
    everythingFanoutFailedTitle: '다른 인스턴스를 업데이트할 수 없어요',
    changeLogNew: '새로운 기능',
    changeLogFixed: '수정된 항목',
    changeLogFaster: '성능 향상',
    changeLogImproved: '개선된 항목',
    changeLogOther: '기타 개선 사항',
    changeLogFallbackLabel: '이번 업데이트 내용',
    changeLogFallbackItem: '개선 사항 및 버그 수정',
    applyStatus: {
      preparing: '백엔드 업데이트 중…',
      pulling: '백엔드 업데이트 중…',
      restarting: '업데이트를 적용하기 위해 백엔드를 다시 시작하는 중…',
      notAvailable: '이 백엔드에서는 업데이트를 사용할 수 없어요.',
      failed: '백엔드 업데이트에 실패했어요.',
      noReturn: '백엔드가 다시 온라인 상태가 되지 않았어요. 업데이트가 완료되지 않았을 수 있으니 백엔드 호스트를 확인해 주세요.'
    },
    // Update-status overlay + version-details (mechanism-aware update UI).
    appName: 'Hermes',
    version: (value: string) => `버전 ${value}`,
    versionUnavailable: '버전 정보 없음',
    checkNow: '지금 확인',
    seeWhatsNew: '새로운 기능 보기',
    releaseNotes: '릴리스 노트',
    onLatest: '최신 버전을 사용하고 있어요.',
    installing: '현재 업데이트를 설치하는 중이에요.',
    cantReach: '업데이트 서버에 연결할 수 없었어요.',
    tapCheck: '업데이트를 확인하려면 "지금 확인"을 누르세요.',
    updateReady: count => `새로운 업데이트가 준비되었어요 (${count}개 변경 사항 포함).`,
    updateReadyUnknown: '새로운 업데이트가 준비되었어요.',
    availableBodyRelease: tag => `버전 ${tag}을(를) 설치할 준비가 되었어요.`,
    lastChecked: age => `마지막 확인: ${age}`,
    never: '확인 기록 없음',
    justNow: '방금',
    minAgo: count => `${count}분 전`,
    hoursAgo: count => `${count}시간 전`,
    daysAgo: count => `${count}일 전`,
    justNowSuffix: ' · 방금',
    bundleOutOfSync: '앱 빌드가 이전 버전이에요',
    bundleOutOfSyncDesc:
      'Hermes 런타임은 업데이트되었지만 데스크톱 앱 자체는 이전 빌드예요. 최신 수정 사항을 적용하려면 업데이트해 주세요.',
    bundleOutOfSyncAction: '설치 프로그램 다운로드',
    checkingShort: '확인 중…',
    releaseAvailable: tag => `버전 ${tag}을(를) 사용할 수 있어요.`,
    versionDetailsTitle: '버전 세부정보',
    versionDetailsBody: '이 설치는 앱 외부에서 관리돼요. 처음 설치했던 방법과 동일하게 업데이트해 주세요.',
    versionDetailsVersion: '버전',
    versionDetailsCommit: '커밋',
    versionDetailsBuildOrigin: '빌드 출처',
    versionDetailsDistribution: '배포 방식',
    versionDetailsDistributionDesktop: '데스크톱 앱',
    versionDetailsDistributionDesktopMsix: '데스크톱 앱 (MSIX)',
    versionDetailsDistributionDesktopInstaller: '데스크톱 앱 (설치 프로그램)',
    versionDetailsDistributionSourceInstaller: '소스 (설치 스크립트)',
    versionDetailsDistributionSourceInstallerDesktop: '소스 (설치 스크립트) + hermes desktop',
    versionDetailsDistributionSource: '소스',
    versionDetailsDistributionSourceDesktop: '소스 + hermes desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: '런타임',
    versionDetailsRuntimeEmbedded: '임베디드 런타임',
    versionDetailsRuntimeExternal: '외부 (시스템 런타임 사용)',
    versionDetailsInstallId: '설치 ID',
    versionDetailsUncommittedChanges: '커밋되지 않은 변경 사항'
  },
  handoffTour: {
    profileTitle: '첫 번째 작업은 default 프로필에서 실행돼요',
    profileText:
      '이 레일에서 프로필을 전환할 수 있어요. 지금 켜져 있는 것은 작업 세션이 있는 default 프로필이에요. 다른 하나는 환영 대화가 있는 setup 프로필이에요.',
    sessionsTitle: '프로필마다 자체 세션이 보관돼요',
    sessionsText:
      '이 목록은 default 프로필에 속해 있어요. 새 세션을 시작하면 현재 선택된 프로필에서 세션이 생성돼요. 레일에서 프로필을 전환하면 목록도 함께 변경돼요.',
    stayTitle: '언제든 클릭 한 번으로 Hermes를 부를 수 있어요',
    stayText: '도움이 필요할 때마다 setup 프로필로 전환하여 Welcome to Hermes를 열어보세요. 언제나 그곳에 준비되어 있어요.'
  },
  guidedGreeting: {
    line: "안녕하세요, 어서 오세요! 저는 Hermes예요. 환경을 준비할 수 있도록 2분만 시간을 주세요. 준비가 끝나면 원하시는 작업을 바로 도와드릴게요.\n\n그전에, 어떻게 불러드리면 될까요?",
    nameSuggestion: (name: string) => `(원하신다면 편하게 ${name}(으)로 불러드릴 수도 있어요.)`
  }
}
