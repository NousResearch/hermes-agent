// ko/04.ts — Korean translation of the `notifications, remoteDisplayBanner, billingBlock, sendDiagnostics, titlebar` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko04: TranslationOverrides = {
notifications: {
    sharedProfileWarning:
      '다른 Hermes 설치가 이 프로필을 사용하고 있어요. 두 설치가 같은 설정과 데이터를 공유하므로 변경 사항이 충돌할 수 있어요. 계속 진행하거나, 변경하기 전에 다른 설치를 닫아 주세요.',
    region: '알림',
    hide: '숨기기',
    show: '표시',
    more: count => `알림 ${count}개${count === 1 ? '' : ''} 더 보기`,
    clearAll: '모두 지우기',
    dismiss: '알림 닫기',
    details: '자세히',
    copyDetail: '세부 정보 복사',
    copyDetailFailed: '알림 세부 정보를 복사할 수 없어요',
    compressDeferredDone: '컨텍스트 압축 완료',
    backendOutOfDateTitle: '백엔드가 오래됐어요',
    backendOutOfDateMessage:
      'Hermes 백엔드가 이 데스크톱 빌드보다 오래되어 제대로 동작하지 않을 수 있어요. 업데이트해서 버전을 맞춰 주세요.',
    installMethodUnsupportedTitle: '지원되지 않는 설치 방식',
    updateHermes: 'Hermes 업데이트',
    updateReadyTitle: '업데이트 준비됨',
    updateReadyMessage: count => `새 변경 사항 ${count}개${count === 1 ? '' : ''}를 사용할 수 있어요.`,
    updateReadyMessageUnknown: '새 업데이트가 있어요.',
    updateReadyMessageAppInstaller: '새 버전의 Hermes가 준비됐어요. 지금 업데이트하면 Windows가 마무리해 줘요.',
    seeWhatsNew: "새로운 기능 보기",
    mcp: {
      needsAuthTitle: 'MCP 서버 재인증 필요',
      needsAuthMessage: name => `${name} MCP 재인증이 필요해요.`,
      errorTitle: 'MCP 서버에 연결할 수 없어요',
      errorMessage: name => `${name} MCP가 상태 확인에 실패했어요.`,
      signIn: '로그인',
      view: '보기',
      disable: '사용 안 함',
      disabledMessage: name => `${name} MCP를 사용 안 함으로 설정했어요. 언제든지 기능 → MCP에서 다시 사용할 수 있어요.`,
      disableFailed: name => `${name} MCP를 사용 안 함으로 설정할 수 없어요.`
    },
    errors: {
      elevenLabsNeedsKey: '음성 입력에는 ElevenLabs 키가 필요해요. 설정 → 키에서 추가해 주세요.',
      elevenLabsRejectedKey: "ElevenLabs가 API 키를 거부했어요. 설정 → 키에서 업데이트한 뒤 다시 시도해 주세요.",
      diskFull: '디스크가 가득 찼어요 — 공간을 확보한 뒤 다시 시도해 주세요.',
      storageFailure: "Hermes가 데이터 폴더에 저장하지 못했어요. 유지 관리를 열어 확인하고 복구해 주세요.",
      gatewayAuthFailed:
        '이 Hermes는 저장된 로그인 정보를 더 이상 받아들이지 않아요. 게이트웨이를 열어 다시 로그인하거나 새 액세스 토큰을 붙여 넣은 뒤 다시 시도해 주세요.',
      methodNotAllowed:
        "Hermes 백그라운드 서비스가 앱과 맞지 않아요. 아마 업데이트 후일 거예요. 재시작하면 해결돼요.",
      microphonePermission: '마이크 권한이 거부됐어요.',
      openaiRejectedApiKey: "OpenAI가 API 키를 거부했어요. 설정 → 키에서 업데이트한 뒤 다시 시도해 주세요.",
      openaiTtsNeedsKey: '음성 기능에는 OpenAI 키가 필요해요. 설정 → 키에서 추가해 주세요.',
      codeSkewRestartRequired:
        'Hermes가 업데이트됐지만 아직 이전 버전으로 실행 중이에요. 재시작해서 업데이트를 마무리해 주세요.',
      rpcOutOfSync: '앱과 백엔드의 버전이 서로 달라요. 둘 다 업데이트해 주세요.',
      restartHermesFailed: "Hermes를 재시작할 수 없어요"
    },
    actions: {
      restartHermes: 'Hermes 재시작',
      openKeys: '키 열기',
      openGateways: '게이트웨이 열기',
      openMaintenance: '유지 관리 열기'
    },
    voice: {
      configureSpeechToText: '음성 모드를 사용하려면 음성 텍스트 변환을 설정해 주세요.',
      couldNotStartSession: '음성 세션을 시작할 수 없어요',
      microphoneAccessDenied: '마이크 접근이 거부됐어요.',
      microphoneConstraintsUnsupported: '이 기기에서는 마이크 제약 조건을 지원하지 않아요.',
      microphoneFailed: '마이크 오류',
      microphoneInUse: '마이크를 다른 앱에서 이미 사용 중이에요.',
      microphonePermissionDenied: '마이크 권한이 거부됐어요.',
      microphoneStartFailed: '마이크 녹음을 시작할 수 없어요.',
      microphoneUnsupported: '이 런타임은 마이크 녹음을 지원하지 않아요.',
      noMicrophone: '마이크를 찾을 수 없어요.',
      noSpeechDetected: '음성이 감지되지 않았어요',
      playbackFailed: '음성 재생에 실패했어요',
      recordingFailed: '음성 녹음에 실패했어요',
      sayStopToEnd: phrase => `"${phrase}"라고 말하면 음성 채팅이 끝나요.`,
      transcriptionFailed: '음성 변환에 실패했어요',
      transcriptionUnavailable: '음성 변환은 아직 사용할 수 없어요.',
      tryRecordingAgain: '다시 녹음해 보세요.',
      unavailable: '음성을 사용할 수 없어요',
      liveEnded: '실시간 음성 세션이 종료됐어요',
      liveEndedConnectionLost: '실시간 음성 세션 연결이 끊어졌어요.',
      liveEndedClosed: '실시간 음성 세션이 서비스에서 종료됐어요.',
      liveError: '실시간 음성',
      liveDelegationFailed: '요청을 Hermes에 넘기지 못했어요',
      liveUnavailable: reason => `GPT-Live 음성 채팅을 사용할 수 없어요: ${reason}. 대신 음성 텍스트 변환을 사용해요.`
    },
    native: {
      approvalTitle: '승인 필요',
      approvalTitleNamed: session => `승인 필요 — ${session}`,
      approveAction: '승인',
      rejectAction: '거부',
      inputTitle: '입력 필요',
      inputTitleNamed: session => `입력 필요 — ${session}`,
      inputBody: 'Hermes가 응답을 기다리고 있어요.',
      turnDoneTitle: 'Hermes 작업 완료',
      turnDoneBody: '',
      turnErrorTitle: '턴 실패',
      backgroundDoneTitle: '백그라운드 작업 완료',
      backgroundFailedTitle: '백그라운드 작업 실패',
      creditsTitle: '크레딧'
    }
  },
remoteDisplayBanner: {
    message: reason =>
      `소프트웨어 렌더링 사용 중 — 원격 디스플레이가 감지됐어요 (${reason}). 깜박임을 막기 위해 GPU 가속을 사용하지 않아요.`
  },
billingBlock: {
    titleNous: 'Nous 크레딧 소진',
    titleProvider: provider => `크레딧 소진 — ${provider}`,
    fallbackMessage: '계정의 크레딧이 모두 소진됐어요. 계속하려면 크레딧을 충전해 주세요.',
    openBilling: '결제 열기',
    addCredits: '크레딧 충전',
    dismiss: '닫기'
  },
sendDiagnostics: {
    title: 'Nous에 진단 정보 보내기',
    privacyNotice:
      '디버그 번들을 Nous 내부 저장소에 업로드해요(공개 붙여넣기 공간이 아니에요). 시스템 정보(OS, 버전, 제공자, 어떤 API 키가 설정되어 있는지 — 키 자체는 절대 포함되지 않아요)와 전체 에이전트, 게이트웨이, 데스크톱 로그(각각 최대 512KB)가 포함되며, 여기에는 대화 내용, 도구 출력, 파일 경로가 들어 있을 수 있어요. 업로드 전에 비밀 값은 가려요. 이 번들은 Nous 직원과 허용된 Discord 모더레이터만 볼 수 있고 14일 후 자동으로 삭제돼요.',
    upload: '업로드',
    uploading: '업로드 중…',
    cancel: '취소',
    close: '닫기',
    copyLink: '링크 복사',
    uploadIdFallback: id => `보기 링크가 반환되지 않았어요 — 지원팀에 업로드 ID ${id}를 알려 주세요`,
    doneTitle: '진단 정보 전송됨',
    doneDescription:
      '번들이 비공개로 업로드됐어요. 지원 스레드에 아래 링크를 공유하면 팀에서 로그를 확인할 수 있어요.',
    failedTitle: '업로드 실패',
    failedHint:
      '터미널에서 `hermes debug share --nous`를 실행하거나, 업로드 없이 보고서를 출력하려면 `hermes debug share --local`을 실행해도 돼요.',
    handoffLead: '다음에서 논의를 이어가세요:',
    links: {
      github: 'GitHub 이슈',
      portal: 'Nous 포털 지원',
      discord: 'Discord'
    }
  },
titlebar: {
    hideSidebar: '사이드바 숨기기',
    showSidebar: '사이드바 표시',
    search: '검색',
    searchTitle: '세션, 뷰, 작업 검색',
    swapSidebarSides: '사이드바 좌우 바꾸기',
    hideRightSidebar: '오른쪽 사이드바 숨기기',
    showRightSidebar: '오른쪽 사이드바 표시',
    unreadSessions: count => (count === 1 ? '읽지 않은 세션 1개' : `읽지 않은 세션 ${count}개`),
    muteHaptics: '햅틱 끄기',
    unmuteHaptics: '햅틱 켜기',
    openSettings: '설정 열기',
    openStarmap: '메모리 그래프 열기',
    enterHud: 'HUD 모드',
    exitHud: 'HUD 모드 종료',
    resetHudLayout: 'HUD 크기와 위치 초기화',
    layoutEditor: '레이아웃 편집기',
    layoutEditorTitle: mod => `레이아웃 편집기 — ${mod}-클릭하면 레이아웃이 초기화돼요`
  },
}
