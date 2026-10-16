/** Bulk machine-translated ko overlay — settings keys (Gemini 3.5 Flash-Lite, then spot-checked). */
import type { TranslationOverrides } from './define-locale'

export const koSettingsOverrides = {
  billingBlock: {
    addCredits: '크레딧 추가',
    dismiss: '닫기',
    fallbackMessage: '크레딧이 부족합니다. 계속 사용하려면 크레딧을 추가하세요.',
    openBilling: '결제 열기',
    titleNous: 'Nous 크레딧 소진됨'
  },
  freeTier: {
    alreadySignedInBody: '이 Hermes는 이미 Nous 계정에 로그인되어 있습니다.',
    alreadySignedInHeading: '이미 로그인되어 있습니다.',
    begin: '시작',
    busyHeading: '거의 다 되었습니다',
    change: '변경',
    codeBody: '로그인을 완료하려면 브라우저에 이 코드를 입력하세요.',
    completedBody: '이제 계정에 추론 및 도구가 포함됩니다.',
    copyLink: '링크 복사',
    defaultModel: '기본 모델',
    didNotComplete: '로그인이 완료되지 않았습니다',
    dismiss: '닫기',
    doNotShare: '이 코드를 공유하지 마세요.',
    done: '완료',
    errorBody: '로그인이 완료되지 않았습니다. 준비되었을 때 다시 시도하세요.',
    finishingBody: '브라우저에서 승인되었습니다.계정 토큰을 수집하는 중입니다.',
    finishingHeading: '로그인 완료 중…',
    notNow: '나중에',
    openModelPicker: '모델 선택기 열기',
    otherProviders: '기타 공급자',
    providerName: 'Nous',
    providerRowPitch: 'Nous 계정으로 로그인하여 더 많은 모델과 도구를 잠금 해제하세요.',
    providerRowTitle: 'Nous · 무료 티어',
    readyCaption: '무료 · 커넥터 포함',
    readyTitle: 'Hermes가 준비되었습니다.',
    rejectedBody: '문제가 없습니다. 여전히 무료 Nous 서비스를 이용 중입니다. 준비될 때 로그인하세요.',
    retiredBody:
      '로그인이 완료되기 전에 세션이 종료되었습니다. Hermes가 새 세션을 시작합니다. 준비되면 다시 로그인하세요.',
    settingUp: '무료 추론 설정 중…',
    setupFailed: {
      gateClosed:
        '이 버전의 Hermes는 Nous 계정 없이는 시작할 수 없습니다. 로그인하거나 계정을 만드세요. 무료이며 단 1분이면 됩니다.',
      generic:
        'Hermes가 로그인 없이 무료 액세스를 설정할 수 없습니다. 로그인은 무료이며, 다른 공급자에 연결할 수도 있습니다.',
      locked: '로그인하지 않으면 이 세션을 계속할 수 없습니다. 계속하려면 로그인하거나 무료 Nous 계정을 만드세요.',
      paused:
        '로그인하지 않고 Hermes를 사용하는 기능이시 잠시 일시 중지되었습니다. Hermes가 계속 확인합니다. 로그인은 무료이며 즉시 이용하실 수 있습니다.',
      powRequired:
        'Nous 서버가 작업 증명(PoW)을 요청했지만, 현재 Agent에 구현되어 있지 않습니다. 계속하려면 로그인하거나 무료 Nous 계정을 만드세요.',
      retrying: '다시 시도하는 중…',
      serverError:
        "Nous 서비스에 일시적인 문제가 발생했습니다. 잠시 후 '다시 시도'를 누르거나 당분간 다른 공급자에 연결하세요.",
      signInBelow: '로그인은 무료입니다. 아래에서 Nous를 선택하세요.',
      tryAgain: '다시 시도',
      unreachable:
        "Hermes가 Nous 서비스에 연결할 수 없습니다. 인터넷 연결을 확인한 후 '다시 시도'를 누르거나, 당분간 다른 공급자에 연결하세요."
    },
    signIn: '로그인',
    signInHeading: 'Nous 계정으로 로그인하여 더 많은 모델과 도구를 잠금 해제하세요.',
    signInInstead: '대신 Nous 계정으로 로그인',
    signedIn: '로그인됨.',
    startAgain: '다시 시작',
    stripBody: '모델 선택기를 열어 사용해 보거나 Nous 계정으로 로그인하세요.',
    stripTitle: '이제 무료 Nous 추론 및 커넥터를 사용할 수 있습니다.',
    supersededBody: '최신 로그인 코드가 이 코드를 대체했습니다. 최신 코드를 사용하거나 다시 시작하세요.',
    timedOutBody: '준비되었을 때 다시 시작하세요. 여전히 무료 Nous 서비스를 이용 중입니다.',
    timedOutHeading: '로그인 링크가 만료되었습니다',
    tryAgain: '다시 시도',
    unreachableBody:
      '로그인을 완료하기 위해 Hermes가 Nous 서비스에 연결할 수 없습니다. 인터넷 연결을 확인하고 다시 시도하세요. 세션은 그대로 유지되어 있습니다.',
    waiting: '로그인 대기 중…'
  },
  interfaceMode: {
    advanced: {
      description: '개발자용. 터미널, 파일, diff, 상태 표시줄 및 설정한 레이아웃을 제공합니다.',
      label: '고급'
    },
    hint: 'Hermes가 할 수 있는 일이 아니라 화면에 표시되는 방식을 변경합니다.',
    sessionNote:
      '단순 모드에서 설정했습니다. 여기에서의 변경 사항은 이번 세션 동안 유지됩니다. 나만의 설정으로 만들려면 고급 모드로 전환하세요.',
    simple: {
      description: 'Hermes와 대화하기 위한 모드입니다. 사이드바와 채팅만 표시되며 터미널, 파일, diff 창이 없습니다.',
      label: '단순'
    },
    title: '인터페이스 모드'
  },
  modelAssignment: {
    confirmAction: '확인',
    confirmDetail: '이 트레이드오프를 수락하는 경우에만 확인하세요.',
    confirmTitle: '모델 선택 경고',
    declined: '모델 변경이 취소되었습니다 — 데이터 학습 계층 경고를 거부했습니다.',
    saveFailed: 'Hermes가 해당 모델 변경 사항을 저장하지 못했습니다.'
  },
  modelPicker: {
    addCustomModelAction: '사용자 정의 모델 추가...',
    addProvider: '공급자 추가',
    current: '현재:',
    customModel: '사용자 정의 모델',
    customModelPlaceholder: '모델 ID 입력 (예: openai/gpt-5)',
    downloading: '다운로드 중',
    free: '무료',
    freeTier: '무료 티어',
    loadFailed: '모델을 불러올 수 없습니다',
    loadingIntoMemory: '메모리에 로드하는 중',
    localDownloadsHeading: '로컬',
    noAuthenticatedProviders: '인증된 공급자가 없습니다.',
    noModels: '모델을 찾을 수 없습니다.',
    priceTitle: '백만 토큰당 입력 / 출력 가격',
    pro: 'Pro',
    proNeedsSubscription: 'Pro 모델은 유료 Nous 구독이 필요합니다.',
    search: '공급자 및 모델 필터링...',
    title: '모델 전환',
    unknown: '(알 수 없음)',
    wasPrice: '기존 가격'
  },
  modelVisibility: {
    addCustomModel: '사용자 정의 모델 추가',
    addProvider: '공급자 추가...',
    noAuthenticatedProviders: '인증된 공급자가 없습니다.',
    removeCustomModel: '사용자 정의 모델 제거',
    resetAction: '재설정',
    resetConfirm: '모델 표시 설정을 기본값으로 재설정하시겠습니까?',
    resetDescription:
      '표시 및 숨김 처리된 모델 선택이 지워지고 모든 공급자의 기본 목록이 복원됩니다. 추가한 사용자 정의 모델은 유지되고 표시됩니다.',
    resetToDefaults: '기본값으로 재설정',
    search: '모델 검색',
    title: '모델'
  },
  settings: {
    about: {
      updates: '업데이트'
    },
    appearance: {
      appActionsDesc:
        '제목 표시줄에 설정, 레이아웃, HUD가 위치하는 곳입니다. 오른쪽은 왼쪽에 탭을 위한 공간을 남겨둡니다.',
      appActionsLeft: '왼쪽',
      appActionsRight: '오른쪽',
      appActionsTitle: '앱 동작',
      backdropDesc: '대화 뒤에 흐리게 표시되는 동상 이미지입니다.',
      backdropTitle: '채팅 배경',
      chatFontDesc:
        '채팅 및 앱 전반에 사용할 설치된 폰트를 선택하세요. OpenDyslexic 같은 가독성 폰트에 유용합니다. 비워두면 테마의 폰트를 사용합니다.',
      chatFontPlaceholder: 'OpenDyslexic 또는 CSS 폰트 스택',
      chatFontPreview: '미리보기',
      chatFontReset: '테마 폰트 사용',
      chatFontSample: '웬 여우가 게으른 개를 가뿐히 뛰어넘었다. 0123456789',
      chatFontTitle: '채팅 폰트',
      colorMode: '색상 모드',
      colorModeDesc: '고정 모드를 선택하거나 Hermes가 시스템 설정을 따르도록 하세요.',
      composerPopoutDesc: '작성기를 도크에서 드래그하여 분리할 수 있습니다. 끄면 하단에 고정됩니다.',
      composerPopoutTitle: '플로팅 작성기',
      embedsAlways: '항상',
      embedsAsk: '묻기',
      embedsDesc:
        "서드파티 사이트(YouTube, X, …)에서 리치 미리보기를 불러옵니다. '묻기'는 각 항목을 허용할 때까지 자리표시자를 표시하고, '항상'은 자동으로 불러오며, '끄기'는 일반 링크로 유지합니다.",
      embedsOff: '끄기',
      embedsTitle: '인라인 임베드',
      fileBrowserDesc:
        '워크스페이스가 열려 있을 때 채팅 옆에 파일 브라우저를 표시합니다. 제목 표시줄 토글도 이 설정을 변경합니다.',
      fileBrowserTitle: '파일 브라우저',
      hideCodeDiffsDesc: '코드 없이 추가/삭제된 줄 수가 표시된 인라인 도구 행으로 파일 수정을 표시합니다.',
      hideCodeDiffsTitle: '코드 변경 내용 숨기기',
      hideThreadTimelineDesc: '각 대화의 오른쪽 가장자리에 있는 탐색 바를 숨깁니다.',
      hideThreadTimelineTitle: '스레드 타임라인 바 숨기기',
      importedBadge: '가져옴',
      installButton: '설치',
      installDesc:
        'Marketplace 확장 ID(예: dracula-theme.theme-dracula)를 붙여넣어 해당 색상 테마를 데스크톱 팔레트로 변환하세요.',
      installError: '해당 테마를 설치할 수 없습니다.',
      installPlaceholder: 'publisher.extension',
      installTitle: 'VS Code에서 설치',
      installing: '설치 중…',
      intro: '데스크톱 전용. 모드는 밝기이며, 테마는 팔레트와 채팅 크롬을 결정합니다.',
      introSplashDesc: '빈 채팅에 표시되는 워드마크와 프롬프트입니다.',
      introSplashTitle: '인트로 스플래시',
      modelPricingDesc: '모델 선택기에 백만 토큰당 입력, 출력, 캐시 읽기 가격을 표시합니다.',
      modelPricingTitle: '모델 가격',
      pet: {
        chooseDesc: '선택하면 (필요시) 설치되고 활성화됩니다.',
        chooseTitle: '펫 선택',
        deleteBody: '이 작업은 펫을 영구적으로 삭제하며 다시 설치할 수 없습니다.',
        deleteConfirm: '삭제',
        generatedTag: '생성됨',
        installedTag: '설치됨',
        intro:
          '앱 위를 떠다니며 Hermes의 동작(도구 실행 중 달리기, 성공 시 축하, 오류 시 뾰루퉁함)에 반응하는 애니메이션 petdex 마스코트를 입양하세요.',
        noneAvailable: '현재 활성화할 수 있는 펫이 없습니다.',
        renamePlaceholder: '펫 이름 지정',
        renameSave: '저장',
        renameTitle: '펫 이름 변경',
        restartHint:
          '펫 기능을 사용하려면 간단한 재시작이 필요합니다. 실행 중인 앱이 이 기능이 추가되기 전에 시작되었습니다. Hermes를 종료했다가 다시 연 후 여기로 돌아오세요.',
        roamDesc: '유휴 상태일 때 펫이 창 안을 자유롭게 돌아다니도록 합니다.',
        roamTitle: '돌아다니기',
        scaleDesc: '플로팅 마스코트의 크기를 조절합니다. 모든 곳에 즉시 적용됩니다.',
        scaleTitle: '크기',
        searchPlaceholder: '펫 검색…',
        title: '펫',
        turnOffFailed: '펫을 끌 수 없습니다.',
        turnOnFailed: '펫을 켤 수 없습니다.',
        unreachable: 'petdex 갤러리에 연결할 수 없습니다. 연결 상태를 확인하고 이 페이지를 다시 여세요.'
      },
      product: '제품',
      productDesc: '간결한 요약과 함께 사람이 읽기 쉬운 도구 활동을 표시합니다.',
      reactionsDesc: 'iMessage 스타일의 이모지 탭백 — 메시지에 반응하고 Hermes도 내 메시지에 반응할 수 있습니다.',
      reactionsTitle: '메시지 반응',
      reasoningCollapsedDesc: '스트리밍된 추론 내용을 직접 펼치기 전까지 접힌 상태로 유지합니다.',
      reasoningCollapsedTitle: '기본적으로 생각 과정 접기',
      removeTheme: '테마 제거',
      resumeLastSessionDesc:
        '활성화하면 앱이 콜드 스타트 시 가장 최근 채팅을 다시 엽니다. 항상 새 채팅으로 시작하려면 끄세요.',
      resumeLastSessionTitle: '실행 시 마지막 채팅 다시 열기',
      sessionDensityComfortable: '편안함',
      sessionDensityCompact: '컴팩트',
      sessionDensityDesc: '사이드바의 세션 제목 아래에 표시될 컨텍스트 양을 선택하세요.',
      sessionDensityDetailed: '상세함',
      sessionDensityTitle: '세션 목록 밀도',
      tabStripAlways: '항상',
      tabStripAuto: '자동',
      tabStripDesc:
        '영역 위에 탭을 표시합니다. 자동 모드는 다른 채팅이나 타일 영역이 열려 있지 않은 한 단일 창에서는 숨깁니다.',
      tabStripNever: '안 함',
      tabStripTitle: '탭 스트립',
      technical: '기술적',
      technicalDesc: '원시 도구 인자/결과 및 저수준 세부 정보를 포함합니다.',
      terminalFontDesc:
        '데스크톱 터미널에 사용할 설치된 폰트를 선택하세요. Nerd Font는 Powerlevel10k 및 셸 아이콘을 렌더링합니다. 비워두면 번들된 JetBrains Mono를 사용합니다.',
      terminalFontPlaceholder: 'MesloLGS NF 또는 CSS 폰트 스택',
      terminalFontPreview: '글리프 미리보기',
      terminalFontReset: '기본값 사용',
      terminalFontTitle: '터미널 폰트',
      textDirection: {
        auto: '자동',
        ltr: '왼쪽에서 오른쪽으로',
        rtl: '오른쪽에서 왼쪽으로'
      },
      textDirectionDesc:
        '채팅 메시지와 작성기의 방향을 결정하는 방식입니다. 자동은 각 단락의 첫 글자를 따르며, 혼합 텍스트가 잘못 정렬될 경우 방향을 직접 선택하세요. 코드는 항상 왼쪽에서 오른쪽으로 유지됩니다.',
      textDirectionTitle: '텍스트 방향',
      themeDesc: '데스크톱 팰릿 전용입니다. 선택한 모드가 그 위에 적용됩니다.',
      themeSearchPlaceholder: '테마 또는 VS Code Marketplace 검색…',
      themeTitle: '테마',
      tipsDesc:
        '앱과 Hermes가 제공하는 가끔의 힌트입니다. 각 팁은 한 번씩 나타납니다. 처음 30일이 지나면 자동으로 꺼지며, 다시 켤 수 있습니다.',
      tipsTitle: '앱 내 팁',
      title: '모양새',
      toolViewDesc: "'제품'은 원시 도구 페이로드를 숨기고, '기술적'은 전체 입력/출력을 표시합니다.",
      toolViewTitle: '도구 호출 표시',
      toursDesc:
        '앱 사용법을 안내할 때 Hermes가 각 단계를 비추도록 합니다. 처음 30일이 지나면 자동으로 꺼지며, 다시 켤 수 있습니다.',
      toursTitle: '가이드 투어',
      translucencyDesc:
        '텍스트를 포함하여 창 전체로 데스크톱이 비쳐 보이게 합니다. 밝은 모드와 어두운 모드를 따로 조절할 수 있습니다.',
      translucencyFadeTitle: '페이드',
      translucencyFrost: {
        header: '눈부심',
        popover: '부드럽게',
        titlebar: '밝게',
        'under-window': '깊게'
      },
      translucencyFrostTitle: '프로스트',
      translucencyGlassDesc:
        '매트 유리: 텍스트는 선명하게 유지되면서 데스크톱이 부드러운 블러로 비쳐 보입니다. 밝은 모드와 어두운 모드를 따로 조절할 수 있습니다.',
      translucencyModeClear: '투명',
      translucencyModeGlass: '유리',
      translucencyScope: {
        sidebar: '사이드바만',
        window: '전체 창'
      },
      translucencyScopeTitle: '영역',
      translucencyTintTitle: '색조',
      translucencyTitle: '창 투명도',
      uiScaleTitle: 'UI 크기 비율',
      userBubbleDesc: '내 메시지가 얼마나 투명해질지 설정합니다. 0이면 불투명하고, 100이면 테두리만 남습니다.',
      userBubbleTitle: '메시지 말풍선',
      vibeHeartsDesc:
        "감사, ily, 좋은 봇 등의 말을 하거나 하트를 보낼 때 떠오르는 하트입니다. 위의 '메시지 반응'과는 별개입니다.",
      vibeHeartsTitle: '바이브 하트',
      chatTextScaleDesc:
        'UI 배율에 상대적으로 대화 텍스트와 메시지 편집기 크기를 조절해요. 사이드바와 컨트롤은 크기가 유지돼요.',
      chatTextScaleTitle: '채팅 텍스트 크기'
    },
    billing: {
      amountValidation: {
        greaterThanThreshold: '자동 충전 금액은 임계값보다 커야 합니다.',
        reloadTo: '충전 금액'
      },
      autoReload: {
        cancel: '취소',
        disable: '비활성화',
        manage: '관리',
        reloadTo: '충전 금액',
        reloadToAria: '자동 충전 충전 금액',
        save: '저장',
        saving: '저장 중…',
        threshold: '임계값',
        thresholdAria: '자동 충전 임계값',
        turnOff: '끄기',
        turnOffConfirm: '자동 충전을 끌까요?',
        turnedOff: '자동 충전이 꺼졌습니다.',
        updated: '자동 충전이 업데이트되었습니다.'
      },
      buyCredits: {
        buyButton: '구매',
        customAmount: '사용자 지정 크레딧 금액',
        openPortal: '포털 열기',
        processing: '처리 중… 결제 확인 중',
        retry: '다시 시도',
        title: '지금 크레딧 구매'
      },
      charge: {
        authenticationRequired: '은행 인증(3DS)이 필요합니다. 결제를 완료하려면 포털에서 인증을 진행하세요.',
        checkBody: '결제 상태를 확인할 수 없습니다.',
        checkTitle: '결제를 확인할 수 없음',
        declined: '카드가 거절되었습니다. 포털에서 다른 카드를 시도해 보세요.',
        expired: '카드가 만료되었습니다. 포털에서 카드를 업데이트하세요.',
        failedTitle: '결제 실패',
        timeoutBody: '결제가 아직 처리 중일 수 있습니다. 다시 시도하기 전에 포털을 확인하세요.',
        timeoutTitle: '5분이 지나도 처리 중입니다',
        unconfirmedTitle: '결제 결과를 확인할 수 없습니다',
        untrackedBody: '결제 서비스에서 요청을 수락했으나 결제 ID를 반환하지 않았습니다.',
        untrackedTitle: '결제 내역을 추적할 수 없습니다'
      },
      errors: {
        cliBillingDisabled: {
          message:
            '이 계정의 원격 사용량이 꺼져 있습니다. 포털의 Hermes Agent 페이지에서 결제 관리자가 켤 수 있습니다.',
          title: '원격 사용량이 꺼져 있음'
        },
        consentRequired: {
          message: '포털에서 터미널 결제를 위해 이 카드를 확인해 주세요.',
          title: '카드 확인 필요'
        },
        default: {
          message: '결제 요청에 실패했습니다.',
          title: '결제 요청 실패'
        },
        endpointUnavailable: {
          message: '결제 엔드포인트가 JSON이 아닌 응답을 반환했습니다 (이 배포 환경에서는 지원되지 않을 수 있습니다).',
          title: '결제 엔드포인트를 사용할 수 없음'
        },
        idempotencyConflict: {
          message: '🔴 해당 결제 키가 이미 다른 금액으로 사용되었습니다. 새로 충전해 주세요.',
          title: '새로 충전 시작'
        },
        insufficientScope: {
          message: '원격 사용량이 허용되어야 합니다. 충전을 시작하여 허용한 후 다시 시도해 주세요.',
          title: '원격 사용량 승인 필요'
        },
        monthlyCapExceeded: {
          messageReached: '🔴 월간 사용 한도에 도달했습니다.',
          title: '월간 사용 한도 도달'
        },
        noPaymentMethod: {
          message:
            '💳 터미널 결제에 사용할 저장된 카드가 없습니다. 포털에서 카드를 등록해 주세요 (일회성 크레딧 구매 시 재사용 가능한 카드는 저장되지 않습니다).',
          title: '저장된 카드 없음'
        },
        orgAccessDenied: {
          message: '이 토큰은 관리할 수 있는 조직에 연결되어 있지 않습니다.',
          title: '조직 액세스 거부됨'
        },
        rateLimited: {
          title: '요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.'
        },
        remoteSpendingRevoked: {
          messageByAdmin: '관리자가 이 터미널의 원격 사용량을 중지했습니다.',
          messageBySelf: '이 터미널의 원격 사용량을 중지했습니다.',
          title: '원격 사용량이 중지되었습니다'
        },
        roleRequired: {
          message:
            '자금을 추가하려면 조직 관리자/소유자 권한이 필요합니다. 관리자에게 요청하거나 포털에서 관리해 주세요.',
          title: '관리자 권한 필요'
        },
        sessionRevoked: {
          message: '세션이 로그아웃되었습니다. 설정 → Gateway에서 다시 로그인해 주세요.',
          title: '세션 로그아웃됨'
        },
        stripeUnavailable: {
          title: 'Stripe에 문제가 발생했습니다'
        },
        timeout: {
          message: '결제 요청 시간 초과되었습니다.',
          title: '결제 요청 시간 초과'
        },
        transport: {
          message: '게이트웨이에 도달하기 전에 결제 요청이 실패했습니다.',
          title: '결제 연결 실패'
        },
        upgradeCapExceeded: {
          message: '일일 요금제 변경 횟수 한도에 도달했습니다. 내일 다시 시도해 주세요.',
          title: '일일 요금제 변경 한도 도달'
        }
      },
      freeTier: {
        caption:
          '커넥터가 포함된 nous/welcome에서 실행됩니다. 로그인하면 커넥터가 유지되며 계정이 필요한 도구와 모든 모델이 추가됩니다.',
        connectors: '커넥터',
        footnote:
          '무료 티어에는 잔액이 없으며 요금이 청구되지 않습니다. Nous 계정으로 로그인하면 결제 및 사용량이 표시됩니다.',
        included: '포함됨',
        message: 'Nous 계정으로 로그인하여 더 많은 모델과 도구를 잠금 해제하세요.',
        model: '모델',
        name: 'Nous · 무료 티어',
        plan: '무료 티어',
        signIn: '로그인',
        title: 'Nous 무료 티어를 사용 중입니다'
      },
      plan: {
        backAria: '결제로 돌아가기',
        cancel: '취소',
        cannotChange: '여기서는 해당 변경을 수행할 수 없습니다.',
        changePlan: '요금제 변경',
        checkingChange: '변경 사항 확인 중…',
        confirmDowngrade: '다운그레이드 확인',
        current: '현재 요금제',
        downgrade: '다운그레이드',
        empty: '현재 변경할 수 있는 요금제가 없습니다.',
        notScheduleable: '여기서는 이 변경을 예약할 수 없습니다.',
        scheduled: '예약됨',
        scheduling: '예약 중…',
        title: '요금제',
        tryAgain: '다시 시도',
        undo: '실행 취소',
        undoing: '실행 취소 중…',
        viewPlans: '요금제 보기'
      },
      preview: '미리보기',
      sections: {
        invoices: '인보이스',
        paymentAndCredits: '결제 및 크레딧',
        plan: '요금제',
        usage: '사용량'
      },
      state: {
        autoRefill: {
          distinctCardFallback: '다른 카드',
          enabledPill: '활성화됨',
          genericDescription: '잔액이 임계값 아래로 떨어지면 자동으로 충전합니다.',
          manageCaption: '포털에서 자동 충전을 관리하세요.',
          notAvailablePill: '—',
          offPill: '꺼짐',
          reconcileAction: '대사 ↗',
          title: '잔액 부족 시 자동 충전',
          turnOnCaption: '포털에서 자동 충전을 켜세요'
        },
        buyCredits: {
          description: '카드로 일회성 결제를 진행하여 오늘 잔액에 추가합니다.'
        },
        notice: {
          loggedOut: {
            action: '로그인',
            message: 'Nous 계정으로 로그인하여 잔액, 요금제, 사용량을 확인하세요.',
            title: 'Nous 계정 연결'
          },
          noCard: {
            action: '카드 추가 ↗',
            message:
              '카드가 등록될 때까지 크레딧 충전 및 자동 충전 기능을 사용할 수 없습니다. 포털에서 카드를 추가해 주세요.',
            title: '등록된 결제 수단 없음'
          },
          openPortal: '포털 열기 ↗'
        },
        paymentMethod: {
          addAction: '결제 수단 추가',
          description: '충전 및 구독 갱신에 사용되는 카드를 관리합니다.',
          provenance: {
            autoRefill: '자동 충전 카드',
            customerDefault: '고객 기본 카드',
            subPin: '구독 카드'
          },
          title: '결제 수단',
          updateAction: '업데이트'
        },
        planCard: {
          adjustPlanAction: '요금제 조정 ↗',
          chooseAction: '선택 ↗',
          freeTier: '무료',
          noSubscriptionCaption: '활성 구독이 없습니다. 유료 모델 사용 시 충전 크레딧이 차감됩니다.',
          unavailableCaption: '구독 정보를 사용할 수 없습니다. 포털을 열어서 확인할 수 있습니다.'
        },
        usage: {
          monthlyCap: {
            barLabel: '월간 사용 한도 소모량',
            captionDefault: '기본 상한',
            captionSpending: '월간 원격 사용량',
            title: '월간 사용 한도'
          },
          topupCredits: {
            caption: '만료되지 않음',
            title: '충전 크레딧'
          }
        }
      },
      stepUp: {
        deniedBody: '인증이 완료되었으나 이 터미널의 원격 사용량이 허용되지 않았습니다.',
        deniedTitle: '인증이 승인되지 않았습니다',
        dismiss: '닫기',
        openVerification: '인증 페이지 열기',
        successBody: '이 터미널의 원격 사용량이 허용되었습니다.',
        successTitle: '인증 완료',
        verify: '계속하려면 인증하세요',
        waiting: '인증 링크 대기 중…'
      },
      summary: {
        autoRefill: '자동 충전',
        balance: '잔액',
        plan: '요금제'
      },
      title: '청구',
      usage: {
        title: '사용량'
      }
    },
    computerUse: {
      accessibility: '손쉬운 사용',
      driverHealth: '드라이버 상태',
      screenRecording: '화면 녹화'
    },
    config: {
      alwaysExternalLinksDesc:
        '클릭한 모든 링크를 앱 내 브라우저 대신 시스템 브라우저로 엽니다. 마우스 우클릭 메뉴의 "앱 내 브라우저는 외부 브라우저에서 열기"는 계속 작동합니다.',
      alwaysExternalLinksTitle: '항상 외부 브라우저로 링크 열기',
      attachmentSizeDesc:
        'Desktop이 미리보기 및 이미지 첨부를 위해 로드할 로컬 파일의 최대 크기(MB)입니다. 기본값은 16입니다. 원격 비이미지 첨부는 별도의 256MB 상한을 사용합니다. 이 값을 너무 높게 설정하면 전체 파일이 메모리에 로드되어 앱이 멈추거나 충돌할 수 있습니다.',
      attachmentSizeLabel: '최대 미리보기 / 이미지 로드 크기 (메가바이트)',
      attachmentSizeTitle: '최대 미리보기 / 이미지 로드 크기',
      attachmentSizeUnit: 'MB',
      autosaveFailed: '자동 저장 실패',
      builtinOnly: '내장형 전용',
      commaSeparated: '쉼표로 구분된 값',
      disableF12Desc:
        'F12 키를 눌러 개발자 도구가 열리는 것을 차단합니다. Ctrl+Shift+I (Mac의 경우 Cmd+Opt+I)는 계속 작동합니다.',
      disableF12Title: 'F12 개발자 도구 비활성화',
      emptyDesc: '이 섹션에는 조정할 수 있는 설정이 없습니다.',
      emptyTitle: '구성할 항목 없음',
      failedLoad: '설정을 불러오지 못했습니다',
      imported: '구성을 가져왔습니다',
      invalidJson: '잘못된 구성 JSON입니다',
      keepAwakeDesc:
        '장치가 절전 모드로 전환되지 않도록 하여 야간 실행 등이 계속 유지되도록 합니다. 디스플레이는 어두워질 수 있습니다.',
      keepAwakeTitle: '컴퓨터 절전 모드 방지',
      loading: 'Hermes 구성을 불러오는 중...',
      minimizeToTrayDesc:
        "창을 최소화하거나 메인 창을 닫으면 시스템 트레이(macOS의 경우 메뉴 모음)로 숨겨지고 Hermes가 계속 실행됩니다. 트레이 메뉴의 'Hermes 종료' 또는 Cmd+Q를 사용하여 종료하세요. 기본값은 꺼져 있으며, 이 기기에만 적용됩니다.",
      minimizeToTrayTitle: '트레이로 최소화',
      minimizeToTrayUnavailable:
        '시스템 트레이를 사용할 수 없습니다. 창이 일반적인 방식으로 최소화되고 닫힙니다. 다시 시도하려면 이 옵션을 껐다 켜세요.',
      noResults: '검색 결과가 없습니다',
      none: '없음',
      noneParen: '(없음)',
      notSet: '설정되지 않음',
      searchPlaceholder: '검색…',
      showOptions: '옵션 보기',
      systemDefault: '시스템 기본값',
      toolsetsWipeConfirm:
        '활성화된 모든 툴셋을 제거하시겠습니까? 다시 활성화할 때까지 메모리, 터미널, 웹 검색, 위임 및 기타 대부분의 도구가 비활성화됩니다.',
      voiceShortcutHintDesc:
        '설정 → 키보드 단축키("음성 대화 시작 / 중지")에서 음성 녹음 단축키를 설정하세요. voice.record_key 구성 값은 CLI 및 TUI에만 적용됩니다.',
      voiceShortcutHintTitle: '음성 녹음 단축키',
      keepAwakeAlways: '항상',
      keepAwakeOff: '끔',
      keepAwakeWhileWorking: '작업 중일 때'
    },
    connections: {
      addConnection: '연결 추가',
      cancel: '취소',
      cloudAddHint:
        '팁: 위의 Hermes Cloud에 로그인하면 에이전트가 자동으로 검색됩니다. 이 양식은 알려진 인스턴스 URL을 수동으로 등록할 때만 사용하세요.',
      currentPill: '현재',
      duplicateLocal: '이 앱은 이미 로컬 연결을 관리하고 있습니다. 로컬 연결은 하나만 존재할 수 있습니다.',
      editConnection: '편집',
      empty: '등록된 연결이 없습니다.',
      headerAdd: '헤더 추가',
      headerRemove: '제거',
      headerValuePlaceholder: '값',
      headerValueSaved: '저장됨 — 유지하려면 비워두세요',
      headersDesc:
        '이 게이트웨이로 향하는 모든 HTTP 및 WebSocket 요청과 함께 전송됩니다. Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret)와 같은 액세스 프록시용입니다. 값은 암호화되어 저장됩니다. Hermes가 관리하는 헤더(Authorization, Cookie, Host 등)는 무시됩니다.',
      headersTitle: '추가 게이트웨이 헤더',
      intro: '이 기기와 원격, SSH, 클라우드 연결을 통해 접근할 수 있는 모든 Hermes 게이트웨이를 관리하세요.',
      kindCloud: 'Hermes Cloud',
      kindCloudDesc: 'Hermes Cloud 계정을 통해 검색된 호스팅 인스턴스입니다.',
      kindLocal: '로컬',
      kindLocalDesc: '이 앱이 관리하는 Hermes 런타임입니다.',
      kindRemote: '원격 게이트웨이',
      kindRemoteDesc: 'HTTP(S)를 통해 접근할 수 있는 Hermes 게이트웨이 (LAN, Tailscale 또는 인터넷).',
      kindSsh: 'SSH',
      kindSshDesc: 'SSH를 통해 접근하는 Hermes 설치입니다.',
      labelDesc:
        '필수 항목입니다. 이 인스턴스가 표시되는 모든 곳에 나타나며 고유해야 합니다 (예: “홈랩”, “업무용 노트북”).',
      labelPlaceholder: '홈랩',
      labelTitle: '이름',
      launchModeDesc: '꺼져 있으면 세션이 기본 게이트웨이에서 열립니다.',
      launchModeTitle: '시작 시 마지막으로 사용한 게이트웨이의 세션으로 돌아가기',
      loadFailed: '연결을 불러오지 못했습니다',
      localAddHint:
        '로컬을 사용할 수 없습니다. 관리되는 로컬 연결이 이미 존재합니다 (로컬 연결은 항상 하나만 존재할 수 있습니다).',
      makePrimary: '기본으로 설정',
      managedPill: '앱 관리됨',
      noSearchResults: '검색 결과와 일치하는 게이트웨이가 없습니다.',
      primaryPill: '기본',
      removeConfirmTitle: '이 연결을 제거하시겠습니까?',
      removeConnection: '제거',
      removeFailed: '연결을 제거하지 못했습니다',
      save: '연결 저장',
      saveFailed: '연결을 저장하지 못했습니다',
      saving: '저장 중…',
      searchPlaceholder: '게이트웨이 검색…',
      sshHostTitle: 'SSH 호스트',
      stagedNote:
        '세션에서 게이트웨이를 전환하세요. 프로필, 채팅, 메시징 및 크론 작업은 해당 게이트웨이에 유지되며, 다른 게이트웨이의 작업은 계속 실행됩니다.',
      testConnection: '테스트',
      testFailed: '연결 테스트 실패',
      testOk: '연결 가능',
      title: '등록된 게이트웨이',
      updateAll: '모든 인스턴스 업데이트',
      updateAllDone: '업데이트가 전송되었습니다',
      updateAllFailed: '일괄 업데이트에 실패했습니다',
      updateAllRunning: '모든 인스턴스를 업데이트하는 중…',
      updateSkippedCloud: 'Hermes Cloud에서 관리됨',
      urlTitle: 'Gateway URL'
    },
    credentials: {
      couldNotSave: '자격 증명을 저장할 수 없습니다.',
      enterValueFirst: '먼저 값을 입력하세요.',
      getKey: '키 발급받기',
      optional: '선택 사항',
      pasteKey: '키 붙여넣기',
      remove: '제거',
      saving: '저장 중'
    },
    customEndpoints: {
      activationFailed: '활성화 실패',
      active: '활성',
      addTitle: '엔드포인트 추가',
      apiKeySet: 'API 키 설정됨',
      apiMode: 'API 모드',
      autoDetect: '자동 감지',
      contextPlaceholder: '자동',
      couldNotLoad: '사용자 지정 엔드포인트를 불러올 수 없습니다',
      deleteEndpoint: '엔드포인트 삭제',
      deleteFailed: '삭제 실패',
      editTitle: '엔드포인트 편집',
      emptyDescription: '아래에 OpenAI 호환 엔드포인트를 추가하세요.',
      emptyTitle: '사용자 지정 엔드포인트 없음',
      endpointReachable: '엔드포인트에 접근할 수 있습니다.',
      endpointSaved: '사용자 지정 엔드포인트가 저장되었습니다.',
      endpointValidationFailed: '엔드포인트 유효성 검사에 실패했습니다.',
      fields: {
        apiKey: 'API 키',
        apiKeyNewPlaceholder: '현재 키를 유지하려면 비워 두세요',
        apiKeyPlaceholder: '선택 사항',
        context: '컨텍스트',
        defaultModel: '기본 모델',
        discoverModels: '모델 검색',
        endpointUrl: '엔드포인트 URL',
        name: '이름',
        providerId: '공급자 ID',
        useNewChats: '새 채팅에 사용'
      },
      namePlaceholder: 'Axet Proxy',
      newEndpoint: '새 엔드포인트',
      save: '저장',
      saveFailed: '저장 실패',
      test: '테스트',
      title: '사용자 지정 엔드포인트',
      use: '사용',
      validationFailed: '유효성 검사 실패'
    },
    envActions: {
      actions: '작업',
      clear: '지우기',
      docs: '문서',
      hideValue: '값 숨기기',
      manageInKeys: 'API 키에서 관리',
      replace: '교체',
      revealValue: '값 표시',
      set: '설정'
    },
    fieldDescriptions: {
      'agent.imageInputMode': '모델에 이미지 첨부 파일이 전송되는 방식을 제어합니다.',
      'agent.maxTurns': 'Hermes가 실행을 중단하기 전의 도구 호출 턴 상한선입니다.',
      'approvals.mode': '명시적 승인이 필요한 명령을 Hermes가 처리하는 방식입니다.',
      'approvals.timeout': '승인 프롬프트가 시간 초과될 때까지 대기하는 시간입니다.',
      'auxiliary.compression.timeout':
        '호출당 보조 압축 모델을 대기하는 시간(초, 기본값 120). 느린 로컬 모델의 경우 값을 높이세요.',
      'browser.useRealProfile':
        '로컬 브라우징은 실제 로그인 정보를 사용합니다. Hermes는 기본 브라우저의 프로필(쿠키, 로그인, 환경 설정)을 관리되는 스냅샷으로 복사하여 패키지된 Chromium으로 구동합니다. 실제 라이브 프로필은 직접 열리지 않으며, 실행할 때마다 복사본이 갱신됩니다. 또한 클라우드 브라우저 백엔드가 구성된 경우에도 요청 시 에이전트가 로컬 실제 프로필 세션을 열 수 있도록 합니다. Chromium 기반 브라우저(Chrome, Edge, Brave, Brave Origin, Chromium)만 지원되며, 비 Chromium 기본 브라우저는 명확한 메시지와 함께 실패합니다. 기본값은 꺼져 있습니다.',
      'checkpoints.enabled': '파일을 편집하기 전에 롤백 스냅샷을 생성합니다.',
      'codeExecution.mode': '코드 실행이 현재 프로젝트로 한정되는 엄격 정도입니다.',
      'compression.codexGpt55Autoraise': '지원되는 ChatGPT Codex OAuth 모델의 경우 압축률을 85%로 높입니다.',
      'compression.enabled': '대화가 길어지면 오래된 컨텍스트를 요약합니다.',
      'context.engine': '컨텍스트 제한에 가까워진 긴 대화를 관리하는 전략입니다.',
      'desktop.repoScanEnabled': '프로젝트에 표시할 Git 리포지토리를 로컬 폴더에서 검색합니다.',
      'desktop.repoScanExcludePaths': '리포지토리 검색 중에 건너뛸 폴더 및 그 하위 항목입니다.',
      'desktop.repoScanRoots': '검색할 폴더입니다. 홈 디렉터리를 검색하려면 비워 두세요.',
      'display.personality': '새 세션에 적용할 기본 어시스턴트 스타일입니다.',
      'display.showReasoning': '백엔드에서 추론 섹션을 제공하는 경우 표시합니다.',
      fallbackProviders: '기본 모델 실패 시 시도할 백업 공급자:모델 항목입니다.',
      fileReadMaxChars: 'Hermes가 단일 파일 요청에서 읽을 수 있는 최대 문자 수입니다.',
      'memory.memoryEnabled': '향후 세션에 도움이 될 수 있는 영구 메모리를 저장합니다.',
      'memory.userProfileEnabled': '사용자 선호도에 대한 간략한 프로필을 유지합니다.',
      model: '작성기에서 다른 모델을 선택하지 않는 한 새 채팅에 사용됩니다.',
      modelContextLength:
        '메인 채팅 모델의 감지된 컨텍스트 창만 재정의합니다(토큰). 선택한 모델의 감지된 값을 사용하려면 0으로 두세요. 보조/MoA 모델에는 영향을 미치지 않습니다.',
      'security.redactSecrets': '가능한 경우 모델에 표시되는 콘텐츠에서 감지된 비밀 정보를 숨깁니다.',
      'stt.echoTranscripts': '음성 메시지의 원본 🎙️ 텍스트 변환 내용을 채팅에 다시 게시합니다.',
      'stt.streaming': '말하는 동안 텍스트를 표시합니다(OpenAI, xAI, ElevenLabs). 실패하면 녹음으로 대체됩니다.',
      'stt.elevenlabs.languageCode':
        '선택 사항인 ISO-639-3 언어 코드입니다. 비워 두면 ElevenLabs가 자동으로 감지합니다.',
      'stt.enabled': '로컬 또는 공급자 기반 음성 텍스트 변환을 활성화합니다.',
      'terminal.cwd': '도구 및 터미널 작업용 기본 프로젝트 폴더입니다.',
      'terminal.daytonaImage': '실행 백엔드가 Daytona일 때 사용되는 이미지입니다.',
      'terminal.dockerImage': '실행 백엔드가 Docker일 때 사용되는 컨테이너 이미지입니다.',
      'terminal.envPassthrough': '도구 실행에 전달할 환경 변수입니다.',
      'terminal.modalImage': '실행 백엔드가 Modal일 때 사용되는 이미지입니다.',
      'terminal.persistentShell': '백엔드가 지원하는 경우 명령 간에 셸 상태를 유지합니다.',
      'terminal.singularityImage': '실행 백엔드가 Singularity일 때 사용되는 이미지입니다.',
      timezone: 'IANA 시간대 식별자입니다. 비워 두면 시스템 시간대를 사용합니다.',
      'tts.neutts.device': 'NeuTTS용 로컬 추론 장치입니다.',
      'tts.xai.autoSpeechTags':
        '합성 전에 LLM이 스크립트에 표현력 있는 오디오 태그([laughing], [sighs])를 삽입하도록 합니다.',
      'tts.xai.bitRate': 'bps 단위의 MP3 비트레이트입니다. 코덱이 mp3인 경우에만 적용됩니다.',
      'tts.xai.language': '음성 언어 코드(예: en, pt-BR) 또는 자동 감지를 위한 "auto"입니다.',
      'tts.xai.optimizeStreamingLatency': '지연 시간 대 품질 트레이드오프. 0 = 최고 품질, 2 = 최저 지연 시간.',
      'tts.xai.sampleRate': 'Hz 단위의 오디오 샘플 레이트입니다. 높을수록 품질이 좋고 파일 크기가 커집니다.',
      'tts.xai.speed': '재생 속도. 0.7 = 느리게, 1.0 = 보통, 1.5 = 빠르게.',
      'tts.xai.voiceId': 'xAI 음성 ID(예: eve) 또는 사용자 지정 음성 ID입니다.',
      'updates.nonInteractiveLocalChanges':
        '앱에서 Hermes를 자체 업데이트할 때(터미널 프롬프트 없음), 로컬 소스 편집 내용을 유지(stash)하거나 버립니다(discard). 터미널 업데이트 시에는 항상 확인합니다.',
      'voice.autoTts': '어시스턴트 응답을 자동으로 음성으로 읽어줍니다.',
      'voice.gptLive.instructions':
        '라이브 음성 페르소나를 위한 추가 문장(톤, 속도, 언어). Hermes는 자체 시스템 프롬프트를 유지합니다.',
      'voice.gptLive.voice': 'GPT-Live 모드용 음성입니다. 사용자 지정 음성 ID를 사용할 수 있습니다.',
      'voice.voiceChatMode':
        'chained: 아래 공급자를 사용하는 음성 텍스트 변환 → Hermes → 텍스트 음성 변환. gpt-live: 하나의 전이중 OpenAI 음성 모델(gpt-live-1)이 듣고 말하며, 모든 실제 요청을 Hermes에 전달합니다. 선택한 모든 모델이 전체 도구 세트로 응답합니다. OpenAI API 키가 필요하며, 음성 레이어는 분당 $0.05가 청구됩니다.'
    },
    fieldLabels: {
      'agent.apiMaxRetries': 'API 재시도 횟수',
      'agent.imageInputMode': '이미지 첨부 파일',
      'agent.maxTurns': '최대 에이전트 단계',
      'agent.serviceTier': '서비스 계층',
      'agent.toolUseEnforcement': '도구 사용 강제',
      'approvals.mcpReloadConfirm': 'MCP 새로고침 확인',
      'approvals.mode': '승인 모드',
      'approvals.timeout': '승인 시간 초과',
      'auxiliary.compression.timeout': '압축 모델 시간 초과(초)',
      'browser.allowPrivateUrls': '브라우저 비공개 URL',
      'browser.autoLocalForPrivateUrls': '비공개 URL용 로컬 브라우저',
      'browser.useRealProfile': '내 실제 브라우저 프로필 사용',
      'checkpoints.enabled': '파일 체크포인트',
      'checkpoints.maxSnapshots': '체크포인트 제한',
      'codeExecution.mode': '코드 실행 모드',
      commandAllowlist: '명령 허용 목록',
      'compression.codexGpt55Autoraise': 'Codex 압축 자동 상향',
      'compression.enabled': '자동 압축',
      'compression.protectLastN': '보호된 최근 메시지',
      'compression.targetRatio': '압축 목표',
      'compression.threshold': '압축 임계값',
      'context.engine': '컨텍스트 엔진',
      'delegation.childTimeoutSeconds': '하위 에이전트 시간 초과',
      'delegation.maxConcurrentChildren': '병렬 하위 에이전트',
      'delegation.maxIterations': '하위 에이전트 턴 제한',
      'delegation.model': '하위 에이전트 모델',
      'delegation.provider': '하위 에이전트 공급자',
      'delegation.reasoningEffort': '하위 에이전트 추론 수준',
      'desktop.repoScanEnabled': '자동 리포지토리 검색',
      'desktop.repoScanExcludePaths': '제외된 리포지토리 경로',
      'desktop.repoScanRoots': '리포지토리 검색 루트',
      'display.personality': '페르소나',
      'display.showReasoning': '추론 블록',
      fallbackProviders: '대체 모델',
      fileReadMaxChars: '파일 읽기 제한',
      'memory.memoryCharLimit': '메모리 예산',
      'memory.memoryEnabled': '영구 메모리',
      'memory.provider': '메모리 공급자',
      'memory.userCharLimit': '프로필 예산',
      'memory.userProfileEnabled': '사용자 프로필',
      model: '기본 모델',
      modelContextLength: '메인 모델 컨텍스트 창(재정의)',
      'security.allowPrivateUrls': '비공개 URL 허용',
      'security.redactSecrets': '비밀 정보 가리기',
      'stt.echoTranscripts': '텍스트 변환 에코',
      'stt.elevenlabs.diarize': '화자 분리',
      'stt.elevenlabs.languageCode': 'ElevenLabs 언어',
      'stt.elevenlabs.modelId': 'ElevenLabs STT 모델',
      'stt.elevenlabs.tagAudioEvents': '오디오 이벤트 태그 지정',
      'stt.enabled': '음성 텍스트 변환',
      'stt.groq.model': 'Groq STT 모델',
      'stt.local.language': '전사 언어',
      'stt.local.model': '로컬 전사 모델',
      'stt.mistral.model': 'Mistral STT 모델',
      'stt.xai.model': 'xAI STT 모델',
      'stt.deepinfra.model': 'DeepInfra STT 모델',
      'stt.openai.model': 'OpenAI STT 모델',
      'stt.openai.streamingModel': 'OpenAI 실시간 전사 모델',
      'stt.provider': '음성 텍스트 변환 공급자',
      'stt.streaming': '실시간 전사',
      'terminal.backend': '실행 백엔드',
      'terminal.cwd': '작업 디렉터리',
      'terminal.daytonaImage': 'Daytona 이미지',
      'terminal.dockerImage': 'Docker 이미지',
      'terminal.envPassthrough': '환경 변수 전달',
      'terminal.modalImage': 'Modal 이미지',
      'terminal.persistentShell': '영구 셸',
      'terminal.singularityImage': 'Singularity 이미지',
      'terminal.timeout': '명령 시간 초과',
      timezone: '시간대',
      'toolOutput.maxBytes': '터미널 출력 제한',
      'toolOutput.maxLineLength': '줄 길이 제한',
      'toolOutput.maxLines': '파일 페이지 제한',
      toolsets: '활성화된 도구 세트',
      'tts.deepinfra.model': 'DeepInfra TTS 모델',
      'tts.deepinfra.voice': 'DeepInfra 음성',
      'tts.edge.voice': 'Edge 음성',
      'tts.elevenlabs.modelId': 'ElevenLabs 모델',
      'tts.elevenlabs.voiceId': 'ElevenLabs 음성',
      'tts.gemini.model': 'Gemini TTS 모델',
      'tts.gemini.voice': 'Gemini 음성',
      'tts.kittentts.model': 'KittenTTS 모델',
      'tts.kittentts.voice': 'KittenTTS 음성',
      'tts.minimax.model': 'MiniMax TTS 모델',
      'tts.minimax.voiceId': 'MiniMax 음성',
      'tts.mistral.model': 'Mistral TTS 모델',
      'tts.mistral.voiceId': 'Mistral 음성',
      'tts.neutts.device': 'NeuTTS 장치',
      'tts.neutts.model': 'NeuTTS 모델',
      'tts.openai.model': 'OpenAI TTS 모델',
      'tts.openai.voice': 'OpenAI 음성',
      'tts.piper.voice': 'Piper 음성',
      'tts.provider': '텍스트 음성 변환 공급자',
      'tts.xai.autoSpeechTags': 'xAI 자동 음성 태그',
      'tts.xai.bitRate': 'xAI 비트레이트',
      'tts.xai.language': 'xAI 언어',
      'tts.xai.optimizeStreamingLatency': 'xAI 스트리밍 지연 시간 최적화',
      'tts.xai.sampleRate': 'xAI 샘플 레이트',
      'tts.xai.speed': 'xAI 재생 속도',
      'tts.xai.voiceId': 'xAI (Grok) 음성',
      'updates.nonInteractiveLocalChanges': '앱 내 업데이트 로컬 변경 사항',
      'voice.autoTts': '응답 소리 내어 읽기',
      'voice.gptLive.instructions': 'GPT-Live 페르소나',
      'voice.gptLive.voice': 'GPT-Live 음성',
      'voice.maxRecordingSeconds': '최대 녹음 길이',
      'voice.voiceChatMode': '음성 채팅 모드'
    },
    gateway: {
      applyFailed: '게이트웨이 설정을 적용할 수 없습니다',
      authNeedsPassword: '이 게이트웨이는 사용자 이름과 비밀번호를 사용합니다. 데스크톱 앱을 인증하려면 로그인하세요.',
      authSignedInOauth: '이 게이트웨이는 OAuth를 사용합니다. 로그인되어 있으며 세션이 자동으로 새로 고쳐집니다.',
      authSignedInPassword:
        '이 게이트웨이는 사용자 이름과 비밀번호를 사용합니다. 로그인되어 있으며 세션이 자동으로 새로 고쳐집니다.',
      authTitle: '인증',
      cloudActive: '현재 창에서 활성 상태',
      cloudAgentProvisioning: '프로비저닝 중…',
      cloudAgentsTitle: '내 에이전트',
      cloudConnect: '연결',
      cloudConnectFailed: '해당 에이전트에 연결할 수 없습니다',
      cloudConnectedPill: '연결됨',
      cloudConnectedTitle: '연결됨',
      cloudConnecting: '연결 중…',
      cloudDesc:
        'Hermes Cloud에 한 번만 로그인하면 URL을 붙여넣을 필요 없이 내 계정의 에이전트 중에서 선택할 수 있습니다.',
      cloudDiscoverFailed: 'Hermes Cloud 에이전트를 불러올 수 없습니다',
      cloudLoadingAgents: '에이전트를 불러오는 중…',
      cloudNeedsSignIn: '계정의 에이전트를 찾으려면 Hermes Cloud에 로그인하세요.',
      cloudNoAgents: {
        after: ' 후 새로 고침하세요.',
        before: '이 계정에서 에이전트를 찾지 못했습니다. 다음에서 생성하세요: ',
        linkText: 'Nous 포털'
      },
      cloudOrgChange: '조직 변경',
      cloudOrgPickerTitle: '조직 선택',
      cloudOrgSelect: '선택',
      cloudRefresh: '새로 고침',
      cloudSavedDesc:
        '기본 설정을 변경하지 않고 저장된 Gateway를 사용합니다. 인스턴스를 추가하려면 아래에서 로그인하세요. 저장된 연결 목록에서 이름과 로그인을 관리할 수 있습니다.',
      cloudSavedTitle: '저장된 Cloud 게이트웨이',
      cloudSignIn: 'Hermes Cloud 로그인',
      cloudSignInFailed: 'Hermes Cloud 로그인 실패',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignedIn: 'Hermes Cloud 로그인됨',
      cloudSignedInDesc: '로그인되었습니다. 아래에서 에이전트를 선택하세요. 세션이 자동으로 새로 고쳐집니다.',
      cloudSignedOutMessage: 'Hermes Cloud 세션이 해제되었습니다.',
      cloudSignedOutTitle: 'Hermes Cloud 로그아웃됨',
      cloudTitle: 'Hermes Cloud',
      cloudUseSaved: '게이트웨이 사용',
      diagnostics: '진단',
      diagnosticsDesc: '파일 관리자에서 desktop.log를 표시합니다. 게이트웨이가 시작되지 않을 때 유용합니다.',
      enterUrlFirst: '먼저 원격 URL을 입력하세요.',
      envOverride: '환경 변수 재정의',
      envOverrideDesc:
        '앱 외부의 시작 설정으로 인해 이 연결이 선택되어 아래 옵션이 읽기 전용으로 설정되었습니다. 여기에서 변경하려면 해당 설정 없이 Hermes를 다시 시작하거나 설정을 구성한 관리자에게 문의하세요.',
      envOverrideTitle: '이 연결은 Hermes 실행 방식에 의해 고정되었습니다.',
      failedLoad: '게이트웨이 설정을 불러오지 못했습니다',
      incompleteSignIn: '원격으로 전환하기 전에 원격 URL을 입력하고 로그인하세요.',
      incompleteSignInTest: '테스트하기 전에 원격 URL을 입력하고 로그인하세요.',
      incompleteTitle: '원격 게이트웨이 설정 미완료',
      incompleteToken: '원격으로 전환하기 전에 원격 URL과 세션 토큰을 입력하세요.',
      incompleteTokenTest: '테스트하기 전에 원격 URL과 세션 토큰을 입력하세요.',
      intro:
        '기본적으로 로컬에서 실행됩니다. 이 앱이 다른 곳의 Hermes 백엔드를 제어해야 할 때 원격을 사용하세요. 게이트웨이 연결은 머신 단위로 적용되며, 프로필은 연결한 게이트웨이에서 감지됩니다.',
      keychainEncryptionDesc:
        '기본적으로 꺼져 있습니다. 켜면 게이트웨이 토큰과 로그인 자격 증명이 시스템 키체인(Keychain Access, GNOME Keyring 또는 Windows DPAPI)으로 암호화됩니다. 시스템에서 권한이나 비밀번호를 요구할 수 있습니다. 꺼져 있으면 사용자 계정에서만 읽을 수 있는 일반 파일로 저장됩니다.',
      keychainEncryptionFailed: '비밀 암호화를 변경할 수 없습니다',
      keychainEncryptionTitle: 'OS 키체인으로 저장된 비밀 정보 암호화',
      loading: '게이트웨이 설정을 불러오는 중...',
      localDesc: 'localhost에서 개인 Hermes 백엔드를 시작합니다. 기본 설정이며 오프라인에서도 작동합니다.',
      localTitle: '로컬 게이트웨이',
      modeTitle: '연결 모드',
      openLogs: '로그 열기',
      pasteSessionToken: '세션 토큰 붙여넣기',
      plainTextConfirmAction: '일반 텍스트로 저장',
      plainTextConfirmDesc:
        '이 머신에서 OS 키체인 서비스를 찾을 수 없어, 토큰이 앱의 연결 설정 파일에 암호화되지 않은 상태로 저장되며 이 사용자로 실행되는 모든 프로세스에서 읽을 수 있게 됩니다. 암호화된 저장을 위해 시스템 키체인(Linux의 경우 GNOME Keyring 또는 KWallet)을 설치하거나 활성화하세요.',
      plainTextConfirmTitle: '게이트웨이 토큰을 일반 텍스트로 저장하시겠습니까?',
      plainTextStoredDesc:
        '보안 스토리지를 사용할 수 없으므로 저장된 토큰이 이 머신의 앱 연결 설정 파일에 암호화되지 않은 채로 저장됩니다. 암호화하려면 시스템 키체인(Linux의 경우 GNOME Keyring 또는 KWallet)을 설치하거나 활성화하세요.',
      plainTextStoredTitle: '토큰이 일반 텍스트로 저장됨',
      probeError:
        'Hermes가 해당 주소에 연결할 수 없습니다. URL을 확인하고 다른 컴퓨터에서 Hermes가 실행 중인지 확인하세요. 응답이 오면 로그인 옵션이 표시됩니다.',
      probing: '이 게이트웨이의 인증 방식을 확인하는 중…',
      reachableTitle: '원격 게이트웨이 연결 가능',
      remoteAuthHint:
        '호스팅된 게이트웨이는 OAuth나 사용자 이름 및 비밀번호를 사용하며, 자체 호스팅된 게이트웨이는 세션 토큰을 사용할 수 있습니다.',
      remoteDesc: '이 데스크톱 셸을 원격 Hermes 백엔드에 연결합니다.',
      remoteTitle: '원격 게이트웨이',
      remoteUrlDesc: '원격 대시보드 백엔드의 기본 URL입니다. /hermes와 같은 경로 접두사가 지원됩니다.',
      remoteUrlTitle: '원격 URL',
      restartingMessage: 'Hermes Desktop이 저장된 설정을 사용하여 다시 연결됩니다. 셸은 열려 있습니다.',
      restartingTitle: '게이트웨이 연결 재시작 중',
      saveAndReconnect: '저장 및 재연결',
      saveFailed: '게이트웨이 설정을 저장할 수 없습니다',
      saveForRestart: '다음 재시작 시 적용하도록 저장',
      savedMessage: '다음 재시작 시 적용되도록 저장되었습니다.',
      savedTitle: '게이트웨이 설정 저장됨',
      savedToken: '저장됨',
      signIn: '로그인',
      signInFailed: '로그인 실패',
      signOut: '로그아웃',
      signOutFailed: '로그아웃 실패',
      signedIn: '로그인됨',
      signedOutMessage: '원격 게이트웨이 세션이 해제되었습니다.',
      signedOutTitle: '로그아웃됨',
      sshButtonsHint: '저장하면 다음 실행 시 적용됩니다. 연결은 지금 다시 시도합니다.',
      sshConnect: '연결',
      sshDesc:
        'SSH를 통해 원격에서 Hermes를 실행하고 이 앱으로 터널링합니다. 직접 시작하거나 노출할 필요가 없습니다. 호스트에 대한 키 기반 SSH 접근 권한이 필요합니다.',
      sshErrAuth:
        'SSH 인증에 실패했습니다. ssh-agent에 키를 로드(ssh-add)하거나 ~/.ssh/config에 IdentityFile을 설정하세요. Hermes는 비대화형으로 ssh를 실행합니다.',
      sshErrHostKey:
        '마지막 연결 이후 호스트 키가 변경되었습니다. 의도된 변경인지 확인한 후 ssh-keygen -R <host>를 실행하고 다시 연결하세요.',
      sshErrNotInstalled:
        '원격 호스트에 Hermes가 설치되어 있지 않습니다. 원격 호스트에 설치하거나(curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) Hermes 경로를 설정하세요.',
      sshErrPlatform:
        '지원되지 않는 원격 플랫폼입니다. Hermes Desktop SSH 모드는 Linux, macOS 및 Windows 원격 호스트를 지원합니다.',
      sshErrTimeout: 'SSH 연결 시간이 초과되었습니다. 호스트에 도달할 수 없거나 절전 모드일 수 있습니다.',
      sshErrUnknown: 'SSH 연결에 실패했습니다.',
      sshErrUnreachable: 'SSH를 통해 해당 호스트에 도달할 수 없습니다. 호스트, 포트 및 네트워크를 확인하세요.',
      sshErrUpdateRequired: 'Desktop SSH로 연결하기 전에 원격 호스트의 Hermes를 업데이트하세요.',
      sshErrInteractiveAuth:
        'Tailscale SSH는 대화형 브라우저 확인이 필요합니다. 터미널에서 `ssh <host> true`를 실행해 확인을 완료한 다음 다시 시도하세요 — Hermes는 SSH를 비대화형으로 실행합니다.',
      sshHermesPathDesc: '원격 hermes 바이너리의 전체 경로입니다. 비워두면 자동 감지됩니다.',
      sshHermesPathPlaceholder: '자동 감지',
      sshHermesPathTitle: 'Hermes 경로 (선택 사항)',
      sshHostCustom: '사용자 지정 (직접 입력)…',
      sshHostDesc: 'user@host 또는 ~/.ssh/config의 Host 별칭입니다.',
      sshHostPick: '호스트 선택…',
      sshHostPickDesc: '~/.ssh/config의 Host 별칭이거나 직접 입력하려면 사용자 지정을 선택하세요.',
      sshHostPickTitle: '호스트',
      sshHostTitle: '호스트',
      sshIncompleteHost: '연결하기 전에 SSH 호스트를 입력하세요.',
      sshKeyDesc: '개인키 경로입니다. 비워두면 ssh-agent 또는 ~/.ssh/config가 사용됩니다.',
      sshKeyTitle: 'ID 파일',
      sshPortDesc: '비워두면 22번 또는 ~/.ssh/config의 포트가 사용됩니다.',
      sshPortTitle: '포트',
      sshTestConnection: 'SSH 테스트',
      sshTitle: 'SSH를 통해 연결',
      sshTrustHint: '처음 제시된 호스트 키는 신뢰되며 고정됩니다. 이후 변경되면 연결이 차단됩니다.',
      sshUserDesc: '비워두면 ~/.ssh/config 또는 현재 사용자가 사용됩니다.',
      sshUserPlaceholder: '~/.ssh/config에서 가져옴',
      sshUserTitle: '사용자',
      testFailed: '원격 게이트웨이 테스트 실패',
      testRemote: '원격 테스트',
      title: '게이트웨이 연결',
      tokenDesc: 'REST 및 WebSocket 액세스에 사용되는 대시보드 세션 토큰입니다. 저장된 토큰을 유지하려면 비워두세요.',
      tokenTitle: '세션 토큰',
      unavailableDesc: '연결 설정은 해당 컴퓨터에서 실행 중인 Hermes Desktop 앱에서만 변경할 수 있습니다.',
      unavailableTitle: '게이트웨이 설정을 사용할 수 없음'
    },
    hudModifier: {
      description:
        'Mac에서는 ⌘ + Option을, Windows/Linux에서는 Ctrl + Alt를 짧게 누르면 어떤 앱에서든 HUD를 앞으로 불러올 수 있습니다. 기본적으로 꺼져 있으며 이 기기에만 적용됩니다.',
      missingHelper:
        '이 Hermes 설치에 HUD 제스처 도우미가 누락되었습니다. Hermes를 업데이트하거나 재설치한 후 다시 시도하세요.',
      permission:
        '시스템 설정 → 개인정보 보호 및 보안 → 입력 모니터링에서 Hermes를 허용하고 다시 시도하세요. 이 제스처는 키 입력이나 화면을 기록하지 않습니다.',
      title: '탭하여 HUD 호출',
      unavailable:
        'HUD 제스처 도우미를 시작할 수 없거나 예기치 않게 중지되었습니다. 다시 시도하거나 Hermes를 다시 시작하세요. 기존 HUD 단축키는 Hermes 내에서 계속 작동합니다.',
      unsupportedSession:
        '이 데스크톱 세션은 전역 수정자 키 탭을 지원하지 않습니다. Linux는 X11이 필요하며 Wayland는 지원되지 않습니다.'
    },
    keys: {
      empty: '이 카테고리에 구성된 항목이 없습니다.',
      failedLoad: 'API 키를 불러오지 못했습니다',
      loading: 'API 키 및 자격 증명을 불러오는 중...'
    },
    localModels: {
      activating: '시작 중…',
      activeDetail: '새 채팅에서 이 모델을 사용합니다. 첫 번째 메시지를 보내면 로드됩니다.',
      activeNotLoaded: '첫 번째 메시지 전송 시 로드됨',
      activePill: '기본값',
      addedByYou: '내가 추가함',
      browseAlreadyDownloaded: '이미 다운로드되었습니다.',
      browseDownloadAria: '{name} 다운로드',
      browseDownloadStarted: '{name} 다운로드 중',
      browseDownloads: '다운로드',
      browseFitUnknown: '적합성 알 수 없음',
      browseGated: 'Hugging Face 로그인 필요',
      browseHint:
        'Hugging Face 전체에서 검색하세요. 여기서 다운로드한 모델은 내 기기 사양에 맞게 크기가 자동으로 조절되지만, 당사에서 테스트한 것은 아닙니다.',
      browseLikes: '좋아요',
      browseListing: '모델 파일 읽는 중',
      browseNoGguf: '호환되는 모델 파일을 찾을 수 없습니다.',
      browsePlaceholder: '이름이나 작성자로 모델 검색…',
      browseRefresh: '새로 고침',
      browseSearching: 'Hugging Face 검색 중',
      browseShowFiles: '파일 표시',
      browseTitle: '더 많은 모델 찾기',
      connectionChanged: '로컬 모델 연결이 변경되었습니다',
      deleteAction: '모델 삭제',
      deleteFailed: '삭제 실패',
      downloadPauseAction: '일시 정지',
      downloadPausedLabel: '일시 정지됨',
      downloadResumeAction: '재개',
      downloadStatusRunning: '다운로드 중',
      downloaded: '다운로드됨',
      ejectFailed: '모델을 언로드할 수 없습니다',
      ejectTip: 'GPU 메모리 해제 (다음 메시지 전송 시 다시 로드됨)',
      ejected: '모델이 언로드되었습니다 — GPU 메모리가 해제되었습니다.',
      hardwareLoading: '하드웨어 확인 중…',
      hardwareTitle: '이 기기',
      installAction: '런타임 설치',
      installDetail:
        'llama.cpp 추론 엔진을 다운로드합니다(수백 MB). 다운로드한 모델은 이 기기에서만 완전히 실행되며, 계정이 필요 없고 외부로 데이터가 전송되지 않습니다.',
      installDoneToast: '로컬 런타임이 설치되어 사용할 준비가 되었습니다.',
      installFailed: '런타임 설치 실패',
      installTitle: '로컬 런타임 설치',
      installing: '런타임 설치 중…',
      loadedPill: '메모리에 로드됨',
      loadingPill: '로드 중…',
      modelsTitle: '모델',
      noRecommendationAction: '모델 찾아보기',
      noRecommendationDetail:
        '자동 설정에는 GPU 또는 통합 메모리에 완전히 들어맞는 선별된 모델이 필요합니다. 아래에서 직접 모델을 선택하거나 더 많은 모델을 찾아볼 수 있습니다.',
      noRecommendationTitle: '이 기기에 대한 자동 추천 모델이 없습니다',
      pillFitsGpu: 'GPU에 맞음',
      pillFullContextTip: '처음부터 모델의 전체 컨텍스트 창으로 실행됩니다.',
      pillGrowsTip: '대화에 더 많은 공간이 필요할 때 자동으로 늘어납니다.',
      pillTooBig: '이 기기에 비해 너무 큼',
      pillUsesRam: '시스템 RAM 사용',
      pillVision: '이미지 인식',
      placementResident: '전체 GPU 상주',
      placementResidentTip: '현재 컨텍스트 창에서 GPU 메모리에 완전히 상주하여 실행됩니다 — 최고 속도.',
      placementSpilled: '일부 RAM 사용',
      placementSpilledTip:
        '모델의 일부가 시스템 RAM에서 실행되어 작동은 하지만 속도가 느려집니다. 더 압축된 빌드나 작은 컨텍스트를 사용하면 완전히 들어맞을 수 있습니다.',
      quickstartAction: '자동 설정',
      quickstartConfigure: '직접 선택',
      quickstartFailed: '로컬 모델 설정 실패',
      quickstartStageEngine: '엔진',
      quickstartStageFinish: '완료',
      quickstartStageModel: '모델',
      quickstartTitle: '이 기기에서 모델 실행',
      recommended: '추천',
      recommendedReason: {
        'best-quality-resident':
          'GPU에서 최고 속도로 완전히 실행되는 가장 품질이 높은 모델입니다. 이 하드웨어에서의 품질과 예상 속도를 비교하여 선택됩니다.',
        'fastest-resident':
          '이 하드웨어에서는 최고 속도에 도달하는 모델이 없으며, 이 모델이 GPU 메모리에 완전히 상주하면서 그에 가장 가깝게 작동합니다.',
        'speed-gated-quality':
          '이 기기에 더 고성능인 모델이 맞지만 메모리 대역폭 때문에 응답이 너무 느려질 수 있어요. 속도를 유지하는 가장 좋은 모델이에요.',
        'product-default': '제조사가 이 기기에 지정한 기본 모델입니다.'
      },
      runtimeInstalled: 'llama.cpp 런타임이 설치되었습니다',
      runtimeRunningDetail:
        '로컬 서버가 실행 중입니다. 끄면 모든 GPU 메모리가 해제되며, 다시 켤 때까지 새 채팅에서 로컬 모델을 사용하지 않습니다.',
      runtimeTitle: '로컬 런타임',
      serverRunning: '실행 중',
      serverStartFailed: '로컬 서버를 시작할 수 없습니다',
      serverStarted: '로컬 서버가 실행 중입니다.',
      serverStopFailed: '로컬 서버를 중지할 수 없습니다',
      serverStopped: '로컬 서버가 중지되었습니다 — GPU 메모리가 해제되었습니다.',
      sideloadAlreadyPresent: '이미 라이브러리에 있습니다.',
      sideloadButton: '모델 파일 추가',
      sideloadDone: '{name}을(를) 추가했습니다.',
      sideloadTitle: 'GGUF 모델 파일 선택',
      startServer: '켜기',
      stopServer: '끄기',
      title: '로컬 모델',
      unifiedMemory: '통합 메모리',
      upToDateTitle: '엔진이 최신 버전입니다',
      updateAction: '엔진 업데이트',
      updateTitle: '엔진 업데이트 사용 가능',
      updating: '엔진 업데이트 중…',
      useAction: '사용'
    },
    managedUpdates: {
      alreadyRunning: '업데이트가 이미 진행 중입니다',
      failed: '업데이트 실패',
      intro:
        '데스크톱 관리형 SSH 설치를 트랜잭션 방식으로 업데이트합니다. 세션을 비우고, 원격 체크아웃을 업데이트하며, 모든 프로필이 연관된 영수증과 함께 복원됩니다.',
      partial: '업데이트됨 — 복원 실패',
      progress: '세션을 비우고, 원격 설치를 업데이트하고, 프로필을 복원하는 중…',
      refused: '거부됨',
      sshConnection: '데스크톱 관리형 SSH 설치',
      title: '관리형 업데이트',
      update: '업데이트',
      updated: '업데이트됨',
      updating: '업데이트 중…'
    },
    mcp: {
      allServers: '모든 서버',
      authenticate: '인증',
      authenticatedTitle: '인증됨',
      catalogEnvRequired: '설치하기 전에 필수 값을 입력하세요.',
      catalogLoading: 'MCP 카탈로그를 불러오는 중...',
      deepLinkConfirm: '서버 추가',
      deepLinkDescription:
        '링크를 통해 이 MCP 서버를 Hermes에 추가하려고 합니다. 아래의 정확한 구성을 검토하세요. 이는 Hermes가 아니라 링크에서 제공된 것입니다.',
      deepLinkErrorConfig: '링크의 구성이 유효한 base64 인코딩 JSON이 아닙니다.',
      deepLinkErrorName: '링크의 서버 이름이 누락되었거나 유효하지 않습니다.',
      deepLinkErrorShape: '구성은 문자열 `url` 또는 `command` 필드가 있는 JSON 객체여야 합니다.',
      deepLinkErrorTitle: 'MCP 설치 링크 거부됨',
      deepLinkErrorTooLarge: '구성 페이로드가 32KB 제한을 초과했습니다.',
      deepLinkErrorUrl: 'http:// 및 https:// 서버 URL만 허용됩니다.',
      deepLinkNameInvalid: '이름에는 1~64자의 영문자, 숫자, 점, 대시 또는 밑줄을 사용할 수 있습니다.',
      deepLinkStdioWarning:
        '이 서버는 아래에 표시된 명령어로 사용자의 기기에서 로컬 프로세스를 실행합니다. 소스를 신뢰하는 경우에만 계속하세요.',
      deepLinkTitle: 'MCP 서버를 추가하시겠습니까?',
      disabled: '사용 안 함',
      invalidJson: '잘못된 MCP JSON',
      loading: 'MCP 서버를 불러오는 중...',
      name: '이름',
      noOutput: '아직 출력이 없습니다.',
      reloadFailed: 'MCP 다시 불러오기 실패',
      remove: '제거',
      removeFailed: '제거 실패',
      saveFailed: '저장 실패',
      savedTitle: 'MCP 서버가 저장되었습니다',
      serverJson: '서버 JSON',
      statusConnecting: '연결 중…',
      statusError: '오류',
      statusNeedsAuth: '인증 필요',
      statusOff: '꺼짐',
      test: '연결 테스트'
    },
    modeOptions: {
      dark: {
        description: '눈부심이 적은 작업 공간',
        label: '어두운 테마'
      },
      light: {
        description: '밝은 데스크톱 화면',
        label: '밝은 테마'
      },
      system: {
        description: 'OS 외형 설정 따르기',
        label: '시스템'
      }
    },
    model: {
      appliesDesc: '새 세션에 적용됩니다. 컴포저의 모델 선택기를 사용하여 활성 채팅을 빠르게 전환하세요.',
      applying: '적용 중...',
      autoUseMain: '자동 · 메인 모델 사용',
      auxiliaryDesc: '도우미 작업은 기본적으로 메인 모델에서 실행됩니다. 재정의하려면 작업에 전용 모델을 할당하세요.',
      auxiliaryTitle: '보조 모델',
      change: '변경',
      chooseFromList: '목록에서 선택',
      customModel: '사용자 지정 모델…',
      customModelPlaceholder: '모델 ID',
      defaultsFailed: '모델 기본값 저장 실패',
      defaultsLabel: '기본값',
      fallbackAdd: '대체 모델 추가',
      fallbackEmpty: '대체 모델이 없습니다. 실패하지 않는 한 기본 모델이 사용됩니다.',
      inheritMainEffort: '상속 · 메인 모델 노력도',
      loadFailed: '모델을 불러올 수 없습니다',
      loading: '모델 구성을 불러오는 중...',
      mainAppliedTitle: '메인 모델이 업데이트되었습니다',
      moaAddPreset: '사전 설정 추가',
      moaAddReference: '참조 모델 추가',
      moaAggregator: '집계기',
      moaAggregatorBilled: '작업 모델 · 실행 비용 청구 대상',
      moaDefault: '기본값:',
      moaDescription:
        'Mixture of Agents 공급자 아래에 모델로 표시되는 이름 지정된 사전 설정을 구성합니다. 집계기는 작업 모델로서 도구 루프의 모든 단계를 실행하며, 실행 비용의 거의 전부가 해당 공급자에 청구됩니다. 참조는 기본적으로 사용자 턴당 한 번만 조언합니다.',
      moaEnabled: '사용함',
      moaNewPresetPlaceholder: '새 사전 설정',
      moaPreset: '사전 설정',
      moaReferenceHint: '기본적으로 턴당 한 번 조언',
      moaSetDefault: '기본값으로 설정',
      moaTitle: 'Mixture of Agents',
      model: '모델',
      notInCatalog: '이 공급자의 모델 목록에 없습니다. 호출이 백업으로 전환될 수 있습니다.',
      provider: '공급자',
      providerDefault: '(공급자 기본값)',
      reasoning: '추론',
      reasoningOff: '끄기',
      resetAllToMain: '모두 메인으로 초기화',
      restartBackend: '백엔드 재시작',
      restartFailed: '백엔드를 재시작할 수 없습니다',
      restartRequired: '이 백엔드는 업데이트 후 이전 코드를 실행 중입니다. 새 코드를 불러오려면 재시작하세요.',
      restartingBackend: '백엔드를 재시작하는 중...',
      setToMain: '메인으로 설정',
      setupProviderFallback: '공급자',
      staleAuxAfter: ', 메인 모델이 아님.',
      staleAuxDismiss: '다시 보지 않기',
      staleAuxOtherProviders: '다른 공급자',
      tasks: {
        approval: {
          hint: '스마트 자동 승인',
          label: '승인'
        },
        compression: {
          hint: '컨텍스트 압축',
          label: '압축'
        },
        curator: {
          hint: '스킬 사용 검토',
          label: '큐레이터'
        },
        kanban_decomposer: {
          hint: '작업 분해',
          label: 'Kanban 분해기'
        },
        mcp: {
          hint: 'MCP 도구 라우팅',
          label: 'MCP'
        },
        profile_describer: {
          hint: '자동 프로필 설명',
          label: '프로필 설명기'
        },
        review: {
          hint: '/review 검토자 서브에이전트',
          label: '검토'
        },
        skills_hub: {
          hint: '스킬 검색',
          label: '스킬 허브'
        },
        title_generation: {
          hint: '세션 제목',
          label: '제목 생성'
        },
        triage_specifier: {
          hint: 'Kanban 사양 구체화',
          label: '분류 사양기'
        },
        vision: {
          hint: '이미지 분석',
          label: '비전'
        }
      },
      speed: '속도',
      speedStandard: '표준'
    },
    notifications: {
      completionSoundDesc: '에이전트 턴이 완료될 때 재생됩니다. 사전 설정을 선택하고 여기서 미리 들어보세요.',
      completionSoundPreview: '미리보기',
      completionSoundTitle: '완료 사운드',
      enableAll: '알림 활성화',
      enableAllDesc: '끄면 아래의 모든 알림이 음소거됩니다.',
      focusedHint: '완료 알림은 Hermes가 백그라운드에 있을 때만 울립니다.',
      intro: 'OS 알림 (앱 내 토스트 아님). 기기별로 설정됩니다.',
      kinds: {
        approval: {
          description: '명령어 승인 또는 거부를 기다리고 있습니다.',
          label: '승인 필요'
        },
        backgroundDone: {
          description: '백그라운드 터미널 명령어가 완료되었습니다.',
          label: '백그라운드 작업 완료'
        },
        credits: {
          description: '크레딧 액세스가 일시 중지되었거나 복원되었습니다.',
          label: '크레딧 알림'
        },
        input: {
          description: 'Hermes가 질문을 했거나 비밀번호 또는 보안 정보가 필요합니다.',
          label: '입력 필요'
        },
        plugin: {
          description: 'Hermes가 백그라운드에 있는 동안 데스크톱 플러그인이 알림을 보냈습니다.',
          label: '플러그인 알림'
        },
        turnDone: {
          description: 'Hermes가 백그라운드에 있는 동안 턴이 완료되었습니다.',
          label: '응답 준비 완료'
        },
        turnError: {
          description: '백그라운드 턴 오류입니다.',
          label: '턴 실패'
        }
      },
      test: '테스트 알림 보내기',
      testBody: '알림이 정상적으로 작동하고 있습니다.',
      testSent: '테스트가 전송되었습니다. 아무것도 나타나지 않으면 OS 알림 권한과 방해 금지 모드를 확인하세요.',
      testTitle: 'Hermes',
      testUnsupported: '이 시스템은 네이티브 알림을 지원하지 않습니다.',
      title: '알림'
    },
    plugins: {
      failed: '실패',
      installModal: {
        agentFailed: '에이전트 플러그인 설치 실패',
        agentLabel: '에이전트 플러그인',
        description: '설치하기 전에 이 저장소의 내용을 검토하세요.',
        desktopFailed: '데스크톱 플러그인 설치 실패',
        desktopLabel: '데스크톱 UI',
        desktopOnlyNote: '데스크톱 전용 패키지는 백엔드 에이전트 플러그인을 설치하지 않습니다.',
        desktopTarget: '이 앱의 로컬 desktop-plugins 폴더에 설치합니다',
        desktopTargetFromPackage: '위 패키지에서 이 앱으로 로드됨 — 모든 프로필에 동일하게 적용됨',
        desktopUnavailable: '이 환경에서는 데스크톱 플러그인 설치를 사용할 수 없습니다.',
        enableAgent: '설치 후 에이전트 플러그인 활성화',
        forceReinstall: '강제 재설치 (이미 설치된 경우 교체)',
        gitCloneLabel: 'Git 클론 URL',
        includesHeading: '이 패키지 포함 항목',
        insecureWarning:
          '이 URL은 안전하지 않거나 로컬 스킴을 사용합니다. 프로덕션 설치 시에는 https:// 또는 git@를 사용하는 것이 좋습니다.',
        install: '설치',
        installFromGit: 'Git에서 설치',
        installUncertain:
          "Hermes가 설치 결과를 기다리는 것을 중단했지만 플러그인이 계속 설치 중일 수 있습니다. 이 대화상자를 닫고 '설치'를 다시 시도하기 전에 플러그인에서 '다시 검색'을 사용하세요.",
        installing: '설치 중…',
        missingEnvAction: '설정하기',
        nextChat: '다음 채팅에서 더 많은 도구를 사용할 수 있습니다',
        pinToCommit: '커밋에 고정 (선택 사항)',
        pinToCommitHint:
          '이 SHA를 설치하는 모든 사용자는 동일한 코드를 받게 되며, 플러그인은 다시 고정될 때까지 업데이트를 거부합니다. 최신 커밋을 사용하려면 비워 두세요.',
        pinToCommitInvalid: '전체 40자 커밋 SHA여야 합니다 (브랜치 및 태그는 허용되지 않음).',
        pinToCommitPlaceholder: '전체 40자 커밋 SHA',
        probeUnavailable: '이 환경에서는 플러그인 검사를 사용할 수 없습니다.',
        probing: '저장소 검사 중…',
        profileLabel: '프로필용 설치',
        repoLabel: '저장소',
        repoPlaceholder: 'https://github.com/owner/repo',
        reviewRepository: '저장소 검토',
        reviewedHeading: '검토된 카탈로그 항목',
        reviewedIntro:
          '이 항목은 고정된 커밋 시점에 사람이 직접 검토했습니다. 아래에서 코드를 직접 확인할 수도 있습니다.',
        securityHeading: '설치 전 확인 사항',
        securityIntro: '신뢰할 수 있는 소스에서만 설치하세요. 추가될 내용을 확인하려면 아래 저장소를 검토하세요.',
        selectComponent: '설치할 구성 요소를 하나 이상 선택하세요.',
        sourceHeading: '소스 코드',
        title: '플러그인 설치',
        viewPluginFiles: '플러그인 파일 보기',
        viewRepository: '저장소 보기'
      },
      kinds: {
        bundled: '번들됨',
        disk: '디스크에 있음',
        runtime: '런타임'
      },
      openFolder: '데스크톱 플러그인 폴더 열기',
      rescan: '다시 검색',
      reveal: '파일 관리자에서 표시',
      title: '데스크톱 플러그인'
    },
    poolLimits: {
      backendIdleTimeoutAria: '백엔드 유휴 시간 초과(밀리초)',
      backendIdleTimeoutTitle: '백엔드 유휴 시간 초과',
      warmBotBackendsAria: '대기 중인 봇 백엔드',
      warmBotBackendsTitle: '대기 중인 봇 백엔드'
    },
    profileScope: {
      appliesTo: '적용 대상'
    },
    providers: {
      collapse: '접기',
      connectAccount: '계정 연결',
      connectAnother: '다른 공급자 연결',
      connected: '연결됨',
      disconnect: '연결 해제',
      disconnectInTerminal: '연결 해제 (터미널에서 제거 명령 실행)',
      haveApiKey: 'API 키를 가지고 계신가요?',
      intro:
        '구독으로 로그인하세요. API 키를 복사할 필요가 없습니다. Hermes가 앱 내에서 브라우저 로그인을 직접 처리합니다.',
      loading: '공급자 로딩 중...',
      localEndpoint: {
        description: 'OpenAI 호환 엔드포인트(Zyphra, vLLM, llama.cpp, Ollama 등)를 Hermes에 연결합니다.',
        title: '로컬 / 사용자 정의 엔드포인트'
      },
      noKeysMatch: '검색 결과와 일치하는 공급자가 없습니다.',
      noProviderKeys: '사용 가능한 공급자 API 키가 없습니다.',
      otherProviders: '기타 공급자',
      removedTitle: '계정이 제거되었습니다.',
      searchKeys: '공급자 검색…'
    },
    quickEntry: {
      active: '단축키가 활성화되었습니다.',
      enabledDesc: '어디서나 전역 단축키로 작은 작성기를 불러와 Hermes를 열지 않고 프롬프트를 전송하세요.',
      enabledTitle: '빠른 입력',
      invalidShortcut: '유효하지 않은 단축키입니다. 수정자 키를 하나 이상 포함하세요.',
      shortcutDesc: 'CommandOrControl+Shift+Space 같은 수정자 키가 하나 이상 필요합니다.',
      shortcutTitle: '빠른 입력 단축키',
      takenBy: '다른 앱에서 이미 사용 중인 단축키입니다. 다른 단축키를 선택하세요.'
    },
    screenshot: {
      captureFailed: '맨 앞 창을 캡처할 수 없습니다. 첨부되거나 전송되지 않았습니다.',
      checking: '스크린샷 단축키 확인 중…',
      contextChanged: '캡처 중에 현재 초안이 변경되었습니다. 스크린샷이 첨부되거나 전송되지 않았습니다.',
      disabled: '스크린샷 단축키가 꺼져 있습니다.',
      enabledDesc:
        '어떤 앱에서든 양쪽 Command 키를 함께 눌러 맨 앞 창을 캡처하고 현재 Hermes 초안에 첨부하세요. 자동으로 전송되지 않습니다. 기본적으로 꺼져 있으며 이 Mac에만 적용됩니다. 창 내용에 민감한 정보가 포함될 수 있으므로 전송 전에 첨부파일을 확인하세요.',
      enabledTitle: '스크린샷 단축키',
      errorTitle: '스크린샷 단축키 오류',
      inputPermission:
        '입력 모니터링 권한을 통해 다른 앱이 활성화된 상태에서도 Hermes가 양쪽 Command 키를 감지할 수 있습니다. 시스템 설정 → 개인정보 보호 및 보안 → 입력 모니터링에서 Hermes를 허용하고 여기로 돌아와 다시 시도하세요.',
      loadFailed: '단축키 상태를 읽을 수 없습니다. 현재 설정을 확인하려면 다시 시도하세요.',
      openSettings: '시스템 설정 열기',
      permissionFailed: '시스템 설정을 열 수 없습니다. 수동으로 개인정보 보호 및 보안을 연 다음 다시 시도하세요.',
      ready: '단축키가 준비되었습니다. 스크린샷은 전송되지 않고 현재 초안에 첨부됩니다.',
      retry: '다시 시도',
      saveFailed: '단축키 변경 사항을 확인할 수 없습니다. 현재 설정을 확인하려면 다시 시도하세요.',
      screenPermission:
        '화면 기록 권한을 통해 이 단축키를 사용할 때 Hermes가 맨 앞의 앱 창을 캡처할 수 있습니다. 시스템 설정 → 개인정보 보호 및 보안 → 화면 기록에서 Hermes를 허용하고 여기로 돌아와 다시 시도하세요. macOS에서 요청하는 경우 Hermes를 다시 시작하세요.',
      starting: '단축키 리스너를 시작하는 중입니다. 아직 준비되지 않았습니다.',
      statusTitle: '스크린샷 단축키 상태',
      unavailable: '스크린샷 단축키를 사용할 수 없습니다. 다시 시도하거나 끄세요.'
    },
    search: {
      pill: '검색',
      placeholder: '모든 설정 검색…'
    },
    searchPlaceholder: {
      about: 'Hermes Desktop 정보',
      config: '설정 검색...',
      gateway: '게이트웨이 연결...',
      keys: 'API 키 검색...',
      mcp: 'MCP 서버 검색...',
      sessions: '보관된 세션 검색...'
    },
    sections: {
      advanced: '고급',
      appearance: '모양',
      chat: '채팅',
      memory: '메모리 및 컨텍스트',
      model: '모델',
      safety: '안전성',
      voice: '음성',
      workspace: '작업 공간'
    },
    sessions: {
      archivedIntro:
        '보관된 채팅은 사이드바에서 숨겨지지만 모든 메시지는 유지됩니다. 사이드바의 채팅을 Alt/⌥+Shift+클릭하여 보관할 수 있습니다.',
      archivedTitle: '보관된 세션',
      autoArchiveDaysLabel: '다음 기간 경과 후 보관:',
      autoArchiveDaysUnit: '일 동안 활동 없음',
      autoArchiveDesc:
        '한동안 사용하지 않은 채팅을 자동으로 보관합니다. 고정된 채팅은 보관되지 않으며 항목이 삭제되지 않고 보관된 채팅이 이쪽으로 이동합니다.',
      autoArchiveFailed: '자동 보관을 업데이트할 수 없습니다.',
      autoArchiveTitle: '오래된 채팅 자동 보관',
      change: '변경',
      choose: '선택',
      clear: '지우기',
      clearDirFailed: '기본 디렉터리를 지울 수 없습니다.',
      defaultDirDesc:
        '다른 폴더를 선택하지 않으면 새 세션이 이 폴더에서 시작됩니다. 홈 디렉터리를 사용하려면 비워 두세요.',
      defaultDirTitle: '기본 프로젝트 디렉터리',
      defaultDirUpdated:
        '기본 프로젝트 디렉터리가 업데이트되었습니다. 변경 사항을 적용하려면 새 채팅(Ctrl/⌘+N)을 시작하세요.',
      deleteFailed: '삭제 실패',
      deletePermanently: '영구 삭제',
      emptyArchivedDesc: '채팅을 보관하여 여기에 숨기세요.',
      emptyArchivedTitle: '보관된 항목 없음',
      failedLoad: '보관된 세션을 불러올 수 없습니다.',
      loading: '보관된 세션 로딩 중…',
      notSet: '설정되지 않음',
      restored: '복원됨',
      unarchive: '보관 취소',
      unarchiveFailed: '보관 취소 실패',
      updateDirFailed: '기본 디렉터리를 업데이트할 수 없습니다.'
    },
    toolsets: {
      activeBackend: '활성',
      activeBackendHint: '현재 활성화된 백엔드입니다.',
      browserRealProfile: {
        description:
          '기본 브라우저의 로그인 정보와 쿠키를 에이전트가 탐색하는 관리형 스냅샷으로 복사합니다. 라이브 프로필은 직접 열리지 않습니다. 새 세션에 적용됩니다.',
        disabledMessage: '프로필 스냅샷이 삭제됩니다. 새 세션은 깨끗한 브라우저를 사용합니다.',
        disabledTitle: '실제 프로필 브라우징 꺼짐',
        enabledMessage: '새 세션이 기본 브라우저 프로필의 스냅샷으로 탐색합니다.',
        enabledTitle: '실제 프로필 브라우징 켜짐',
        failedSave: '실제 프로필 설정을 저장할 수 없습니다.',
        label: '내 실제 브라우저 프로필 사용',
        prompt: {
          body: 'Hermes가 기본 브라우저 프로필의 스냅샷으로 브라우징하도록 허용하면 사이트가 이미 로그인된 상태로 열립니다.',
          bulletLiveProfile: '라이브 브라우저 프로필은 절대 직접 열리지 않습니다.',
          bulletLocal: '어떤 데이터도 이 컴퓨터를 떠나지 않습니다.',
          bulletSnapshot: '쿠키와 로그인 정보가 관리형 스냅샷으로 복사됩니다.',
          dontShowAgain: '다시 보지 않기',
          enable: '내 프로필 사용',
          notNow: '나중에',
          title: '사이트 로그인 상태 유지'
        }
      },
      failedLoad: '도구 구성을 불러오지 못했습니다.',
      loadingConfig: '구성 로딩 중',
      loadingModels: '모델 카탈로그 로딩 중...',
      modelDefault: '기본값',
      modelInUse: '사용 중',
      modelInactiveHint: '모델을 변경하려면 먼저 이 백엔드를 선택하세요.',
      modelSectionTitle: '모델',
      modelSelectedTitle: '선택된 모델',
      needsSetup: '설정 필요',
      needsSignIn: '로그인 필요',
      noApiKeyRequired: 'API 키가 필요하지 않습니다.',
      noProviderOptions: '이 툴셋에는 공급자 옵션이 없습니다. 활성화하면 현재 설정에서 작동합니다.',
      noProviders: '현재 이 툴셋에 사용할 수 있는 공급자가 없습니다.',
      notSet: '설정되지 않음',
      nousAuthDoneMessage: '구독 백엔드가 이제 활성화되었습니다.',
      nousAuthDoneTitle: 'Nous 계정 연결됨',
      nousAuthFailed: 'Nous 로그인이 완료되지 않았습니다.',
      nousAuthFailedMessage: '다시 시도하세요.',
      nousAuthNeededTitle: 'Nous 계정으로 로그인',
      nousAuthSignIn: '로그인',
      nousAuthTryAgain: '다시 시도',
      nousIncluded: 'Nous 구독에 포함되어 있습니다. Nous 계정으로 로그인하여 활성화하세요.',
      postSetupCompleteTitle: '설정 완료',
      postSetupErrorTitle: '오류와 함께 설정이 완료되었습니다.',
      postSetupInstalled: '설치됨',
      postSetupInstalledHint: '설치되었습니다. 문제가 있는 경우에만 설정을 다시 실행하세요.',
      postSetupOpenLogs: '로그 열기',
      postSetupRerun: '설정 다시 실행',
      postSetupRun: '설정 실행',
      postSetupRunAgain: '다시 실행',
      postSetupRunning: '설치 중…',
      postSetupStarting: '시작 중…',
      ready: '준비됨',
      removedTitle: '자격 증명이 제거되었습니다.',
      savedTitle: '자격 증명이 저장되었습니다.',
      selectedTitle: '선택된 공급자',
      set: '설정됨',
      terminalBackend: {
        failedLoad: '터미널 백엔드를 불러올 수 없습니다.',
        inUse: '사용 중',
        loading: '실행 백엔드 확인 중…',
        needsSetup: '설정 필요',
        needsSetupConfirmAction: '그래도 선택',
        needsSetupConfirmDescriptionGeneric:
          '이 백엔드는 아직 설정되지 않았습니다. 이 변경 후에 시작되는 세션은 설정이 완료될 때까지 터미널이나 파일 도구를 사용할 수 없습니다.',
        needsSetupHint: '이 백엔드는 현재 완전한 설정 없이 선택되어 있습니다. 설정이 완료될 때까지 명령이 실패합니다.',
        openBackendSettings: '터미널 설정 열기',
        ready: '준비됨',
        sectionTitle: '실행 백엔드',
        selectedTitle: '선택된 백엔드',
        switchedToLocal: '이제 터미널 명령이 로컬에서 실행됩니다. 새 세션에 적용됩니다.',
        unavailable: '사용 불가',
        unavailableTitle: '터미널 명령을 사용할 수 없습니다.',
        useLocal: '로컬 사용'
      },
      useBackend: '이 백엔드 사용',
      webCapabilityUnset: '설정되지 않음',
      webUseForExtract: '추출에 사용',
      webUseForSearch: '검색에 사용',
      webUsedForExtract: '추출 백엔드',
      webUsedForSearch: '검색 백엔드'
    },
    uninstallSection: {
      appLabel: '앱:',
      checkingInstalled: '설치된 항목 확인 중…',
      chooseHowMuch:
        '제거할 범위를 선택하세요. 작업 완료를 위해 앱이 종료되며, 언제든지 설치 프로그램을 다시 열어 돌아올 수 있습니다.',
      confirmUninstall: '제거 확인',
      couldNotStart: '제거를 시작할 수 없습니다.',
      dangerZone: '위험 구역',
      options: {
        full: {
          consequence: '모든 항목 — 채팅 GUI, Hermes 에이전트, 그리고 모든 설정, 채팅, 보안 비밀, 로그',
          description: '앱, 에이전트 및 모든 사용자 데이터(설정, 채팅, 예약된 작업, 보안 비밀, 로그)를 제거합니다.',
          title: '모든 항목 제거'
        },
        gui: {
          consequence: '데스크톱 채팅 GUI(이 앱 및 해당 데이터)',
          description: '이 데스크톱 앱을 제거합니다. Hermes 에이전트, 설정 및 채팅은 유지됩니다.',
          title: '채팅 GUI만 제거'
        },
        lite: {
          consequence: '채팅 GUI 및 Hermes 에이전트(설정, 채팅, 보안 비밀은 유지됨)',
          description:
            '앱과 Hermes 에이전트를 제거하되, 나중에 다시 설치할 수 있도록 설정, 채팅, 보안 비밀은 유지합니다.',
          title: 'GUI + 에이전트 제거, 내 데이터 유지'
        }
      },
      uninstallHermes: 'Hermes 제거',
      managedBody: '이 설치는 시스템에서 관리하므로 Hermes는 스스로 제거할 수 없습니다.',
      openAppsSettings: '앱 설정 열기',
      uninstalling: '제거 중…',
      yesUninstall: '예, 제거합니다.'
    },
    vault: {
      add: '추가',
      addConfirm: '저장',
      addDescription: '이 기기에 암호화되어 저장됩니다. 에이전트는 비밀번호를 볼 수 없습니다.',
      addTitle: '로그인, 카드 또는 주소 추가',
      added: '저장되었습니다.',
      adding: '저장 중...',
      addressLine1Field: '주소 1',
      addressLine2Field: '주소 2',
      blurb:
        '"GitHub에 로그인해 줘"라고 말하면 에이전트가 대신 로그인합니다. 로그인 페이지를 처음 만났을 때 그 자리에서 로그인을 물어보며, 그 후에는 자동으로 작동합니다. 비밀번호는 이 기기에 암호화되어 페이지에 바로 입력되므로 모델은 비밀번호를 볼 수 없습니다.',
      cardNameField: '카드 명의자',
      cardNumberField: '카드 번호',
      cityField: '시/군/구',
      countryField: '국가',
      cvcField: 'CVC',
      deleteAction: '저장된 항목 제거',
      deleteConfirm: '삭제',
      deleteTitle: '이 항목을 삭제하시겠습니까?',
      empty: '저장된 항목이 없습니다.',
      emptyDesc:
        "여기에 미리 무언가를 추가할 필요는 없습니다. 에이전트에게 사이트 로그인을 요청하면 현장에서 한 번 로그인을 물어봅니다. 미리 입력해 두고 싶다면 '추가'를 사용하세요.",
      expMonthField: '만료 월',
      expYearField: '만료 연도',
      identifierField: '아이디',
      identifierTypeField: '아이디 유형',
      identifierTypes: {
        email: '이메일',
        phone: '전화번호',
        username: '사용자 이름'
      },
      kindField: '종류',
      kinds: {
        address: '주소',
        login: '로그인',
        payment: '결제 카드'
      },
      labelField: '레이블',
      labelPlaceholder: '예: GitHub 회사 계정',
      labelRequired: '레이블이 필요합니다.',
      loadFailed: '보관함 항목을 불러오지 못했습니다.',
      loginFieldsRequired: '아이디와 비밀번호가 필요합니다.',
      optional: '(선택 사항)',
      originField: '사이트 출처',
      originInvalid: 'https://example.com과 같은 유효한 URL을 입력하세요.',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      otpField: '인증기 키',
      otpHint: '2FA를 활성화할 때 사이트에 표시되는 "설정 키"입니다. 이를 저장해 두면 Hermes가 코드를 직접 생성합니다.',
      otpPlaceholder: 'Base32 비밀 키 또는 otpauth:// 링크',
      passwordField: '비밀번호',
      postalField: '우편번호',
      sources: {
        blurb:
          '설치된 비밀번호 관리자가 자동으로 감지됩니다. 에이전트는 로그인 정보가 필요할 때 처음 한 번 잠금 해제를 요청하며(세션당 한 번), 세션 토큰만 메모리에 유지되고 마스터 비밀번호나 로그인 정보는 에이전트에게 절대 노출되지 않습니다.',
        disabledDesc: '감지되었지만 Hermes에서 꺼져 있습니다.',
        lock: '잠금',
        lockedDesc:
          '감지되었습니다. 로그인 정보가 필요할 때 에이전트가 잠금 해제를 요청하거나 지금 잠금을 해제할 수 있습니다.',
        masterPasswordPlaceholder: '마스터 비밀번호',
        statusLocked: '잠김',
        statusNotDetected: '감지되지 않음',
        statusOff: '꺼짐',
        statusUnlocked: '잠금 해제됨',
        title: '비밀번호 관리자',
        toggleFailed: '비밀번호 관리자를 업데이트하지 못했습니다.',
        unlock: '잠금 해제',
        unlockDescription:
          '마스터 비밀번호를 입력하세요. 이 기기의 비밀번호 관리자에 전달된 후 폐기되며, 저장되거나 기록되거나 에이전트에게 표시되지 않습니다.',
        unlockedDesc: '이 세션 동안 잠금 해제되었습니다. 30분 동안 유휴 상태이거나 Hermes가 닫히면 자동으로 잠깁니다.',
        unlocking: '잠금 해제 중...'
      },
      stateField: '주/지역',
      title: '비밀번호 및 로그인',
      twoFactorBadge: '2FA 자동'
    },
    nav: {
      plugins: '플러그인'
    },
    pluginPages: {
      agentSettings: '에이전트 설정',
      missing: '해당 플러그인에는 설정 페이지가 없습니다. 꺼져 있거나 제거되었을 수 있습니다.',
      blurb:
        '설치된 플러그인이 추가하는 옵션입니다. 각 플러그인은 자체 페이지를 갖고, 일부는 하위 페이지를 추가합니다.',
      empty: '설정이 있는 플러그인이 아직 없습니다.',
      manage: '플러그인 관리'
    }
  }
} satisfies Pick<
  TranslationOverrides,
  'billingBlock' | 'freeTier' | 'interfaceMode' | 'modelAssignment' | 'modelPicker' | 'modelVisibility' | 'settings'
>
