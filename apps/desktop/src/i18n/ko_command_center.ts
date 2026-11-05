/** Bulk machine-translated ko overlay — command_center keys (Gemini 3.5 Flash-Lite, then spot-checked). */
import type { TranslationOverrides } from './define-locale'

export const koCommandCenterOverrides = {
  commandCenter: {
    actionDone: '완료됨',
    actionFailed: '실패함',
    actionRunning: '실행 중',
    actionStartedWaiting: '작업이 시작되었습니다. 상태를 기다리는 중...',
    appearance: '모양',
    archivedChats: '보관된 채팅',
    back: '뒤로',
    branches: '브랜치',
    changeColorMode: '색상 모드 변경…',
    changeTheme: '테마 변경',
    close: '명령 센터 닫기',
    commandCenter: '명령 센터',
    commands: '명령어',
    dailyTokens: '일일 토큰',
    deleteSession: '세션 삭제',
    exportSession: '세션 내보내기',
    gatewayRestartFailed: '게이트웨이를 다시 시작하지 못했습니다.',
    gatewayRunning: '메시징 Gat웨이 실행 중',
    gatewayStopped: '메시징 게이트웨이 중지됨',
    generatePet: {
      adopt: '분양받기',
      backgroundHint: '창을 닫아도 됩니다. 완료되면 Hermes가 알려드립니다.',
      generate: '생성',
      generating: '생성 중…',
      genericError: '생성 실패 — 다시 시도하거나 추천 항목을 선택하세요.',
      hatch: '부화',
      hatchComposing: '조합하는 중…',
      hatchSaving: '거의 다 되었습니다…',
      hatched: '부화했습니다!',
      hatching: '펫을 부화시키는 중…',
      hatchingSub: '생명을 불어넣는 중…',
      namePlaceholder: '펫의 이름을 지어주세요',
      placeholder: '생성할 펫을 묘사하세요…',
      promptHint: '설명을 입력한 후 Enter를 눌러 4가지 스타일을 초안으로 만드세요.',
      readyHint: 'Enter를 눌러 설명에서 4가지 스타일의 초안을 만드세요.',
      referenceImageInvalid: '참조 이미지를 읽을 수 없습니다. PNG, JPG, WebP 또는 GIF를 사용해 보세요.',
      referenceImageTooLarge: '참조 이미지가 너무 큽니다. 16MB 미만의 이미지를 사용하세요.',
      remix: '리믹스',
      remixConfirmBody: '현재 펫을 시작점으로 삼아 새로운 초안 세트를 생성합니다. 몇 분 정도 걸릴 수 있습니다.',
      remixConfirmTitle: '이 스타일을 리믹스하시겠습니까?',
      retry: '다시 시도',
      slowProviderHint: '몇 분 정도 걸릴 수 있습니다',
      spawning: '소환 중…',
      staleBackend: '펫을 생성하려면 Hermes를 업데이트하세요.',
      startOver: '처음부터 다시 시작',
      title: '펫 생성'
    },
    goTo: '이동',
    goToSession: '세션으로 이동',
    input: '입력',
    installTheme: {
      empty: '일치하는 테마가 없습니다.',
      error: '마켓플레이스에 접속할 수 없습니다.',
      install: '설치',
      installed: '설치됨',
      installing: '설치 중...',
      loading: '마켓플레이스 검색 중...',
      pageTitle: '테마 설치',
      placeholder: 'VS Code 마켓플레이스 검색...',
      title: '테마 설치…'
    },
    loadingStatus: '상태 불러오는 중...',
    loadingUsage: '사용량 불러오는 중...',
    logFile: '로그 파일',
    logLevel: '수준',
    logSearchPlaceholder: '로그 검색...',
    maintenance: {
      backup: '백업 만들기',
      backupDesc: '설정, 메모리, 스킬 및 세션 압축(Zip)',
      builtinMemory: '기본 제공',
      copyLink: '링크 복사',
      curator: '스킬 큐레이터',
      curatorActive: '활성',
      curatorDesc: '사용되지 않는 에이전트 생성 스킬을 보관 처리하는 백그라운드 검토',
      curatorDisabled: '사용 안 함',
      curatorNeverRan: '실행된 적 없음',
      curatorPaused: '일시 중지됨',
      debugShare: '디버그 공유',
      debugShareDesc: '민감한 정보가 가려진 보고서 및 로그를 업로드하고 공유 가능한 링크 생성 (6시간 후 자동 삭제)',
      debugShareFailed: '디버그 공유 실패',
      debugShareLinks: '공유 링크',
      debugShareRunning: '디버그 보고서 업로드 중...',
      doctor: '진단 실행',
      doctorDesc: '설치 상태, 설정 및 공급자 상태 검사',
      empty: '비어 있음',
      linkCopied: '링크가 복사되었습니다',
      memoryData: '메모리 데이터',
      memoryDataDesc: '모든 세션에 주입되는 기본 제공 메모리 파일',
      memoryFile: '에이전트 메모리 (MEMORY.md)',
      pause: '일시 중지',
      resetAll: '둘 다 재설정',
      resetFailed: '메모리 재설정 실패',
      resetMemory: '메모리 재설정',
      resetUser: '프로필 재설정',
      resume: '재개',
      runNow: '지금 실행',
      runOps: '진단',
      running: '실행 중...',
      securityAudit: '보안 감사',
      securityAuditDesc: '설정과 스킬에서 위험한 설정 스캔',
      userFile: '사용자 프로필 (USER.md)',
      viewLog: '작업 로그'
    },
    mcpServers: 'MCP 서버',
    nav: {
      artifacts: {
        detail: '생성된 출력물 찾아보기',
        title: '아티팩트'
      },
      capabilities: {
        detail: '스킬, 도구, MCP 서버 및 플러그인',
        title: '기능'
      },
      messaging: {
        detail: 'Telegram, Slack, Discord 등 설정',
        title: '메시징'
      },
      newChat: {
        detail: '새 세션 시작',
        title: '새 세션'
      },
      settings: {
        detail: 'Hermes 데스크톱 설정',
        title: '설정'
      }
    },
    noDailyActivity: '일일 활동이 없습니다.',
    noLogs: '불러온 로그가 없습니다.',
    noModelUsage: '모델 사용량이 없습니다.',
    noResults: '일치하는 결과를 찾을 수 없습니다.',
    noSessions: '세션이 없습니다.',
    noSkillActivity: '스킬 활동이 없습니다.',
    openBrowser: '브라우저 토글',
    openFolder: '폴더를 프로젝트로 열기…',
    output: '출력',
    paletteTitle: '명령 팔레트',
    pets: {
      adoptFailed: '해당 펫을 분양받을 수 없습니다.',
      empty: '일치하는 펫이 없습니다.',
      error: 'Petdex 갤러리에 접속할 수 없습니다.',
      generatedTag: '생성됨',
      installed: '설치됨',
      loading: 'Petdex 갤러리 불러오는 중…',
      noneAvailable: '사용 가능한 펫이 없습니다. 아래에서 설치할 펫을 선택하세요.',
      placeholder: '펫 검색…',
      staleBackend: '펫을 사용하려면 Hermes를 다시 시작하세요. 백엔드 버전이 이 기능보다 이전 버전입니다.',
      title: '펫',
      turnOff: '끄기',
      turnOn: '켜기'
    },
    pinSession: '세션 고정',
    projects: '프로젝트',
    providerNavigate: '이동',
    providerSessions: '세션',
    recentLogs: '최근 로그',
    refresh: '새로고침',
    refreshing: '새로고침 중...',
    reloadWindow: '창 새로고침',
    restartGateway: 'Gateway 재시작',
    retry: '다시 시도',
    searchPlaceholder: '세션, 보기 및 작업 검색',
    sectionEntries: {
      sessions: {
        detail: '세션 검색, 고정 및 관리',
        title: '세션 패널'
      },
      system: {
        detail: 'Gateway 상태, 로그, 재시작/업데이트',
        title: '시스템 패널'
      },
      usage: {
        detail: '토큰, 비용 및 스킬 사용량',
        title: '사용량 패널'
      }
    },
    sections: {
      maintenance: '유지보수',
      sessions: '세션',
      system: '시스템',
      usage: '사용량'
    },
    settings: '설정',
    settingsFields: '설정 필드',
    sharedGatewayRestartConfirm: '모두 재시작',
    sharedGatewayRestartTitle: '공유 Gateway를 재시작하시겠습니까?',
    statApiCalls: 'API 호출',
    statCost: '예상 비용',
    statSessions: '세션',
    statTokens: '입출력 토큰',
    toggleBrowser: '브라우저 전환',
    topModels: '상위 모델',
    topSkills: '상위 스킬',
    unpinSession: '세션 고정 해제',
    updateHermes: 'Hermes 업데이트'
  },
  connectors: {
    authorizedToolsUnavailable: '인증됨. 도구를 사용할 수 없습니다.',
    cancel: '대기 중지',
    checking: '앱 확인 중…',
    connect: '연결',
    connectError: '인증을 시작할 수 없습니다. 다시 시도해 주세요.',
    connected: '연결됨',
    disabled: '사용 불가',
    disclaimer: '연결은 선택사항입니다. Hermes가 사용하기 원하는 앱만 인증해 주세요.',
    empty: '일치하는 앱 없음',
    execution: '커넥터 도구',
    failed: '연결 실패',
    grant: '재연결',
    needsAuth: '액세스 만료됨',
    notConnected: '연결되지 않음',
    openInBrowser: '브라우저에서 열기',
    opening: '로그인 창 여는 중…',
    ownerMissing: '연결을 관리하려면 이 대화를 다시 여세요.',
    refresh: '상태 새로고침',
    required: '필수',
    retry: '다시 시도',
    search: '앱 찾기',
    setupCancel: '취소',
    skip: '나중에',
    skipped: '건너뜀',
    timeout: '여전히 인증을 기다리는 중입니다.',
    title: '앱 연결',
    unavailable: '이 세션에서는 커넥터를 사용할 수 없습니다.',
    waiting: '브라우저 대기 중…'
  },
  connectorsPage: {
    add: {
      action: '직접 추가',
      addArg: '+ 인자 추가',
      addEnvVar: '+ 환경 변수 추가',
      addHeader: '+ 헤더 추가',
      addPassthrough: '+ 변수 추가',
      args: '인자',
      auth: '인증',
      authBearer: 'Bearer 토큰',
      authNone: '없음',
      authOauth: 'OAuth',
      command: '실행 명령',
      cwd: '작업 디렉토리',
      editJson: 'mcp.json 편집',
      envVars: '환경 변수',
      headers: '헤더',
      hint: '이 기기의 mcp.json에 새 항목 1개가 추가됩니다',
      keyPlaceholder: '키',
      name: '이름',
      nameTaken: '이미 사용 중인 이름입니다.',
      passthrough: '환경 변수 패스스루',
      pasteLabel: '명령 또는 스니펫 붙여넣기',
      pasteNoMatch: '서버로 인식된 내용이 없습니다. 대신 아래 필드를 채워주세요.',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /path/to/dir',
      removeRow: '이 행 제거',
      saveFailed: '해당 서버가 저장되지 않았습니다.',
      title: '사용자 정의 MCP에 연결',
      type: '유형',
      typeHttp: '스트리밍 가능 HTTP',
      typeStdio: 'STDIO',
      url: 'URL',
      valuePlaceholder: '값'
    },
    card: {
      alsoLocal: '이 기기에서도 실행됨',
      hostedTwin: '관리형 버전 사용 가능',
      inCatalog: 'Hermes 카탈로그에 있음',
      kindCatalog: 'MCP · 카탈로그',
      kindCustom: 'MCP · 사용자 정의',
      kindManaged: '관리형',
      reason: {
        finishSignIn: '브라우저에서 로그인을 완료하세요.',
        reconnect: '이 앱을 계속 사용하려면 다시 연결하세요.',
        serverError: '서버에서 연결을 거부했습니다.',
        serverNeedsAuth: '서버가 응답할 수 있도록 로그인하세요.'
      },
      state: {
        accessExpired: '액세스 만료됨',
        available: '사용 가능',
        connected: '연결됨',
        connecting: '연결 중',
        connectionUnknown: '상태 알 수 없음',
        couldNotConnect: '연결할 수 없음',
        offByYourOrganisation: '조직에 의해 비활성화됨',
        offForYou: '나에게 비활성화됨',
        serverConnecting: '연결 중…',
        serverError: '오류',
        serverNeedsAuth: '인증 필요',
        serverOff: '끔',
        serverOn: '켬',
        serverOnUnused: '켜짐, 미사용'
      },
      verb: {
        authenticate: '인증',
        connect: '연결',
        install: '설치',
        openLogs: '로그 열기',
        reconnect: '재연결',
        stopWaiting: '대기 중지',
        tryAgain: '다시 시도',
        turnBackOn: '다시 켜기'
      }
    },
    categoryAll: '모든 카테고리',
    dialog: {
      advanced: '고급',
      advancedHint: 'mcp.json 항목 및 로그',
      connectEnded: '로그인이 완료되지 않았습니다.',
      connectOpenAgain: '링크 다시 열기',
      disconnect: '연결 해제',
      disconnectBody: 'Hermes가 더 이상 이 계정으로 작동하지 않습니다. 언제든지 다시 연결할 수 있습니다.',
      menuRefreshTools: '도구 새로고침',
      moreActions: '추가 작업',
      nousLine: 'Nous 앱은 프로필이 아닌 계정을 따릅니다.',
      openPlugins: '플러그인 탭 열기',
      orgLink: '커넥터 관리자 열기',
      removeServerBody: '이 컴퓨터의 mcp.json에서 항목이 제거됩니다. 다른 항목은 삭제되지 않습니다.',
      rulesReadOnly: '지금은 규칙을 변경할 수 없습니다.',
      rulesSignIn: 'Hermes가 여기서 수행할 수 있는 작업을 변경하려면 로그인하세요.',
      tokensPerCall: '호출당 토큰',
      turnOffLocal: '로컬 서버 끄기',
      usesPerMonth: '30일간 사용 횟수',
      wayHosted: '관리형'
    },
    filterCategory: '카테고리',
    group: {
      available: '사용 가능',
      connected: '연결됨',
      connectedNote: '연결이 끊긴 항목이 먼저 표시됩니다.',
      off: '꺼짐',
      offNote: '로그인은 유지됩니다.'
    },
    page: {
      clearSearch: '검색 지우기',
      disconnectNoAccount: '연결을 해제할 Hermes 계정이 없습니다. 페이지를 새로고침하고 다시 시도해 주세요.',
      disconnectRefused:
        '지금 Nous에서 이 로그인을 제거할 수 없습니다. 대신 스위치를 사용하여 앱을 끄거나 나중에 다시 시도해 주세요.',
      emptyTitle: '아직 앱이 없습니다. 시작하려면 이 컴퓨터에 서버를 추가하세요.',
      freeTierNote: '로그인할 때까지 연결은 이 컴퓨터에 유지됩니다.',
      hostedFailedBody: '이 컴퓨터의 서버는 영향을 받지 않으며 계속 실행 중입니다. 꺼진 항목이 없습니다.',
      hostedFailedTitle: '호스팅된 앱에 연결할 수 없습니다.',
      loading: '카탈로그 및 이 컴퓨터의 서버를 읽는 중',
      managedUnavailable: '이 계정에서는 아직 관리형 앱을 사용할 수 없습니다.',
      noMatchBody: '일치하는 항목이 없습니다. 자체 MCP 서버를 Hermes에 지정하여 추가하세요.',
      noMatchTitle: '일치하는 앱 없음',
      refreshFailed: '도구 목록을 새로고침하지 못했습니다.',
      retry: '다시 시도',
      showAllMatches: '일치하는 항목 모두 보기',
      signIn: '로그인',
      signInLine: '관리형 앱을 사용하려면 Nous에 로그인하세요.',
      writeFailed: '변경 사항이 저장되지 않았습니다.'
    },
    residencyLocal: '이 기기에서',
    segment: {
      all: '전체',
      available: '사용 가능',
      connected: '연결됨',
      off: '꺼짐'
    },
    title: '커넥터',
    tools: {
      allToolsSwitch: '모든 도구 켜기 또는 끄기',
      conflictReload: '상대방 버전으로 새로고침',
      conflictSave: '상대방 버전을 덮어쓰고 저장',
      conflictTitle: '편집하는 동안 다른 사람이 이 규칙을 변경했습니다.',
      discard: '변경사항 취소',
      goneBody:
        'Hermes가 더 이상 이 도구를 호출할 수 없습니다. 제거할 때까지 행이 유지되므로 아무것도 사라지지 않습니다.',
      loading: '도구 목록 읽는 중',
      lockedHint: '조직에 의해 비활성화됨',
      needsAuthBody: '로그인은 이 컴퓨터에 유지됩니다. 외부로 유출되지 않습니다.',
      noMatch: '필터와 일치하는 도구가 없습니다.',
      notInstalledBody: '제공하는 도구를 보려면 이 기기에 설치하세요.',
      offBody: '제공하는 도구를 보려면 위의 스위치로 켜세요.',
      quickEverythingOn: '모두 켜기',
      quickNoDestructive: '파괴적 도구 끄기',
      quickReadOnly: '읽기 전용',
      remove: '제거',
      retry: '다시 시도',
      save: '변경사항 저장',
      saveFailed: '도구 규칙이 저장되지 않았습니다.',
      saving: '저장 중...',
      showSummary: '요약 보기',
      signedOutBody: '이 컴퓨터의 서버는 영향을 받지 않습니다.',
      signedOutTitle: '도구 목록을 읽려면 Nous에 로그인하세요.',
      staleSignIn: '최신 도구 목록을 읽으려면 로그인하세요.',
      summaryAllOn: '모두 켜짐',
      summaryAllTools: '모든 도구',
      summaryOff: '꺼짐',
      summaryOther: '기타',
      title: '도구',
      unavailableLine: '도구 목록을 사용할 수 없습니다.'
    },
    uncategorised: '미분류',
    vocabulary: {
      facetDestructive: {
        label: '파괴적',
        long: '앱 내의 무언가를 영구적으로 삭제할 수 있습니다.'
      },
      facetRead: {
        label: '읽기',
        long: '앱에서 데이터를 읽어오기만 하며, 아무것도 변경하지 않습니다.'
      },
      facetUnclassified: {
        label: '알 수 없는 효과',
        long: '이 도구가 어떤 작업을 하는지 앱에 명시되어 있지 않습니다.'
      },
      facetWrite: {
        label: '쓰기',
        long: '앱 내의 무언가를 생성하거나 변경합니다.'
      },
      hintCreate: {
        label: '생성',
        long: '새로운 항목을 만듭니다.'
      },
      hintDelete: {
        label: '삭제',
        long: '항목을 제거합니다.'
      },
      hintDestructive: {
        label: '파괴적',
        long: '여기서 변경한 내용은 되돌릴 수 없습니다.'
      },
      hintIdempotent: {
        label: '반복 가능',
        long: '두 번 실행해도 한 번 실행한 것과 결과가 같습니다.'
      },
      hintOpenWorld: {
        label: '외부',
        long: '앱 외부의 대상에 접근합니다.'
      },
      hintReadOnly: {
        label: '읽기 전용',
        long: '읽기 작업만 수행하는 도구입니다.'
      },
      hintUpdate: {
        label: '업데이트',
        long: '이미 존재하는 항목을 변경합니다.'
      }
    }
  },
  cron: {
    actionsTitle: 'Cron 작업 작업',
    blueprints: {
      custom: '사용자 지정',
      dialogDesc: '세부 정보를 입력하고 일정을 설정하세요.',
      emptyDesc: '이 백엔드에서 사용할 수 있는 자동화 블루프린트가 없습니다.',
      emptyTitle: '사용 가능한 블루프린트 없음',
      failedLoad: '블루프린트를 불러오지 못했습니다',
      loading: '블루프린트를 불러오는 중...',
      scheduleIt: '일정 예약',
      scheduled: '블루프린트가 예약되었습니다',
      scheduling: '예약 중...',
      startFrom: '시작 기준',
      subtitle: '기성 자동화 기능',
      tab: '블루프린트'
    },
    close: 'Cron 닫기',
    createAction: 'Cron 생성',
    createDesc:
      '프롬프트를 자동으로 실행하도록 예약하세요. cron 구문이나 "15분마다"와 같은 자연어 표현을 사용할 수 있습니다.',
    createTitle: '새로운 Cron 작업',
    created: 'Cron이 생성되었습니다',
    customHint: 'Cron 표현식 또는 "매시간"이나 "평일 오전 9시"와 같은 문구입니다.',
    customPlaceholder: '0 9 * * * 또는 평일 오전 9시',
    customScheduleLabel: '사용자 지정 일정',
    days: {
      '0': '일요일',
      '1': '월요일',
      '2': '화요일',
      '3': '수요일',
      '4': '목요일',
      '5': '금요일',
      '6': '토요일',
      '7': '일요일'
    },
    deleteDescPrefix: '이 작업은 ',
    deleteDescSuffix: ' 항목을 영구적으로 제거합니다. 즉시 실행이 중단됩니다.',
    deleteTitle: 'Cron 작업을 삭제하시겠습니까?',
    deleted: 'Cron이 삭제되었습니다',
    deleting: '삭제 중...',
    deliverLabel: '전송 대상',
    deliverNeedsHomeChannel: '먼저 홈 채널을 설정하세요',
    deliveryLabels: {
      discord: 'Discord',
      email: '이메일',
      local: '이 데스크톱',
      slack: 'Slack',
      telegram: 'Telegram'
    },
    edit: 'Cron 편집',
    editDesc: '일정, 프롬프트 또는 전송 대상을 업데이트합니다. 변경 사항은 다음 실행 시 적용됩니다.',
    editJob: '작업 편집',
    editTitle: 'Cron 작업 편집',
    emptyDescNew:
      'cron 표현식에 따라 프롬프트가 실행되도록 예약하세요. Hermes가 이를 실행하고 선택한 대상으로 결과를 전송합니다.',
    emptyDescSearch: '더 넓은 검색어로 다시 시도해 보세요.',
    emptyTitleNew: '예약된 작업이 없습니다',
    emptyTitleSearch: '일치하는 항목 없음',
    failedDelete: 'Cron 작업 삭제에 실패했습니다',
    failedLoad: 'Cron 작업 불러오기에 실패했습니다',
    failedSave: 'Cron 작업 저장에 실패했습니다',
    failedTrigger: 'Cron 작업 수동 실행에 실패했습니다',
    failedUpdate: 'Cron 작업 업데이트에 실패했습니다',
    frequencyLabel: '빈도',
    hideRuns: '실행 기록 숨기기',
    last: '마지막 실행:',
    lastRunFailed: '마지막 실행 실패:',
    loading: 'Cron 작업 불러오는 중...',
    manage: '관리',
    modelDefault: '기본값 (전역 모델)',
    modelLabel: '모델',
    nameLabel: '이름',
    namePlaceholder: '아침 브리핑',
    newCron: '새 Cron',
    next: '다음 실행:',
    noRuns: '실행 기록이 없습니다',
    optional: '선택 사항',
    overdueSince: '예정 시간 초과:',
    pause: 'Cron 일시 중지',
    pauseTitle: '일시 중지',
    paused: 'Cron이 일시 중지되었습니다',
    promptLabel: '프롬프트',
    promptPlaceholder: '읽지 않은 Slack 스레드를 요약하고 상위 5개를 이메일로 보내줘...',
    promptRequired: '프롬프트는 필수입니다.',
    promptScheduleRequired: '프롬프트와 일정을 모두 입력해야 합니다.',
    resume: 'Cron 재개',
    resumeTitle: '재개',
    resumed: 'Cron이 재개되었습니다',
    runAgain: '다시 실행',
    runHistory: '실행 기록',
    saveChanges: '변경 사항 저장',
    scheduleHints: {
      custom: 'Cron 구문 또는 자연어',
      daily: '매일 오전 9:00',
      'every-15-minutes': '15분마다',
      hourly: '매시간 정각',
      monthly: '매월 1일 오전 9:00',
      weekdays: '월요일부터 금요일까지 오전 9:00',
      weekly: '매주 월요일 오전 9:00'
    },
    scheduleLabels: {
      custom: '사용자 지정',
      daily: '매일',
      'every-15-minutes': '15분마다',
      hourly: '매시간',
      monthly: '매월',
      weekdays: '평일',
      weekly: '매주'
    },
    scheduleRequired: '일정은 필수입니다.',
    scriptBadge: '스크립트',
    scriptLabel: '스크립트',
    scriptOnlyEditHint: '스크립트 전용 작업 (AI 프롬프트 없음). 작업 ID:',
    search: 'Cron 작업 검색...',
    showRuns: '실행 기록 표시',
    states: {
      completed: '완료됨',
      disabled: '비활성화됨',
      enabled: '활성화됨',
      error: '마지막 실행 실패',
      paused: '일시 중지됨',
      running: '실행 중',
      scheduled: '예약됨'
    },
    tabs: {
      blueprints: '블루프린트',
      jobs: '작업'
    },
    title: '예약된 작업',
    topOfHour: '매시간 정각',
    triggerNow: '지금 실행',
    triggered: 'Cron이 실행되었습니다',
    updated: 'Cron이 업데이트되었습니다'
  },
  messaging: {
    appliedLive: '실행 중인 Gateway에 적용되었습니다.',
    approve: '승인',
    approvedHint: '다음 메시지에서 자동으로 인식됩니다.',
    approving: '승인 중...',
    connectingLive: '실행 중인 Gateway가 새 자격 증명으로 연결 중입니다.',
    credentialsSet: '자격 증명 설정됨',
    disabled: '사용 안 함',
    enabled: '사용 중',
    fieldCopy: {
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        help: '참인 경우 BlueBubbles 허용 목록을 건너뜁니다.',
        label: '모든 iMessage 사용자 허용'
      },
      DISCORD_ALLOWED_USERS: {
        help: '권장됨. 쉼표로 구분된 Discord 사용자 ID.',
        label: '허용된 Discord 사용자 ID'
      },
      DISCORD_ALLOW_ALL_USERS: {
        help: '개발 전용. 참인 경우 허용 목록 없이 누구나 봇에게 DM을 보낼 수 있습니다.',
        label: '모든 Discord 사용자 허용'
      },
      DISCORD_BOT_TOKEN: {
        help: 'Discord Developer Portal에서 애플리케이션을 만들고, 봇을 추가한 후 토큰을 붙여넣으세요.',
        label: '봇 토큰'
      },
      DISCORD_HOME_CHANNEL: {
        help: '봇이 선제적 메시지(Cron 출력, 알림)를 보내는 채널입니다.',
        label: '홈 채널 ID'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        help: '로그 및 상태 출력에서 홈 채널에 사용할 표시 이름입니다.',
        label: '홈 채널 이름'
      },
      DISCORD_REPLY_TO_MODE: {
        help: 'first, all, 또는 off.',
        label: '답장 스타일'
      },
      MATRIX_ACCESS_TOKEN: {
        label: '액세스 토큰'
      },
      MATRIX_ALLOWED_USERS: {
        help: '권장됨. @user:server 형식의 쉼표로 구분된 사용자 ID.',
        label: '허용된 Matrix 사용자 ID'
      },
      MATRIX_HOMESERVER: {
        label: '홈서버 URL',
        placeholder: 'https://matrix.org'
      },
      MATRIX_USER_ID: {
        label: '봇 사용자 ID',
        placeholder: '@hermes:example.org'
      },
      MATTERMOST_ALLOWED_USERS: {
        help: '권장됨. 쉼표로 구분된 Mattermost 사용자 ID.',
        label: '허용된 사용자 ID'
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: '모든 Mattermost 사용자 허용'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: '홈 채널'
      },
      MATTERMOST_TOKEN: {
        label: '봇 토큰'
      },
      MATTERMOST_URL: {
        label: '서버 URL',
        placeholder: 'https://mattermost.example.com'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'Cron 전달을 위한 기본 채널 또는 그룹입니다.',
        label: 'QQ 홈 채널'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'QQ 홈 채널 이름'
      },
      QQ_ALLOW_ALL_USERS: {
        label: '모든 QQ 사용자 허용'
      },
      SIGNAL_ACCOUNT: {
        help: 'signal-cli 브리지에 등록된 번호입니다.',
        label: '전화번호'
      },
      SIGNAL_ALLOWED_USERS: {
        help: '권장됨. 쉼표로 구분된 Signal 식별자.',
        label: '허용된 Signal 사용자'
      },
      SIGNAL_HTTP_URL: {
        help: '실행 중인 signal-cli REST 브리지의 URL입니다.',
        label: 'Signal 브리지 URL',
        placeholder: 'http://127.0.0.1:8080'
      },
      SLACK_ALLOWED_USERS: {
        help: '권장됨. 쉼표로 구분된 Slack 사용자 ID.',
        label: '허용된 Slack 사용자 ID'
      },
      SLACK_APP_TOKEN: {
        help: 'Socket Mode에 필요한 앱 수준 토큰을 사용하세요.',
        label: 'Slack 앱 토큰',
        placeholder: 'Slack 앱 토큰 붙여넣기'
      },
      SLACK_BOT_TOKEN: {
        help: 'Slack 앱을 설치한 후 OAuth & Permissions의 봇 토큰을 사용하세요.',
        label: 'Slack 봇 토큰',
        placeholder: 'Slack 봇 토큰 붙여넣기'
      },
      TELEGRAM_ALLOWED_USERS: {
        help: '권장됨. @userinfobot에서 제공하는 쉼표로 구분된 숫자 ID. 이 설정이 없으면 누구나 봇에게 DM을 보낼 수 있습니다.',
        label: '허용된 Telegram 사용자 ID'
      },
      TELEGRAM_BOT_TOKEN: {
        help: '@BotFather로 봇을 만든 후 발급받은 토큰을 붙여넣으세요.',
        label: '봇 토큰',
        placeholder: 'Telegram 봇 토큰 붙여넣기'
      },
      TELEGRAM_PROXY: {
        help: 'Telegram이 차단된 네트워크에서만 필요합니다.',
        label: '프록시 URL'
      },
      WHATSAPP_ALLOWED_USERS: {
        help: '권장됨. 쉼표로 구분된 전화번호 또는 WhatsApp ID.',
        label: '허용된 WhatsApp 사용자'
      },
      WHATSAPP_ENABLED: {
        help: '아래 토글로 자동 설정됩니다. 꼭 필요한 경우가 아니라면 그대로 두세요.',
        label: 'WhatsApp 브리지 사용'
      },
      WHATSAPP_MODE: {
        label: '브리지 모드'
      }
    },
    gatewayStopped: '메시징 Gateway가 중지되었습니다',
    getCredentials: '자격 증명 가져오기',
    hintGatewayStopped: '연결하려면 상태 표시줄에서 Gateway를 시작하세요.',
    hintPendingRestart: '이 변경 사항을 적용하려면 상태 표시줄에서 Gateway를 다시 시작하세요.',
    loadFailed: '메시징 플랫폼을 불러오지 못했습니다',
    loading: '메시징 플랫폼 불러오는 중...',
    needsSetup: '설정 필요',
    noTokenNeeded:
      '이 플랫폼에는 여기에서 토큰이 필요하지 않습니다. 위의 설정 가이드를 사용한 후 아래에서 활성화하세요.',
    openDocs: '문서 열기',
    openLogs: '로그 열기',
    openSetupGuide: '설정 가이드 열기',
    pairingLockedOut: '승인 실패 횟수가 너무 많아 이 플랫폼이 잠겼습니다. 나중에 다시 시도하세요.',
    recommended: '권장',
    replaceValue: '현재 값 바꾸기',
    required: '필수',
    restartAgain: '다시 시작',
    restartFailedManual: '메시징 설정을 적용하기 위해 Hermes를 다시 시작하지 못했습니다',
    restartFailedManualDetail: "'다시 시작'을 다시 시도해 보시고, 계속 실패하면 로그를 열어 진단 정보를 보내주세요.",
    restartNeeded: '저장되었습니다. 새 설정이 적용되도록 메시징 Gateway를 다시 시작하세요.',
    restartNow: '지금 다시 시작',
    restartToApply: '이 변경 사항은 Gateway를 다시 시작한 후에 적용됩니다.',
    restartToReconnect: '새 자격 증명은 Gateway를 다시 시작한 후에 적용됩니다.',
    restarting: '다시 시작하는 중…',
    revoke: '취소',
    revokeTitle: '액세스 취소',
    revoking: '취소 중...',
    saveChanges: '변경 사항 저장',
    saved: '저장됨',
    saving: '저장 중...',
    search: '메시징 검색...',
    sharedListenerUrl: '공유 Gateway 리스너에서 제공됨:',
    states: {
      connected: '연결됨',
      connecting: '연결 중',
      disabled: '사용 안 함',
      fatal: '오류',
      gateway_stopped: '메시징 Gateway가 중지됨',
      not_configured: '설정 필요',
      pending_restart: '다시 시작 필요',
      retrying: '재시도 중',
      startup_failed: '시작 실패'
    },
    statusFilter: {
      all: '전체',
      bad: '오류',
      good: '연결됨',
      muted: '비활성',
      warn: '주의 필요'
    },
    telegramQr: {
      add: '추가',
      addAtLeastOne: 'Telegram 사용자 ID를 하나 이상 추가하세요.',
      allowedUsers: '허용된 사용자',
      applying: '저장 중...',
      createWithQr: 'QR 코드로 생성',
      expired: '만료됨',
      numericOnly: '허용된 Telegram 사용자 ID는 숫자여야 합니다.',
      openTelegram: 'Telegram 열기',
      ownerDetected: '소유자 감지됨',
      pairingExpired: 'Telegram 페어링이 만료되었습니다. 다시 시도하려면 새 QR 설정을 시작하세요.',
      quickHelp:
        'QR 코드를 스캔하고 Telegram에서 확인하세요. Hermes가 봇을 생성하고 Telegram 사용자 ID를 자동으로 감지합니다.',
      quickSetup: '빠른 설정',
      ready: '봇 생성됨',
      recommended: '추천',
      replaceWarning:
        'Telegram 자격 증명이 이미 구성되어 있습니다. 저장하면 새 QR 설정 또는 봇 토큰이 현재 봇을 대체합니다.',
      saveAndRestart: '저장 후 재시작',
      savedRestarting: 'Telegram이 저장되었습니다. 게이트웨이를 재시작하는 중...',
      scanHint: '휴대폰의 Telegram 앱으로 스캔하거나 이 컴퓨터에서 링크를 여세요.',
      starting: '시작하는 중...',
      subtitle: '두 옵션 모두 사용자가 제어하는 봇을 연결하며 자격 증명은 이 Hermes 설치에만 저장됩니다.',
      title: 'Telegram 봇 연결 방식 선택',
      userIdPlaceholder: 'Telegram 사용자 ID',
      waiting: 'Telegram 대기 중...'
    },
    unknown: '알 수 없음',
    unsavedChanges: '저장되지 않은 변경사항',
    addListEntry: '추가',
    removeListEntry: '삭제',
    listEntryPlaceholder: 'ID 입력'
  },
  notifications: {
    actions: {
      openGateways: '게이트웨이 열기',
      openKeys: '키 열기',
      openMaintenance: '유지보수 열기',
      restartHermes: 'Hermes 재시작'
    },
    backendOutOfDateMessage:
      'Hermes 백엔드가 이 데스크톱 빌드보다 오래되어 정상적으로 작동하지 않을 수 있습니다. 버전을 맞추려면 업데이트하세요.',
    backendOutOfDateTitle: '백엔드 버전이 오래되었습니다',
    clearAll: '모두 지우기',
    compressDeferredDone: '컨텍스트 압축 완료됨',
    copyDetail: '세부정보 복사',
    copyDetailFailed: '알림 세부정보를 복사할 수 없습니다',
    desktopOutOfDateMessage:
      '이 Hermes 앱이 연결된 백엔드보다 오래되어 정상적으로 작동하지 않을 수 있습니다. 버전을 맞추려면 앱을 업데이트하세요.',
    desktopOutOfDateTitle: 'Hermes 앱 버전이 오래되었습니다',
    details: '세부정보',
    dismiss: '알림 무시',
    errors: {
      codeSkewRestartRequired:
        'Hermes가 업데이트되었지만 여전히 이전 버전으로 실행 중입니다. 업데이트를 완료하려면 재시작하세요.',
      diskFull: '디스크 용량 부족 — 공간을 확보한 후 다시 시도하세요.',
      elevenLabsNeedsKey: '음성 입력에는 ElevenLabs 키가 필요합니다. 설정 → 키에서 키를 추가하세요.',
      elevenLabsRejectedKey: 'ElevenLabs에서 API 키를 거부했습니다. 설정 → 키에서 업데이트한 후 다시 시도하세요.',
      gatewayAuthFailed:
        '이 Hermes에서 저장된 로그인을 더 이상 허용하지 않습니다. 게이트웨이를 열고 다시 로그인하거나(또는 새 액세스 토큰 붙여넣기) 다시 시도하세요.',
      methodNotAllowed:
        '업데이트 직후 등으로 인해 Hermes의 백그라운드 서비스가 앱과 동기화되지 않았습니다. 문제를 해결하려면 재시작하세요.',
      microphonePermission: '마이크 권한이 거부되었습니다.',
      openaiRejectedApiKey: 'OpenAI에서 API 키를 거부했습니다. 설정 → 키에서 업데이트한 후 다시 시도하세요.',
      openaiTtsNeedsKey: '음성 기능에는 OpenAI 키가 필요합니다. 설정 → 키에서 추가하세요.',
      restartHermesFailed: 'Hermes를 재시작할 수 없습니다',
      rpcOutOfSync: '앱과 백엔드의 버전이 서로 다릅니다. 둘 다 업데이트하세요.',
      storageFailure: 'Hermes가 데이터 폴더에 저장할 수 없습니다. 유지보수를 열어 확인하고 복구하세요.'
    },
    hide: '숨기기',
    installMethodUnsupportedTitle: '지원되지 않는 설치 방식',
    mcp: {
      disable: '비활성화',
      errorTitle: 'MCP 서버에 연결할 수 없습니다',
      needsAuthTitle: 'MCP 서버 재인증이 필요합니다',
      signIn: '로그인',
      view: '보기'
    },
    native: {
      approvalTitle: '승인 필요',
      approveAction: '승인',
      backgroundDoneTitle: '백그라운드 작업 완료됨',
      backgroundFailedTitle: '백그라운드 작업 실패함',
      creditsTitle: '크레딧',
      inputBody: 'Hermes가 응답을 기다리고 있습니다.',
      inputTitle: '입력 필요',
      rejectAction: '거부',
      turnDoneBody: '',
      turnDoneTitle: 'Hermes 작업 완료됨',
      turnErrorTitle: '작업 실패'
    },
    region: '알림',
    seeWhatsNew: '새로운 기능 보기',
    sharedProfileWarning:
      '다른 Hermes 설치가 이 프로필을 사용하고 있습니다. 두 설치가 설정과 데이터를 공유하므로 변경 시 충돌이 발생할 수 있습니다. 그대로 계속하거나, 변경하기 전에 다른 설치를 닫을 수 있습니다.',
    show: '표시',
    updateDesktopApp: '앱 업데이트',
    updateHermes: 'Hermes 업데이트',
    updateReadyMessageAppInstaller:
      '새 버전의 Hermes가 준비되었습니다. 지금 업데이트하면 Windows에서 설치를 완료해 줍니다.',
    updateReadyMessageUnknown: '새로운 업데이트가 있습니다.',
    updateReadyTitle: '업데이트 준비됨',
    voice: {
      configureSpeechToText: '음성 모드를 사용하려면 음성-텍스트 변환을 구성하세요.',
      couldNotStartSession: '음성 세션을 시작할 수 없습니다',
      liveDelegationFailed: '요청을 Hermes에 전달할 수 없습니다',
      liveEnded: '라이브 음성 세션이 종료되었습니다',
      liveEndedClosed: '서비스에 의해 라이브 음성 세션이 종료되었습니다.',
      liveEndedConnectionLost: '라이브 음성 세션 연결이 끊어졌습니다.',
      liveError: '라이브 음성',
      microphoneAccessDenied: '마이크 접근이 거부되었습니다.',
      microphoneConstraintsUnsupported: '이 장치에서는 마이크 제약 조건을 지원하지 않습니다.',
      microphoneFailed: '마이크 오류 발생',
      microphoneInUse: '다른 앱에서 마이크를 이미 사용 중입니다.',
      microphonePermissionDenied: '마이크 권한이 거부되었습니다.',
      microphoneStartFailed: '마이크 녹음을 시작할 수 없습니다.',
      microphoneUnsupported: '이 런타임에서는 마이크 녹음을 지원하지 않습니다.',
      noMicrophone: '마이크를 찾을 수 없습니다.',
      noSpeechDetected: '음성이 감지되지 않았습니다',
      playbackFailed: '음성 재생 실패',
      recordingFailed: '음성 녹음 실패',
      transcriptionFailed: '음성 변환 실패',
      transcriptionUnavailable: '아직 음성 변환을 사용할 수 없습니다.',
      tryRecordingAgain: '다시 녹음해 보세요.',
      unavailable: '음성을 사용할 수 없음'
    }
  },
  profiles: {
    actions: '작업',
    allProfiles: '모든 프로필',
    autoColor: '자동',
    cloneFrom: '다음에서 복제',
    cloneFromDefault: '기본 프로필에서 복제',
    cloneFromDefaultDesc: '기본 프로필의 설정, 스킬, SOUL.md를 복사합니다.',
    cloneFromDesc: '선택한 원본 프로필의 설정, 스킬, SOUL.md를 복사합니다.',
    cloneFromNone: '없음 (빈 상태)',
    close: '프로필 닫기',
    color: '색상…',
    colorFor: '색상',
    connectGateway: '게이트웨이 관리…',
    copySetup: '설정 복사',
    copying: '복사 중...',
    createAction: '프로필 만들기',
    createDesc: '프로필은 독립된 Hermes 환경으로, 각각 별도의 설정, 스킬, SOUL.md를 가집니다.',
    created: '프로필이 만들어졌습니다',
    creating: '만드는 중...',
    default: '기본',
    defaultBadge: '기본값',
    defaultDescription: 'Hermes가 열릴 때와 새 채팅에 사용됩니다. 기존 세션은 각자의 프로필에 유지됩니다.',
    defaultProfile: '기본 프로필',
    deleteDescMid: ' 및 해당 ',
    deleteDescPrefix: '이 작업은 ',
    deleteDescSuffix: ' 디렉터리를 삭제합니다. 이 작업은 취소할 수 없습니다.',
    deleteTitle: '프로필을 삭제하시겠습니까?',
    deleted: '프로필이 삭제되었습니다',
    deleting: '삭제 중...',
    displayNameDesc: '앱 전체에 표시되는 이름을 설정합니다. 내부 프로필 ID는 "default"로 유지됩니다.',
    displayNameLabel: '표시 이름',
    displayNameTitle: '에이전트 이름 지정',
    editSoul: 'SOUL.md 편집…',
    emptySoul: '빈 SOUL.md — 페르소나 작성을 시작하세요...',
    env: '환경 변수',
    exportMenu: '내보내기…',
    exportProfile: '프로필 내보내기…',
    exported: '프로필을 내보냈습니다',
    failedCopy: '설정 명령을 복사하지 못했습니다',
    failedCreate: '프로필을 만들지 못했습니다',
    failedDelete: '프로필을 삭제하지 못했습니다',
    failedExport: '프로필을 내보내지 못했습니다',
    failedImport: '프로필을 가져오지 못했습니다',
    failedLoad: '프로필을 불러오지 못했습니다',
    failedLoadSoul: 'SOUL.md를 불러오지 못했습니다',
    failedRename: '프로필 이름을 변경하지 못했습니다',
    failedSaveSoul: 'SOUL.md를 저장하지 못했습니다',
    failedSetDefault: '기본 프로필로 설정할 수 없습니다',
    fleet: {
      allOnGateway: '이 게이트웨이의 모든 프로필',
      connectExistingInstead: '대신 기존 항목에 연결',
      installDeviceConfirm: '로컬에 설치',
      installDeviceDesc:
        '이 컴퓨터에 Hermes를 로컬로 설치한 후 새 세션을 엽니다. 확인하기 전에는 아무것도 설치되지 않습니다.',
      installDeviceTitle: '이 기기로 전환하시겠습니까?',
      localDevice: '이 기기 (로컬 백엔드 — Hermes가 없으면 설치하고, 그렇지 않으면 새 세션을 엽니다)',
      switchDeviceConfirm: '전환',
      switchDeviceDesc: '이 컴퓨터에서 새 세션을 엽니다. 현재 대화는 다른 게이트웨이에 유지됩니다.',
      switchDeviceTitle: '이 기기로 전환하시겠습니까?'
    },
    importProfile: '프로필 가져오기…',
    imported: '프로필을 가져왔습니다',
    loading: '프로필 불러오는 중...',
    loadingSoul: 'SOUL.md 불러오는 중...',
    manageProfiles: '프로필 관리…',
    modelLabel: '모델',
    nameHint: '소문자, 숫자, 하이픈, 언더바를 사용할 수 있습니다. 영문자나 숫자로 시작해야 합니다.',
    nameLabel: '이름',
    nameRequired: '이름은 필수입니다.',
    newNameLabel: '새 이름',
    newProfile: '새 프로필',
    noProfiles: '프로필이 없습니다.',
    notSet: '설정되지 않음',
    openInNewWindow: '새 창에서 열기',
    refresh: '프로필 새로고침',
    refreshing: '프로필 새로고침 중',
    remoteOverride: {
      authFailedTitle: '원격 호스트가 저장된 토큰을 거부했습니다',
      confirmBack: '뒤로',
      confirmTitle: '이 프로필을 원격 호스트에 연결하시겠습니까?',
      connect: '연결',
      connecting: '연결 중…',
      description: '이 프로필의 세션은 이 컴퓨터 대신 지정한 원격 Hermes에서 실행됩니다.',
      disconnect: '원격 연결 제거',
      menuItem: '원격 호스트에 연결…',
      plainTextOptIn:
        '이 컴퓨터에는 안전한 키 저장소가 없어 토큰이 디스크에 암호화되지 않은 채로 저장됩니다. 그래도 저장하시겠습니까?',
      removeFailed: '원격 연결을 제거할 수 없습니다',
      removedTitle: '원격 연결이 제거되었습니다',
      savedTitle: '프로필이 연결되었습니다',
      tokenLabel: '액세스 토큰',
      tokenPlaceholder: '원격 세션 토큰을 붙여넣으세요',
      tokenSavedHint: '토큰이 이미 저장되어 있습니다. 유지하려면 비워두세요.',
      updateToken: '새 토큰 입력…',
      urlInvalid: 'http:// 또는 https://로 시작하는 전체 주소를 입력하세요',
      urlLabel: '원격 주소',
      urlPlaceholder: 'https://hermes.example.com'
    },
    rename: '이름 변경',
    renameDescPrefix: '이름을 변경하면 프로필 디렉터리와 ',
    renameDescSuffix: '의 래퍼 스크립트가 업데이트됩니다.',
    renameMenu: '이름 변경…',
    renameTitle: '프로필 이름 변경',
    renamed: '프로필 이름이 변경되었습니다',
    renaming: '이름 변경 중...',
    saveSoul: 'SOUL.md 저장',
    saving: '저장 중...',
    search: '프로필 검색...',
    selectPrompt: '세부 정보를 보려면 프로필을 선택하세요.',
    setAsDefault: '기본값으로 설정',
    setupCopied: '설정 명령이 복사되었습니다',
    showAllProfiles: '모든 프로필 보기',
    skillsLabel: '스킬',
    soulDesc: '이 프로필에 반영된 시스템 프롬프트 및 페르소나 지침입니다.',
    soulMissing:
      '이 프로필에 아직 SOUL.md 파일이 없습니다. 아래에 지침을 추가하고 저장하여 파일을 만드세요. config.yaml의 성격 프리셋은 별도로 관리됩니다.',
    soulOptional: '선택 사항',
    soulPlaceholderCloned: '복제됨',
    soulPlaceholderEmpty: '비어 있음',
    soulSaved: 'SOUL.md가 저장되었습니다',
    title: '프로필',
    unsavedChanges: '저장되지 않은 변경 사항'
  },
  sendDiagnostics: {
    cancel: '취소',
    close: '닫기',
    copyLink: '링크 복사',
    doneDescription:
      '번들이 비공개로 업로드되었습니다. 지원 스레드에서 아래 링크를 공유하여 팀이 로그를 확인할 수 있도록 하세요.',
    doneTitle: '진단 정보 전송됨',
    failedHint:
      '터미널에서 `hermes debug share --nous`를 실행하거나, 업로드하지 않고 리포트를 출력하려면 `hermes debug share --local`을 실행할 수도 있습니다.',
    failedTitle: '업로드 실패',
    handoffLead: '대화를 이어갈 곳:',
    links: {
      discord: 'Discord',
      github: 'GitHub Issues',
      portal: 'Nous Portal Support'
    },
    privacyNotice:
      '디버그 번들을 Nous 내부 스토리지에 업로드합니다(공개 페이스트 아님). 시스템 정보(OS, 버전, 제공자, 구성된 API 키 — 키 자체는 제외)와 대화 내용, 도구 출력, 파일 경로가 포함될 수 있는 전체 에이전트, 게이트웨이, 데스크톱 로그(각 최대 512 KB)가 포함됩니다. 업로드 전에 비밀 정보는 마스킹 처리됩니다. 이 번들은 Nous 직원 및 허용된 Discord 관리자만 볼 수 있으며, 14일 후에 자동 삭제됩니다.',
    title: 'Nous에 진단 정보 전송',
    upload: '업로드',
    uploading: '업로드 중…'
  },
  sessionImport: {
    action: '세션 가져오기',
    all: '전체',
    choose: '계속 이어갈 가치가 있는 대화',
    chooseHelp: '세션을 선택하여 기록을 읽은 후 Hermes로 가져오세요.',
    connectedComputer: '연결된 컴퓨터',
    continue: 'Hermes에서 계속하기',
    copyNotice: '대화 텍스트를 복사합니다. 원본 파일은 변경되지 않습니다. 도구 출력 및 추론 내용은 가져오지 않습니다.',
    destination: '가져올 위치',
    empty: '대화를 찾을 수 없음',
    emptyHelp: '이 백엔드의 Claude Code 및 Codex 세션이 여기에 표시됩니다.',
    importError: '이 대화를 가져올 수 없습니다.',
    importing: '가져오는 중…',
    messages: '메시지',
    more: '세션 더 불러오기',
    noMatches: '일치하는 대화 없음',
    open: 'Hermes에서 열기',
    previewError: '미리보기를 사용할 수 없음',
    previewHelp: '원본이 이동되었거나 변경되었을 수 있습니다. 목록을 새로고침하고 다시 시도하세요.',
    previewLimit: '가독성을 위해 미리보기가 단축되었습니다. 전체 대화가 가져와집니다.',
    previewLoading: '미리보기 여는 중',
    readingFrom: '읽어오는 위치:',
    scanError: '세션을 찾을 수 없음',
    scanHelp: '백엔드 연결을 확인한 후 다시 시도하세요. 이전 버전의 백엔드는 업데이트가 필요할 수 있습니다.',
    scanning: '대화 찾는 중',
    search: '불러온 세션 검색',
    searchHelp: '다른 제목이나 폴더를 시도하거나 세션을 더 불러오세요.',
    skipped: '일부 로그가 비어 있거나, 읽을 수 없거나, 미리보기에는 너무 큽니다.',
    snapshot: '이 대화는 이미 Hermes에 있습니다. 계속하려면 기존 사본을 여세요.',
    subtitle: '대화를 Hermes로 가져와 중단했던 부분부터 이어서 하세요.',
    title: '다른 앱에서 이어서 하기',
    you: '사용자'
  },
  webhooks: {
    all: '(모두)',
    copy: '복사',
    create: '만들기',
    created: '생성됨',
    createdSecretHint: '시크릿을 지금 복사하세요. 한 번만 표시됩니다.',
    createdTitle: '구독 생성됨',
    creating: '생성 중...',
    delete: '삭제',
    deleteDescPrefix: '이 작업은 영구적으로 제거합니다: ',
    deleteDescSuffix: '. 이 작업은 취소할 수 없습니다.',
    deleteTitle: '웹훅 삭제',
    deleted: '웹훅이 삭제되었습니다',
    deleting: '삭제 중...',
    deliverOnly: '전달만',
    deliverOptions: {
      discord: 'Discord',
      email: '이메일',
      github_comment: 'GitHub 댓글',
      log: '로그',
      slack: 'Slack',
      telegram: 'Telegram'
    },
    disableRow: '사용 안 함',
    disabledBody:
      '웹훅은 자체 게이트웨이 플랫폼입니다. 들어오는 HTTP 이벤트를 수신하려면 여기서 활성화하세요. 채팅 채널은 구독이 텔레그램, 디스코드, 슬랙 또는 다른 채널로 전달될 때만 필요합니다.',
    disabledTitle: '웹훅 수신기 비활성화됨',
    done: '완료',
    empty: '아직 웹훅 구독이 없습니다.',
    enable: '웹훅 사용',
    enableRow: '사용',
    enabledRestarting: '웹훅이 활성화되었습니다. 게이트웨이를 재시작하는 중...',
    enabling: '활성화 중...',
    fieldDeliver: '전달 대상',
    fieldDeliverOnly: '페이로드만 전달',
    fieldDescription: '설명',
    fieldDescriptionPlaceholder: '이 웹훅이 하는 일 (선택 사항)',
    fieldEvents: '이벤트',
    fieldEventsPlaceholder: '쉼표로 구분, 모두 허용하려면 비워두세요',
    fieldName: '이름',
    fieldNamePlaceholder: '예: github-push',
    fieldPrompt: '프롬프트',
    fieldPromptPlaceholder: '이 웹훅이 실행될 때 에이전트에 전달할 지침 (선택 사항)',
    fieldSkills: '스킬',
    fieldSkillsPlaceholder: '쉼표로 구분된 스킬 이름 (선택 사항)',
    hint: '수신기가 실행되면 구독 변경 사항이 즉시 반영됩니다. 비활성화된 구독은 들어오는 이벤트를 거부합니다.',
    loadFailed: '웹훅을 불러오지 못했습니다',
    loading: '웹훅 불러오는 중...',
    nameRequired: '이름이 필요합니다',
    newSubscription: '새 구독',
    restartGateway: '게이트웨이 재시작',
    restartNeeded: '웹훅이 활성화되었지만, 수신기를 온라인 상태로 전환하려면 게이트웨이를 다시 시작해야 합니다.',
    restarting: '게이트웨이 재시작 중...',
    restartingGateway: '재시작 중...',
    search: '웹훅 검색...',
    secretOnce: '시크릿 (한 번만 표시됨)',
    webhookUrl: '웹훅 URL'
  }
} satisfies Pick<
  TranslationOverrides,
  | 'commandCenter'
  | 'connectors'
  | 'connectorsPage'
  | 'cron'
  | 'messaging'
  | 'notifications'
  | 'profiles'
  | 'sendDiagnostics'
  | 'sessionImport'
  | 'webhooks'
>
