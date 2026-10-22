/** Bulk machine-translated ko overlay — capabilities keys (Gemini 3.5 Flash-Lite, then spot-checked). */
import type { TranslationOverrides } from './define-locale'

export const koCapabilitiesOverrides = {
  agents: {
    ageNow: '지금',
    close: '에이전트 닫기',
    done: '완료',
    emptyDesc: '턴에서 작업을 위임하면 하위 에이전트의 진행 상황이 여기에 스트리밍됩니다.',
    emptyTitle: '실행 중인 하위 에이전트 없음',
    extendedTranscript: '확장된 대화 기록',
    failed: '실패',
    files: '파일',
    queued: '대기 중',
    requestRejected: '하위 에이전트가 요청을 수락하지 않았습니다',
    running: '실행 중',
    steer: '방향 조정',
    steerPlaceholder: '이 하위 에이전트를 위한 지침',
    steerQueued: '다음 체크포인트를 위해 대기 중',
    stopRequested: '중지 요청됨',
    streaming: '스트리밍 중',
    subtitle: '현재 턴의 실시간 하위 에이전트 활동입니다.',
    title: '생성 트리',
    transcriptTruncated: '최근 16 KiB 표시 중',
    transcriptUnavailable: '실시간 대화 기록을 사용할 수 없습니다',
    waitingActivity: '활동 대기 중'
  },
  artifactCard: {
    kind: {
      code: '코드',
      html: '인터랙티브 페이지',
      svg: '그래픽'
    },
    open: '열기'
  },
  artifactPreview: {
    copyContent: '콘텐츠 복사',
    download: '다운로드',
    latest: '최신',
    missingBody: '이 아티팩트가 더 이상 로컬 레지스트리에 없습니다.',
    missingTitle: '아티팩트를 사용할 수 없음',
    newerVersion: '최신 버전',
    olderVersion: '이전 버전',
    openInBrowser: '브라우저에서 열기',
    openInBrowserFailed: '브라우저에서 열지 못했습니다'
  },
  artifacts: {
    chat: '채팅',
    colLocationDefault: '위치',
    colLocationFile: '경로',
    colLocationLink: 'URL',
    colSession: '세션',
    colTitleDefault: '제목 / 이름',
    colTitleFile: '이름',
    colTitleLink: '링크 제목',
    copyPath: '경로 복사',
    copyUrl: 'URL 복사',
    failedLoad: '아티팩트를 로드하지 못했습니다',
    indexing: '최근 세션 아티팩트 색인 생성 중',
    itemsFile: '파일',
    itemsGeneric: '항목',
    itemsImage: '이미지',
    itemsLink: '링크',
    kindFile: '파일',
    kindImage: '이미지',
    kindLink: '링크',
    noArtifactsDesc: '세션에서 생성된 이미지와 파일 출력이 여기에 표시됩니다.',
    noArtifactsTitle: '아티팩트를 찾을 수 없음',
    openFailed: '열기 실패',
    refresh: '아티팩트 새로 고침',
    refreshing: '아티팩트 새로 고치는 중',
    search: '아티팩트 검색...',
    tabAll: '전체',
    tabFiles: '파일',
    tabImages: '이미지',
    tabLinks: '링크',
    zero: '0'
  },
  assistant: {
    approval: {
      allowSession: '이 세션 허용',
      alwaysAllow: '항상 허용',
      alwaysAllowMenu: '항상 허용…',
      alwaysTitle: '이 명령을 항상 허용하시겠습니까?',
      command: '명령',
      commandDetails: '명령 세부정보',
      gatewayDisconnected:
        '현재 Hermes가 오프라인 상태입니다. 명령이 답변을 기다리는 중입니다(승인 제한 시간까지). 다시 연결한 후 다시 전송하세요.',
      jumpToApproval: '승인 필요',
      moreOptions: '추가 승인 옵션',
      openSafetySettings: '안전 설정 열기',
      reconnect: '다시 연결',
      reject: '거부',
      run: '실행',
      sendFailed: '답변을 전송하지 못했습니다',
      timedOutSystemLine:
        '승인 시간이 초과되어 명령이 실행되지 않았습니다. Hermes에게 다시 시도를 요청하거나 설정 → 안전 → 승인 제한 시간을 늘리세요.'
    },
    catalogInstall: {
      advanced: '고급',
      commitLabel: '커밋',
      credentialsHeading: '자격 증명',
      phase: {
        downloading: '다운로드 중…',
        python_packages: 'Python 패키지를 설치하는 중…',
        loading_tools: '도구를 불러오는 중…'
      },
      notEnabled: '설치되었지만 꺼져 있습니다',
      alreadyInstalled: '이미 설치되어 있어 그대로 두었습니다',
      failed: '실패',
      hideNames: '이름 숨기기',
      install: '설치',
      installed: '설치됨',
      installing: '설치 중…',
      kind: {
        plugin: '플러그인',
        skill: '스킬'
      },
      notInstalled: '설치되지 않음',
      preparing: '설치 준비 중…',
      requirementsLabel: '필수 조건',
      scan: {
        failed: '검사 실패',
        passed: '검사 통과',
        warnings: '검사에서 경고 발견'
      },
      securityHeading: '보안',
      sendFailed: '답변을 전송하지 못했습니다. 다시 시도하세요.',
      showNames: '이름 표시',
      skip: '건너뛰기',
      subdirLabel: '폴더',
      tier: {
        community: '커뮤니티',
        official: '공식'
      }
    },
    clarify: {
      confirmAndContinueLabel: '확인 및 계속',
      gatewayDisconnected: '현재 Hermes가 오프라인 상태입니다. 다시 연결한 후 다시 전송하세요.',
      loadingQuestion: '질문 불러오는 중…',
      noAnswer: '답변 없음',
      notDelivered:
        "이 질문이 앱에 도달하지 않아 여기서 답변할 수 없습니다. '중지'를 눌러 턴을 종료한 후 채팅에서 답변하세요.",
      notReady: '명확화 요청이 아직 준비되지 않았습니다',
      other: '기타 (답변 직접 입력)',
      placeholder: '답변을 입력하세요…',
      sendFailed: '명확화 응답을 전송하지 못했습니다',
      skip: '건너뛰기',
      skipped: '건너뜀',
      multiSelectHint: '중복 선택 가능해요',
      singleSelectHint: '하나를 골라주세요'
    },
    mcpSetup: {
      authorizeAction: '승인',
      authorizeTitle: 'MCP 서버 승인',
      enableAction: '사용 설정',
      enableTitle: 'MCP 서버 사용 설정',
      envRequired: '먼저 필수 자격 증명을 입력하세요',
      gatewayDisconnected: '현재 Hermes가 오프라인 상태입니다. 다시 연결한 후 다시 전송하세요.',
      installAction: '설치',
      installTitle: 'MCP 서버 추가',
      reloadFailed: '서버는 저장되었지만 MCP 도구 리로드에 실패했습니다. 다음 세션에 로드됩니다.',
      sendFailed: 'MCP 설정 응답을 전송하지 못했습니다'
    },
    thread: {
      attachingFile: '첨부 중…',
      branchNewChat: '새 채팅에서 분기',
      copy: '복사',
      copyFullResponse: '전체 응답 복사',
      dismissError: '오류 해제',
      editMessage: '메시지 편집',
      errorChooseModel: '모델 선택',
      errorCodes: {
        SESSION_NOT_OWNED: {
          body: '이 채팅은 현재 다른 Hermes 창이나 터미널에서 열려 있습니다. 해당 위치에서 닫은 후 메시지를 다시 보내거나, 여기서 새 채팅을 시작하세요.',
          title: '이 채팅이 다른 곳에서 열려 있습니다'
        },
        billing: {
          title: '크레딧 소진'
        },
        content_policy_blocked: {
          title: 'AI 서비스에서 이 요청을 거부했습니다'
        },
        context_overflow: {
          body: '대화 내용이 모델 허용 범위를 초과했습니다. 압축하거나 새 채팅을 시작한 후 다시 전송하세요.',
          title: '대화가 너무 깁니다'
        },
        disk_full: {
          body: '디스크가 가득 차서 Hermes가 이 대화를 저장할 수 없습니다. 공간을 확보한 후 다시 시도하세요.',
          title: '디스크 용량 부족'
        },
        empty_response: {
          title: 'AI 서비스에서 빈 응답을 보냈습니다'
        },
        format_error: {
          title: 'AI 서비스가 요청을 거부했습니다'
        },
        free_tier_at_capacity: {
          body: '로그인하여 대기열을 건너뛰세요(무료). 또는 잠시 후 다시 시도하세요.',
          title: '현재 로그인하지 않은 상태의 채팅 사용량이 많습니다'
        },
        free_tier_disabled: {
          body: '계속 채팅하려면 Nous 계정으로 로그인하세요(무료).',
          title: '현재 로그인하지 않고 Hermes를 사용하는 기능이 꺼져 있습니다'
        },
        free_tier_model_not_free: {
          body: 'Hermes는 현재 무료 모델을 사용합니다. 더 많은 모델을 사용하려면 Nous 계정으로 로그인하세요(무료).',
          title: '로그인하지 않으면 해당 모델을 사용할 수 없습니다'
        },
        free_tier_outage: {
          body: '1분 후에 메시지를 다시 보내보세요.',
          title: '현재 무료 모델 응답에 문제가 발생했습니다'
        },
        free_tier_rate_limited: {
          body: '잠시 후 갱신됩니다. 더 많은 이용량을 원하시면 Nous 계정으로 로그인하세요(무료).',
          title: '로그인 없이 이용 가능한 채팅 한도를 모두 사용했습니다'
        },
        free_tier_refused: {
          body: 'Nous 계정 로그인은 무료입니다.',
          title: '로그인하지 않고는 Hermes가 해당 작업을 보낼 수 없습니다'
        },
        free_tier_route: {
          body: 'Nous 계정으로 로그인(무료)하거나 NOUS_INFERENCE_BASE_URL 설정을 확인하세요.',
          title: 'Hermes가 이 경로에서 무료 모델에 연결할 수 없습니다'
        },
        invalid_response: {
          title: 'AI 서비스에서 읽을 수 없는 응답을 보냈습니다'
        },
        loop_error: {
          body: '응답에서 동일한 단계가 반복되어 Hermes가 중지했습니다. 다시 시도하거나, 문제가 지속되면 새 채팅을 시작하세요.',
          title: 'Hermes가 무한 루프에 빠졌습니다'
        },
        model_not_found: {
          title: '이 모델을 사용할 수 없습니다'
        },
        no_reply: {
          body: 'Hermes가 답변 없이 이 턴을 종료했습니다. 다시 시도하여 전송하세요.',
          title: '답변이 완료되지 않았습니다'
        },
        overloaded: {
          title: 'AI 서비스가 과부하 상태입니다'
        },
        payload_too_large: {
          body: '요청 내용이 모델이 처리하기엔 너무 큽니다. 대화를 압축하거나 새 채팅을 시작한 후 다시 전송하세요.',
          title: '메시지가 너무 큽니다'
        },
        provider_policy_blocked: {
          title: '계정 설정에 의해 이 모델이 차단되었습니다'
        },
        rate_limit: {
          title: 'AI 서비스가 바쁩니다'
        },
        server_error: {
          title: 'AI 서비스에 문제가 발생했습니다'
        },
        ssl_cert_verification: {
          title: '보안 연결 실패'
        },
        stream_drop: {
          body: '답변이 완료되기 전에 연결이 끊어졌습니다. 다시 시도하여 전송하세요.',
          title: '답변이 잘렸습니다'
        },
        timeout: {
          title: 'AI 서비스에 연결할 수 없습니다'
        },
        truncated: {
          body: '완료되기 전에 모델이 중지되었습니다. 전체 답변을 받으려면 다시 시도하세요.',
          title: '답변이 도중에 끊겼습니다'
        },
        upstream_blocked: {
          title: '방화벽에 의해 요청이 차단되었습니다'
        },
        upstream_rate_limit: {
          title: 'AI 서비스가 바쁩니다'
        }
      },
      errorCompressConversation: '대화 압축',
      errorCompressFailed: '대화를 압축하지 못했습니다',
      errorCopyDiagnostics: '오류 세부정보 복사',
      errorDetails: '세부정보',
      errorGenericProvider: 'AI 서비스',
      errorLayerBodies: {
        auth: 'AI 서비스가 로그인을 거부했습니다. 이 제공자의 자격 증명을 확인한 후 메시지를 다시 보내세요.',
        billing: '이 제공자에 사용할 수 있는 계정 크레딧이 없습니다. 충전하거나 제공자를 변경한 후 다시 보내세요.',
        disk: '디스크가 가득 차서 Hermes가 이 대화를 저장할 수 없습니다. 공간을 확보한 후 다시 시도하세요.',
        endpoint:
          'Hermes가 커스텀 모델 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인한 후 메시지를 다시 보내세요.',
        gateway:
          '이 답변을 시작하는 중 Hermes에 내부 문제가 발생했습니다. 메시지를 다시 보내주세요. 문제가 계속되면 진단 정보를 보내주세요.',
        generic: 'Hermes가 응답하는 동안 문제가 발생했습니다. 다시 시도하거나, 문제가 계속되면 세부정보를 복사하세요.',
        provider: 'AI 서비스가 이 요청을 완료할 수 없습니다. 잠시 후 다시 시도하거나 제공자를 변경하세요.',
        runtime:
          '이 답변을 시작하는 중 Hermes에 내부 문제가 발생했습니다. 메시지를 다시 보내주세요. 문제가 계속되면 진단 정보를 보내주세요.',
        streaming: '답변이 완료되기 전에 연결이 끊어졌습니다. 다시 시도하여 전송하세요.'
      },
      errorLayers: {
        auth: '로그인 문제',
        billing: '크레딧 소진',
        disk: '디스크 용량 부족',
        endpoint: '모델 서버에 연결할 수 없음',
        gateway: 'Hermes에 문제가 발생했습니다',
        generic: 'Hermes가 이 답변을 완료하지 못했습니다',
        provider: 'AI 서비스에서 오류를 반환했습니다',
        runtime: 'Hermes에 문제가 발생했습니다',
        streaming: '답변이 잘렸습니다'
      },
      errorOpenDesktopLogs: '데스크톱 로그 열기',
      errorOpenHermesFolder: 'Hermes 폴더 열기',
      errorOpenHermesFolderFailed: 'Hermes 폴더를 열지 못했습니다',
      errorOpenLogs: '로그 열기',
      errorOpenLogsFailed: '로그 폴더를 열 수 없습니다',
      errorRetry: '재시도',
      errorRetryScheduledCancel: '취소',
      errorSendDiagnostics: '진단 정보 보내기',
      errorSignInFreeTier: 'Nous 계정으로 로그인',
      errorStartNewSession: '새 세션 시작',
      errorSwitchProvider: '공급자 전환',
      errorToastTitle: 'Hermes가 답변을 완료하지 못했습니다',
      errorUpdateApiKey: 'API 키 업데이트',
      expandMessage: '메시지 펼치기',
      goForward: '앞으로 가기',
      loadingResponse: 'Hermes가 응답을 불러오는 중입니다',
      loadingSession: '세션 불러오는 중',
      moreActions: '추가 작업',
      preparingAudio: '오디오 준비 중...',
      processingPrompt: '프롬프트 처리 중',
      react: '반응',
      readAloud: '소리 내어 읽기',
      readAloudFailed: '소리 내어 읽기 실패',
      readAloudFullResponseHint: 'Shift+클릭: 전체 응답 읽기',
      refresh: '새로고침',
      restoreBody: '이 프롬프트 이후의 모든 내용이 대화에서 제거되며, 이 지점부터 프롬프트가 다시 실행됩니다.',
      restoreCheckpoint: '체크포인트 복원',
      restoreConfirm: '복원 후 다시 실행',
      restoreFromHere: '체크포인트 복원 — 이 프롬프트부터 다시 실행',
      restoreNext: '다음 체크포인트 복원',
      restorePrevious: '이전 체크포인트 복원',
      restoreTitle: '이 체크포인트로 복원하시겠습니까?',
      reviewChanges: '검토',
      scrollToBottom: '맨 아래로 스크롤',
      sendEdited: '수정된 메시지 보내기',
      showEarlier: '이전 메시지 표시',
      stop: '중지',
      stopReading: '읽기 중지',
      thinking: '생각 중',
      thought: '생각',
      thoughtBriefly: '간단히 생각함',
      responseStopped: '응답이 중단되었어요'
    },
    tool: {
      actions: {
        failedToOpen: '열기 실패',
        opened: '열었습니다',
        opening: '여는 중',
        ran: '실행됨',
        ranCode: '코드 실행됨',
        read: '읽음',
        reading: '읽는 중',
        running: '실행 중',
        runningCode: '스크립팅 중',
        searched: '검색함',
        searching: '검색 중'
      },
      copyActivity: '활동 복사',
      copyCode: '코드 복사',
      copyCommand: '명령어 복사',
      copyContent: '콘텐츠 복사',
      copyFile: '파일 복사',
      copyOutput: '출력 복사',
      copyPath: '경로 복사',
      copyQuery: '쿼리 복사',
      copyResults: '결과 복사',
      copyUrl: 'URL 복사',
      failedOne: '1단계 실패',
      memoryWriteNoted: '메모리 기록됨',
      outputAlt: '도구 출력',
      prefixes: {
        browser: '브라우저',
        web: '웹'
      },
      rawResponse: '원본 응답',
      recoveredOne: '1개 실패 단계 복구됨',
      renderingImage: '이미지 렌더링 중',
      resultInterrupted: '중단됨',
      resultUnavailable: '결과를 사용할 수 없음',
      skillActivity: {
        listFailed: '스킬 목록을 가져오지 못했습니다',
        listed: '스킬 목록 표시됨',
        listing: '스킬 목록 나열 중',
        loadFailed: '스킬을 불러오지 못했습니다',
        loaded: '스킬을 불러왔습니다',
        loading: '스킬 불러오는 중',
        readResource: '스킬 리소스 읽음',
        readingResource: '스킬 리소스 읽는 중',
        resourceFailed: '스킬 리소스를 읽지 못했습니다',
        unavailable: '스킬 결과를 사용할 수 없음'
      },
      statusDone: '완료',
      statusError: '오류',
      statusRecovered: '복구됨',
      statusRunning: '실행 중',
      titles: {
        browser_click: {
          done: '페이지 요소 클릭함',
          pending: '페이지 요소 클릭 중',
          pendingAction: '클릭 중'
        },
        browser_fill: {
          done: '양식 필드 채움',
          pending: '양식 필드 채우는 중',
          pendingAction: '채우는 중'
        },
        browser_navigate: {
          done: '페이지 열음',
          pending: '페이지 여는 중',
          pendingAction: '여는 중'
        },
        browser_snapshot: {
          done: '페이지 스냅샷 캡처함',
          pending: '페이지 스냅샷 캡처 중',
          pendingAction: '캡처 중'
        },
        browser_take_screenshot: {
          done: '스크린샷 캡처함',
          pending: '스크린샷 캡처 중',
          pendingAction: '캡처 중'
        },
        browser_type: {
          done: '페이지에 입력함',
          pending: '페이지에 입력 중',
          pendingAction: '입력 중'
        },
        clarify: {
          done: '질문함',
          pending: '질문 중',
          pendingAction: '질문 중'
        },
        cronjob: {
          done: 'Cron 작업',
          pending: 'Cron 작업 예약 중',
          pendingAction: '예약 중'
        },
        edit_file: {
          done: '파일 편집됨',
          pending: '파일 편집 중',
          pendingAction: '편집 중'
        },
        execute_code: {
          done: '코드 실행됨',
          pending: '스크립팅 중',
          pendingAction: '스크립팅 중'
        },
        image_generate: {
          done: '이미지 생성됨',
          pending: '이미지 생성 중',
          pendingAction: '생성 중'
        },
        list_files: {
          done: '파일 목록 표시됨',
          pending: '파일 목록 나열 중',
          pendingAction: '나열 중'
        },
        memory: {
          done: '메모리에 저장됨',
          pending: '메모리에 저장 중',
          pendingAction: '저장 중'
        },
        patch: {
          done: '파일 패치됨',
          pending: '파일 패치 중',
          pendingAction: '패치 중'
        },
        read_file: {
          done: '파일 읽음',
          pending: '파일 읽는 중',
          pendingAction: '읽는 중'
        },
        search_files: {
          done: '파일 검색함',
          pending: '파일 검색 중',
          pendingAction: '검색 중'
        },
        session_search_recall: {
          done: '세션 기록 검색함',
          pending: '세션 기록 검색 중',
          pendingAction: '검색 중'
        },
        terminal: {
          done: '명령어 실행됨',
          pending: '명령어 실행 중',
          pendingAction: '실행 중'
        },
        todo: {
          done: '할 일 업데이트됨',
          pending: '할 일 업데이트 중',
          pendingAction: '업데이트 중'
        },
        vision_analyze: {
          done: '이미지 분석함',
          pending: '이미지 분석 중',
          pendingAction: '분석 중'
        },
        web_extract: {
          done: '웹페이지 읽음',
          pending: '웹페이지 읽는 중',
          pendingAction: '읽는 중'
        },
        web_search: {
          done: '웹 검색함',
          pending: '웹 검색 중',
          pendingAction: '검색 중'
        },
        write_file: {
          done: '파일 편집됨',
          pending: '파일 편집 중',
          pendingAction: '편집 중'
        }
      }
    }
  },
  skills: {
    all: '모두',
    archive: '아카이브',
    bulkNoChange: '변경할 내용이 없습니다.',
    changesApplyNewSessions: '변경사항은 새 세션에 적용됩니다.',
    configured: '구성됨',
    configuringProfile: '구성 중:',
    disableAll: '모두 비활성화',
    disableUnused: '사용하지 않는 항목 비활성화',
    edit: '편집',
    enableAll: '모두 활성화',
    hub: {
      actionFailed: '스킬 작업 실패',
      actionLog: '작업 로그',
      close: '닫기',
      connectedHubs: '연결된 허브:',
      connectingHubs: '스킬 허브에 연결하는 중...',
      featured: '추천 스킬',
      files: '파일',
      install: '설치',
      installed: '설치됨',
      installing: '설치 중...',
      landingHint: '공식 인덱스, GitHub 및 커뮤니티 소스에서 설치 가능한 스킬을 찾아보려면 허브를 검색하세요.',
      loadFailed: '스킬 허브를 불러오지 못했습니다.',
      noFindings: '보안 검사 결과가 없습니다.',
      noReadme: '이 스킬에는 SKILL.md 미리보기가 없습니다.',
      noResults: '허브에서 일치하는 스킬을 찾지 못했습니다.',
      openLog: '로그 열기',
      pickerBrowse: '전체 허브 둘러보기',
      pickerHide: '허브 브라우저 숨기기',
      pickerHint: '아무 스킬에서나 "+ 에이전트에 추가"를 누르면 설치되어 위 목록에 나타납니다.',
      pickerTitle: '스킬 허브',
      policyAllow: '설치 허용됨',
      policyAsk: '설치 전 검토',
      policyBlock: '정책에 의해 설치 차단됨',
      preview: '미리보기',
      previewFailed: '스킬 미리보기 실패',
      scan: '검사',
      scanFailed: '보안 검사 실패',
      scanning: '검사 중...',
      search: '검색',
      searchFailed: '허브 검색 실패',
      searchPlaceholder: '스킬 허브 검색',
      searching: '검색 중...',
      trust: {
        builtin: '기본 제공',
        community: '커뮤니티',
        trusted: '신뢰됨'
      },
      uninstall: '제거',
      uninstalling: '제거 중...',
      updateAll: '설치된 항목 업데이트',
      updateStarted: '설치된 스킬을 업데이트하는 중...',
      updating: '업데이트 중...',
      verdictCaution: '주의',
      verdictDangerous: '위험',
      verdictSafe: '안전',
      viewScan: '검사 결과 보기'
    },
    loading: '기능 로드 중...',
    needsKeys: '키 필요',
    noDescription: '설명이 없습니다.',
    noSkillsDesc: '더 넓은 검색어나 다른 카테고리를 시도해 보세요.',
    noSkillsTitle: '스킬을 찾을 수 없습니다',
    noToolsetsDesc: '더 넓은 검색어를 입력해 보세요.',
    noToolsetsTitle: '도구 셋을 찾을 수 없습니다',
    officialCatalog: '설치 가능',
    officialPill: '공식',
    plugins: {
      agentBlurb: '선택한 프로필용 에이전트(도구, 훅, 제공자)를 확장합니다. 게이트웨이를 재시작한 후 적용됩니다.',
      agentTitle: '에이전트 플러그인',
      catalogBrowse: '찾아보기',
      catalogHide: '카탈로그 브라우저 숨기기',
      catalogHint:
        '아무 플러그인에서나 "+ 에이전트에 추가"를 누르세요. 검토된 항목은 선택한 프로필의 고정된 커밋에 설치됩니다. 에이전트 및 데스크톱 통합 플러그인은 양쪽 모두 제공합니다.',
      catalogTitle: '플러그인 카탈로그',
      deepLinkCatalogInvalidName: '링크의 카탈로그 이름이 없거나 잘못되었습니다.',
      deepLinkCatalogUnavailable:
        'Hermes 플러그인 카탈로그를 로드할 수 없습니다. 연결 상태를 확인하고 링크를 다시 여세요.',
      deepLinkErrorTitle: '플러그인 설치 링크가 거부됨',
      defaultProfile: 'Hermes (기본값)',
      desktopHalfPending: '복사 중...',
      desktopHalfPendingTip:
        '이 패키지에는 아직 앱으로 복사되지 않은 데스크톱 파트가 포함되어 있습니다. 다시 스캔하거나 앱을 재시작하세요.',
      desktopHalfRemote: '사용 불가 (원격 백엔드)',
      desktopHalfRemoteTip:
        '이 패키지의 데스크톱 파트는 이 앱에서 읽을 수 없는 원격 백엔드 디스크에 있습니다. 여기서 사용하려면 패키지의 저장소 URL과 데스크톱 대상을 선택한 상태로 Git에서 설치를 실행하여 이 기기에 데스크톱 파트를 복제하세요.',
      empty: '이 프로필에 설치된 에이전트 플러그인이 없습니다.',
      emptyAll: '플러그인이 없습니다.',
      emptyHint: '아래 카탈로그를 둘러보고 클릭 한 번으로 검토된 플러그인을 설치하세요.',
      halfAgent: '에이전트',
      halfDesktop: '데스크톱',
      halfDesktopHint: '이 앱, 모든 프로필에 공통 적용',
      installAgentHere: '여기에 설치',
      installAgentHereNoOrigin:
        '이 프로필에 에이전트 파트가 설치되어 있지 않으며, 이 패키지는 수동으로 복사되었으므로(카탈로그 항목이나 Git 원격 없음) 여기서 설치할 수 없습니다. 폴더를 프로필에 복사하거나 Git에서 다시 설치하세요.',
      kindAgent: '에이전트',
      kindBoth: '에이전트 + 데스크톱',
      kindDesktop: '데스크톱',
      legacyBackend:
        '이 백엔드는 키 기반 플러그인 토글이 도입되기 이전 버전입니다. 여기서 관리하려면 Hermes를 업데이트하세요.',
      loadFailed: '에이전트 플러그인을 로드할 수 없습니다',
      pageBlurb: '플러그인은 이 앱, 에이전트 또는 둘 다를 확장할 수 있으며 각 파트마다 개별 스위치가 있습니다.',
      portableBadge: '포터블',
      serverStates: {
        app_not_running: '앱 실행 중 아님',
        connected: '연결됨',
        endpoint_unavailable: '엔드포인트 사용 불가',
        hermes_not_connected: 'MCP 연결 누락됨',
        missing_app: '앱 누락됨',
        no_interactive_session: '대화형 세션 없음',
        unknown: '상태 알 수 없음',
        version_too_old: '버전이 너무 오래됨',
        unsupported_gpu: '지원되지 않는 GPU'
      },
      settingsForm: {
        save: '설정 저장',
        required: '필수',
        secretSet: '•••••••• (설정됨)'
      },
      tierCommunity: '커뮤니티',
      tierOfficial: '공식',
      uninstall: '제거',
      updateConsentConfirm: '업데이트 적용'
    },
    provenance: {
      agent: '학습됨',
      bundled: '기본 제공',
      hub: '허브'
    },
    refresh: '스킬 새로고침',
    refreshing: '스킬 새로고침 중',
    searchSkills: '스킬 검색...',
    searchToolsets: '도구 검색...',
    skillArchivedMessage: 'hermes curator restore를 통해 복원할 수 있습니다.',
    skillArchivedTitle: '스킬 보관됨',
    skillDisabled: '스킬 비활성화됨',
    skillEnabled: '스킬 활성화됨',
    skillUpdated: '스킬 업데이트됨',
    skillsLoadFailed: '스킬을 로드하지 못했습니다',
    sortAlpha: '이름순',
    sortLeastUsedAsc: '↑ 적게 사용된 순',
    sortMostUsed: '많이 사용된 순',
    sortMostUsedDesc: '↓ 많이 사용된 순',
    tabPlugins: '플러그인',
    tabSkills: '스킬',
    tabToolsets: '도구',
    toolsetDisabled: '도구 셋 비활성화됨',
    toolsetEnabled: '도구 셋 활성화됨',
    toolsetsRefreshFailed: '도구 셋을 새로고침하지 못했습니다',
    visionModelHint:
      '비전은 보조 모델 설정을 사용합니다. 이미지 처리 가능 모델은 개별 제공자가 아닌 해당 설정에서 지정됩니다.',
    visionModelLink: '설정 → 모델에서 비전 모델 선택'
  },
  starmap: {
    close: '메모리 그래프 닫기',
    copied: '복사되었습니다!',
    copy: '맵 코드 복사',
    emptyDesc: 'Hermes가 작업에 대한 스킬과 메모리를 구축하면 여기에 나타납니다.',
    emptyTitle: '아직 학습된 내용이 없습니다',
    filterAll: '전체',
    filterLearned: '학습됨',
    filterUsed: '사용됨',
    importBtn: '로드',
    importEmpty: '로드할 맵 코드를 붙여넣으세요.',
    importMap: '맵 가져오기',
    importedBadge: '가져온 맵',
    loadFailed: '메모리 그래프를 로드할 수 없습니다',
    loading: '로드 중...',
    memory: '메모리',
    refresh: '새로고침',
    resetToMine: '내 맵으로 돌아가기',
    share: '맵 공유',
    shareHint:
      '코드를 복사하여 이 맵을 공유하거나, 코드를 붙여넣어 로드하세요. 메모리나 스킬 텍스트는 포함되지 않고 레이아웃만 포함됩니다.',
    sharePlaceholder: '맵 코드 붙여넣기...',
    shareTitle: '맵 가져오기 / 내보내기',
    title: '메모리 그래프',
    viewGraph: '그래프'
  },
  skillDeepLink: {
    destinationChanged: '대상이 변경되었어요. 이 대화상자를 닫고 설치 링크를 다시 열어주세요.',
    installDescription: '이 스킬은 새 세션에서 사용할 수 있어요. 신뢰할 수 있는 소스만 설치해 주세요.',
    installTo: '설치 위치',
    installed: '설치됨',
    installing: '설치 중…',
    source: '소스',
    thisComputer: '이 컴퓨터'
  }
} satisfies Pick<
  TranslationOverrides,
  'agents' | 'artifactCard' | 'artifactPreview' | 'artifacts' | 'assistant' | 'skills' | 'starmap' | 'skillDeepLink'
>
