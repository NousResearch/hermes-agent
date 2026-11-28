// ko/30.ts — Korean translation of the `desktop, tips, errors, ui` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko30: TranslationOverrides = {
  desktop: {
    audioReadFailed: '녹음된 오디오를 읽을 수 없어요',
    sessionUnavailable: '세션을 사용할 수 없어요',
    createSessionFailed: '새 세션을 만들 수 없어요',
    promptFailed: '프롬프트 실행에 실패했어요',
    staleSessionTitle: '대화가 최신 상태가 아니에요',
    staleSessionBody:
      '이 창은 같은 대화의 다른 보기보다 이전 상태였습니다. 최신 메시지를 불러왔어요. 계속 진행하려면 다시 전송해 주세요.',
    providerCredentialRequired: '첫 메시지를 보내기 전에 제공자 인증 정보를 추가해 주세요.',
    emptySlashCommand: '비어 있는 슬래시 명령어',
    desktopCommands: '데스크톱 명령어',
    skillCommandsAvailable: count => `사용 가능한 스킬 명령어 ${count}개.`,
    warningLine: message => `경고: ${message}`,
    yoloArmed: '이 대화에 YOLO 모드가 활성화되었어요',
    yoloOff: 'YOLO 꺼짐',
    yoloSystem: active => `이 세션에서 YOLO ${active ? '켜짐' : '꺼짐'}`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'YOLO 모드를 전환할 수 없어요',
    profileStatus: current =>
      `프로필: ${current}. 다른 프로필에서 대화를 시작하려면 /profile <이름>을 입력하거나 "새 세션" 선택 메뉴를 사용하세요.`,
    unknownProfile: '알 수 없는 프로필',
    noProfileNamed: (target, available) => `"${target}" 프로필을 찾을 수 없어요. 사용 가능: ${available}`,
    newChatsProfile: name => `새 대화에서 ${name} 프로필을 사용해요.`,
    setProfileFailed: '프로필 설정 실패',
    sttDisabled: '설정에서 음성 텍스트 변환(STT)이 비활성화되어 있어요.',
    stopFailed: '중지 실패',
    regenerateFailed: '재생성 실패',
    editFailed: '수정 실패',
    editTurnUnavailable: '이 대화 턴은 더 이상 서버 기록에 없어요(압축되었을 수 있어요).',
    resumeFailed: '재개 실패',
    readOnlyTranscriptTitle: '읽기 전용으로 열림',
    readOnlyTranscriptBody:
      '연결된 백엔드가 아직 이 이전 대화를 지원하지 않아 읽기 전용 대화록으로 열렸어요. 기록은 유지되지만 백엔드가 연결될 때까지 메시지 전송이 비활성화돼요.',
    readOnlyTranscriptSendBlocked: '이 대화는 읽기 전용 대화록으로 열려 있어 메시지를 보낼 수 없어요.',
    resumeStrandedTitle: '이 세션을 불러올 수 없어요',
    resumeStrandedBody:
      '세션 연결에 실패하여 자동 재시도가 중단되었어요. 게이트웨이가 실행 중인지 확인한 후 다시 시도해 주세요.',
    poolSlotTimeoutBody:
      '이 컴퓨터의 제한에 비해 너무 많은 봇이 동시에 실행 중이에요. 설정 → 고급에서 제한을 늘리거나, 실행 중인 작업이 끝날 때까지 기다린 후 다시 시도해 주세요.',
    poolSlotTimeoutOpenSettings: '고급 설정 열기',
    resumeRetry: '다시 시도',
    nothingToBranch: '브랜치를 생성할 대상이 없어요',
    branchNeedsChat: '브랜치를 생성하기 전에 대화를 시작하거나 재개해 주세요.',
    sessionBusy: '세션 사용 중',
    branchStopCurrent: '이 대화의 브랜치를 생성하기 전에 현재 턴을 중지해 주세요.',
    branchNoText: '이 메시지에는 브랜치를 만들 텍스트가 없어요.',
    branchTitle: n => `초안: 브랜치 #${n}`,
    branchFailed: '브랜치 생성 실패',
    deleteFailed: '삭제 실패',
    archived: '보관됨',
    archiveFailed: '보관 실패',
    restored: '복원됨',
    unarchiveFailed: '보관 해제 실패',
    cwdChangeFailed: '작업 디렉터리 변경 실패',
    cwdStagedTitle: '작업 디렉터리 적용 대기 중',
    cwdStagedMessage: '활성 세션에 작업 디렉터리 변경 사항을 적용하려면 데스크톱 백엔드를 다시 시작하세요.',
    modelSwitchConfirmBody: '이 모델 전환을 확인해야 해요.',
    modelSwitchConfirmLabel: '그래도 전환',
    modelSwitchConfirmTitle: (model: string) => `${model}(으)로 전환할까요?`,
    modelSwitchConfirmTitleFallback: '모델을 전환할까요?',
    modelSwitchFailed: '모델 전환 실패',
    modelSwitchKeepLabel: '현재 모델 유지',
    modelSwitchStaleNotice: '선택 항목이 변경되어 모델 전환이 적용되지 않았어요.',
    hydrationSyncing: (profile: string) => `${profile} 동기화 중\u2026`,
    sessionExported: '세션을 내보냈어요',
    sessionExportFailed: '세션을 내보낼 수 없어요',
    imageSaved: '이미지 저장됨',
    downloadStarted: '다운로드 시작됨',
    restartToUseSaveImage: '이미지 저장 기능을 사용하려면 Hermes Desktop을 다시 시작하세요.',
    restartToSaveImages: '이미지를 저장하려면 Hermes Desktop을 다시 시작하세요',
    imageDownloadFailed: '이미지 다운로드 실패',
    openImage: '이미지 열기',
    downloadImage: '이미지 다운로드',
    savingImage: '이미지 저장 중',
    imagePreviewFailed: '이미지 미리보기 실패',
    imageAttach: '이미지 첨부',
    imageWriteFailed: '이미지를 디스크에 쓰지 못했어요.',
    imageAttachFailed: '이미지 첨부 실패',
    pastedContent: '붙여넣은 내용',
    pasteAttachFailed: '붙여넣은 텍스트를 첨부할 수 없어요',
    attachImages: '이미지 첨부',
    clipboard: '클립보드',
    noClipboardImage: '클립보드에서 이미지를 찾을 수 없어요',
    clipboardPasteFailed: '클립보드 붙여넣기 실패',
    dropFiles: '파일을 여기에 놓으세요',
    handoff: {
      pickPlatform: '대상 플랫폼 선택',
      success: platform => `${platform}(으)로 인계되었어요. 언제든지 여기서 재개할 수 있어요.`,
      systemNote: platform => `↻ ${platform}(으)로 인계됨 — 언제든지 여기서 재개할 수 있어요.`,
      failed: error => `인계 실패: ${error}`,
      timedOut:
        'Hermes가 메시징 연결에 도달하지 못했어요. 설정 → 메시징에서 시작한 후 인계를 다시 시도해 주세요.',
      startMessaging: '메시징 시작'
    }
  },
  tips: {
    close: '이 팁 다시 보지 않기',
    items: {
      'new-session': {
        title: '새롭게 시작하기',
        text: '새 대화는 고유한 컨텍스트, 터미널 및 작업 디렉터리를 가집니다.'
      },
      skills: {
        title: '한 번만 가르치세요',
        text: '스킬은 작업에 필요할 때 Hermes가 불러오는 지침 폴더입니다.'
      },
      messaging: {
        title: '자리 밖에서도 Hermes와 함께',
        text: 'Telegram, Discord, Slack 등을 연결하세요. 동일한 에이전트, 동일한 기억을 유지합니다.'
      },
      artifacts: {
        title: 'Hermes가 만든 모든 것',
        text: '모든 세션의 이미지, 파일, 링크가 한곳에 모여 정리됩니다.'
      },
      cron: {
        title: '스스로 실행되는 작업',
        text: '매시간, 매일 밤 또는 cron 표현식에 따라 프롬프트 실행을 예약하세요.'
      },
      'command-palette': {
        title: '모든 것을 위한 하나의 검색창',
        text: '세션, 설정, 스킬, 명령어 모두 명령 팔레트에서 바로 실행할 수 있습니다.'
      },
      profiles: {
        title: '프로필은 독립적으로 작동해요',
        text: '각 프로필은 고유한 키, 기억, 세션을 가진 독립된 Hermes입니다.'
      },
      'composer-mentions': {
        title: '첨부 및 명령어',
        text: '@를 입력하여 대화에 파일을 추가하고, /를 입력하여 명령어를 실행하세요.'
      },
      'local-runtime-update': {
        title: '로컬 엔진 업데이트가 있어요',
        text: '로컬 모델을 실행하는 엔진을 업데이트합니다. 진행 중인 로컬 요청이 중단될 수 있어요.',
        action: '지금 업데이트'
      },
      'local-setup': {
        title: '이 기기에서 로컬 모델을 실행할 수 있어요',
        text: '현재 하드웨어로 로컬 모델을 구동할 수 있습니다. 대화가 컴퓨터 내에 머무르며 비용이 들지 않아요.',
        action: '설정하기'
      },
      'right-pane': {
        title: '작업 창',
        text: '파일, 터미널, 리뷰 및 앱 내 브라우저가 오른쪽 영역을 공유합니다.'
      }
    }
  },
  errors: {
    genericFailure: '문제가 발생했어요',
    boundaryTitle: '인터페이스에 문제가 발생했어요',
    boundaryDesc: '화면에 예기치 않은 오류가 발생했어요. 대화와 설정은 안전하게 보관되어 있습니다.',
    boundaryDetails: '상세 정보',
    sendDiagnostics: '진단 정보 보내기',
    reloadWindow: '창 다시 불러오기',
    openLogs: '로그 열기'
  },
  ui: {
    search: {
      clear: '검색 초기화'
    },
    pagination: {
      label: '페이지 탐색',
      previous: '이전',
      previousAria: '이전 페이지로 이동',
      next: '다음',
      nextAria: '다음 페이지로 이동'
    },
    sidebar: {
      title: '사이드바',
      description: '모바일 사이드바를 표시합니다.',
      toggle: open => `사이드바 ${open ? '표시' : '숨기기'}`
    }
  }
}
