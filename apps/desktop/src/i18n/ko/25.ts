// ko/25.ts — Korean translation of the `install, onboarding` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko25: TranslationOverrides = {
install: {
    stageStates: {
      pending: '대기 중',
      running: '설치 중',
      succeeded: '완료',
      skipped: '건너뜀',
      failed: '실패'
    },
    oneTimeTitle: 'Hermes는 최초 1회 설치가 필요해요',
    unsupportedDesc: platform =>
      `${platform}에서는 아직 최초 실행 시 자동 설치를 지원하지 않아요. 터미널을 열고 아래 명령어를 실행한 뒤 이 앱을 다시 실행해 주세요. 이후 실행부터는 이 단계를 건너뜁니다.`,
    installCommand: '설치 명령어',
    copyCommand: '명령어 복사',
    viewDocs: '설치 문서 보기',
    installTo: '설치 위치',
    retryAfterRun: '실행 완료 -- 다시 시도',
    setupChoiceTitle: 'Hermes Desktop 설정',
    setupChoiceDesc:
      '이미 실행 중인 Hermes 게이트웨이에 이 앱을 연결하거나, 이 컴퓨터에 Hermes를 로컬로 설치하세요.',
    setupChoiceDescLocal: '이 컴퓨터에 Hermes를 설치하거나, 이미 실행 중인 Hermes 게이트웨이에 연결하세요.',
    connectExistingTitle: '기존 Hermes에 연결',
    connectExistingShort: '기존 인스턴스 연결',
    connectExistingDesc: '세션 토큰 또는 브라우저 로그인으로 원격 백엔드를 사용합니다. 로컬 설치는 시작되지 않아요.',
    installLocalTitle: 'Hermes 로컬 설치',
    installLocalDesc: 'Hermes를 다운로드하고, Python 환경을 생성하며, 이 컴퓨터에서 백엔드를 실행합니다.',
    useLocalTitle: '이 컴퓨터의 Hermes 사용',
    useLocalDesc: 'Hermes 런타임이 이미 설치되어 있어요 — 클릭 한 번으로 시작할 수 있으며 아무것도 다운로드하지 않아요.',
    bundledLocalDesc: '이 앱에 포함된 Hermes 런타임을 사용합니다 — 번들 백엔드가 로컬 설치 항목이에요.',
    localStartUnavailable: '로컬 설치를 시작할 수 없어요. Hermes Desktop을 다시 시작하고 다시 시도해 주세요.',
    remoteSetupTitle: '기존 Hermes에 연결',
    remoteSetupDesc: '게이트웨이 URL을 입력하세요. Hermes Desktop이 토큰 또는 브라우저 로그인이 필요한지 감지합니다.',
    remoteUrlTitle: '게이트웨이 URL',
    remoteUrlDesc: 'Hermes 게이트웨이의 기본 URL을 입력하세요. 원격인 경우 https://를 포함해야 해요.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: '게이트웨이 인증 감지 중...',
    probeError:
      'Hermes가 해당 주소에 연결할 수 없어요. URL을 확인하고 다른 컴퓨터에서 Hermes가 실행 중인지 확인해 주세요 — 응답이 오면 로그인 옵션이 표시됩니다.',
    probeErrorDetails: '상세 정보',
    identityProvider: 'ID 제공자',
    authTitle: '인증',
    authNeedsOauth: provider => `이 게이트웨이를 테스트하기 전에 ${provider}(으)로 로그인해 주세요.`,
    authSignedIn: '브라우저 로그인이 완료되었어요.',
    connected: '연결됨',
    signIn: '로그인',
    signInWith: provider => `${provider}(으)로 로그인`,
    enterUrlFirst: '게이트웨이 URL을 먼저 입력해 주세요.',
    signInIncomplete: '인증이 완료되기 전에 로그인 창이 닫혔어요.',
    tokenTitle: '세션 토큰',
    tokenDesc: '원격 게이트웨이 .env 파일에서 세션 토큰을 복사하여 붙여넣으세요.',
    pasteSessionToken: '세션 토큰 붙여넣기',
    incompleteSignInTest: 'OAuth로 보호된 게이트웨이를 테스트하기 전에 먼저 로그인해 주세요.',
    incompleteTokenTest: '이 게이트웨이를 테스트하기 전에 세션 토큰을 입력해 주세요.',
    testConnection: '연결 테스트',
    testSucceeded: (baseUrl, version) => `${baseUrl}에 연결되었어요${version ? ` (${version})` : ''}.`,
    applyRemote: '적용 및 다시 연결',
    backToSetup: '뒤로',
    failedTitle: '설치 실패',
    settingUpTitle: 'Hermes Agent 설정 중',
    finishingTitle: '마무리 중',
    failedDesc:
      '설치 단계 중 하나를 완료하지 못했어요. 다른 Hermes 인스턴스가 실행 중이거나 인터넷 연결이 끊겼거나 바이러스 백신이 설치 프로그램을 차단했을 때 발생할 수 있어요. 다른 Hermes 창을 닫은 뒤 다시 로드 및 다시 시도를 선택해 주세요. 계속 실패하면 로그를 열어 지원팀에 문의해 주세요.',
    activeDesc:
      '최초 1회 설정입니다. Hermes 설치 프로그램이 의존성을 다운로드하고 시스템을 구성하고 있어요. 이후 실행부터는 이 단계를 건너뜁니다.',
    progress: (completed, total) => `총 ${total}단계 중 ${completed}단계 완료`,
    currentStage: stage => ` -- 현재: ${stage}`,
    fetchingManifest: '설치 프로그램 매니페스트 가져오는 중...',
    error: '오류',
    hideOutput: '설치 프로그램 출력 숨기기',
    showOutput: '설치 프로그램 출력 표시',
    lines: count => `${count}줄`,
    noOutput: '출력이 아직 없어요.',
    cancelling: '취소 중...',
    cancelInstall: '설치 취소',
    transcriptSaved: '전체 기록 저장 위치:',
    copiedOutput: '복사 완료!',
    copyOutput: '출력 복사',
    reloadRetry: '다시 로드 및 다시 시도',
    openLogs: '로그 열기'
  },
onboarding: {
    headerTitle: 'Hermes Agent 설정을 시작해 볼까요',
    headerDesc: '채팅을 시작하려면 모델 제공자를 연결하세요. 대부분 클릭 한 번으로 완료됩니다.',
    preparingInstall: 'Hermes 설치를 마무리하고 있어요. 처음 실행할 때 보통 1분 이내로 걸려요.',
    starting: 'Hermes 시작 중…',
    lookingUpProviders: '제공자 찾는 중...',
    collapse: '접기',
    otherProviders: '기타 제공자',
    haveApiKey: 'API 키가 있어요',
    chooseLater: '나중에 제공자 선택하기',
    recommended: '추천',
    connected: '연결됨',
    featuredPitch: '구독 하나로 300개 이상의 프론티어 모델 사용 — Hermes를 실행하는 추천 방법',
    fireworksPitch: '직접 모델 API — Fireworks 호스팅 프론티어 모델',
    localModelsTitle: '로컬에서 모델 실행',
    localModelsPitch: '계정 불필요 — 모델을 다운로드하여 이 기기에서 바로 실행',
    openRouterPitch: '키 하나로 수백 개의 모델 사용 — 든든한 기본 옵션',
    apiKeyOptions: {
      fireworks: {
        short: '직접 모델 API',
        description: 'Fireworks AI에서 호스팅하는 모델에 직접 액세스합니다.'
      },
      openrouter: {
        short: '키 하나로 다양한 모델',
        description: '단일 키로 수백 개의 모델을 호스팅합니다. 신규 설치 시 좋은 기본값입니다.'
      },
      openai: { short: 'GPT급 모델', description: 'OpenAI 모델에 직접 액세스합니다.' },
      gemini: { short: 'Gemini 모델', description: 'Google Gemini 모델에 직접 액세스합니다.' },
      xai: { short: 'Grok 모델', description: 'xAI Grok 모델에 직접 액세스합니다.' },
      local: {
        short: '자체 호스팅',
        description: 'Hermes가 로컬 또는 자체 호스팅된 OpenAI 호환 엔드포인트(vLLM, llama.cpp, Ollama 등)를 가리키도록 설정합니다.'
      }
    },
    backToSignIn: '로그인으로 돌아가기',
    getKey: '키 발급받기',
    replaceCurrent: '현재 값 교체',
    pasteApiKey: 'API 키 붙여넣기',
    localApiKeyPlaceholder: 'API 키 (선택 사항 — 엔드포인트에서 필요한 경우에만 입력)',
    couldNotSave: '자격 증명을 저장할 수 없어요.',
    connecting: '연결 중',
    update: '업데이트',
    flowSubtitles: {
      pkce: '브라우저를 열어 로그인한 후 여기서 계속 진행해요',
      device_code: '브라우저에서 인증 페이지를 열면 Hermes가 자동으로 연결돼요',
      external: '터미널에서 한 번 로그인한 다음 돌아와서 채팅을 시작하세요'
    },
    startingSignIn: provider => `${provider} 로그인 시작 중...`,
    verifyingCode: provider => `${provider}에서 코드 확인 중...`,
    connectedProvider: provider => `${provider} 연결됨`,
    connectedPicking: provider => `${provider} 연결됨. 기본 모델 선택 중...`,
    signInFailed: '로그인에 실패했어요. 다시 시도해 주세요.',
    signInExpired:
      '완료하기 전에 로그인 페이지가 만료되었어요. 다시 시도하여 몇 분 내에 브라우저 단계를 완료하거나, 대신 API 키를 사용해 주세요.',
    signInDidNotFinish: provider =>
      `${provider} 로그인이 완료되지 않았어요. 인터넷 연결을 확인하고 다시 시도하거나, 다른 제공자를 선택해 주세요.`,
    tryAgain: '다시 시도',
    useApiKeyInstead: 'API 키 대신 사용',
    errorDetails: '상세 정보',
    pickDifferentProvider: '다른 제공자 선택',
    signInWith: provider => `${provider}(으)로 로그인`,
    openedBrowser: provider => `브라우저에서 ${provider} 페이지를 열었어요.`,
    authorizeThere: '해당 페이지에서 Hermes를 승인해 주세요.',
    copyAuthCode: '인증 코드를 복사하여 아래에 붙여넣으세요.',
    pasteAuthCode: '인증 코드 붙여넣기',
    reopenAuthPage: '인증 페이지 다시 열기',
    autoBrowser: provider =>
      `브라우저에서 ${provider} 페이지를 열었어요. 해당 페이지에서 Hermes를 승인하면 복사하거나 붙여넣을 필요 없이 자동으로 연결됩니다.`,
    reopenSignInPage: '로그인 페이지 다시 열기',
    waitingAuthorize: '승인 대기 중...',
    externalPending: provider =>
      `${provider}은(는) 자체 CLI를 통해 로그인합니다. 터미널에서 이 명령어를 실행한 다음 돌아와서 "로그인 완료"를 선택하세요:`,
    signedIn: '로그인 완료',
    deviceCodeOpened: provider => `브라우저에서 ${provider} 페이지를 열었어요. 해당 페이지에 다음 코드를 입력하세요:`,
    reopenVerification: '인증 페이지 다시 열기',
    copy: '복사',
    defaultModel: '기본 모델',
    freeTier: '무료 티어',
    pro: 'Pro',
    free: '무료',
    price: (input, output) => `Mtok당 입력 ${input} / 출력 ${output}`,
    change: '변경',
    startChatting: '시작하기',
    docs: provider => `${provider} 문서`
  },
}