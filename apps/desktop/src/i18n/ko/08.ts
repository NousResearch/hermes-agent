// ko/08.ts — Korean translation of the `appearance, fieldLabels, fieldDescriptions, uninstallSection, poolLimits` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.
import { defineFieldCopy } from '@/app/settings/field-copy'

import type { TranslationOverrides } from '../define-locale'

export const ko08: TranslationOverrides = {
  settings: {
appearance: {
      title: '모양',
      intro: '데스크톱 전용이에요. 모드는 밝기를, 테마는 색상 팔레트와 채팅 UI를 뜻해요.',
      colorMode: '색상 모드',
      colorModeDesc: '고정 모드를 선택하거나 Hermes가 시스템 설정을 따르게 하세요.',
      toolViewTitle: '도구 호출 표시',
      toolViewDesc: '제품 모드는 원시 도구 페이로드를 숨기고, 기술 모드는 전체 입력/출력을 보여줘요.',
      hideCodeDiffsTitle: '코드 차이 숨기기',
      hideCodeDiffsDesc: '코드 없이 파일 편집을 추가/삭제된 줄 수와 함께 인라인 도구 행으로 표시해요.',
      hideThreadTimelineTitle: '스레드 타임라인 막대 숨기기',
      hideThreadTimelineDesc: '각 대화의 오른쪽 가장자리에 있는 탐색 막대를 숨겨요.',
      reasoningCollapsedTitle: '기본적으로 사고 과정 접기',
      reasoningCollapsedDesc: '스트리밍되는 추론을 직접 열기 전까지 펼치지 않고 그대로 유지해요.',
      uiScaleTitle: 'UI 크기',
      uiScaleDesc: (percent: number) =>
        `앱 전체의 텍스트와 컨트롤 크기를 조절해요. Cmd/Ctrl과 +, -, 0 키도 사용할 수 있어요. 현재: ${percent}%.`,
      sessionDensityTitle: '세션 목록 밀도',
      sessionDensityDesc: '사이드바에서 세션 제목 아래에 표시할 정보량을 선택하세요.',
      sessionDensityCompact: '촘촘하게',
      sessionDensityComfortable: '보통',
      sessionDensityDetailed: '자세히',
      tabStripTitle: '탭 표시줄',
      tabStripDesc:
        '영역 위에 탭을 표시해요. 자동은 다른 채팅이나 타일 영역이 열려 있지 않으면 단일 창에서 탭을 숨겨요.',
      tabStripAuto: '자동',
      tabStripAlways: '항상',
      tabStripNever: '안 함',
      appActionsTitle: '앱 작업',
      appActionsDesc: '설정, 레이아웃, HUD가 제목 표시줄에서 자리 잡는 위치예요. 오른쪽을 선택하면 왼쪽에 탭 공간이 남아요.',
      appActionsLeft: '왼쪽',
      appActionsRight: '오른쪽',
      terminalFontTitle: '터미널 글꼴',
      terminalFontDesc:
        '데스크톱 터미널에 사용할 설치된 글꼴을 선택하세요. Nerd Fonts는 Powerlevel10k와 셸 아이콘을 렌더링해요. 비워 두면 함께 제공되는 JetBrains Mono를 사용해요.',
      terminalFontPlaceholder: 'MesloLGS NF 또는 CSS 글꼴 스택',
      terminalFontPreview: '글리프 미리보기',
      terminalFontReset: '기본값 사용',
      chatFontTitle: '채팅 글꼴',
      chatFontDesc:
        '채팅과 나머지 앱에 사용할 설치된 글꼴을 선택하세요. OpenDyslexic처럼 읽기 편한 글꼴에 유용해요. 비워 두면 테마의 글꼴을 사용해요.',
      chatFontPlaceholder: 'OpenDyslexic 또는 CSS 글꼴 스택',
      chatFontPreview: '미리보기',
      chatFontSample: '다람쥐 헌 쳇바퀴에 타고파. 0123456789',
      chatFontReset: '테마 글꼴 사용',
      translucencyTitle: '창 투명도',
      translucencyDesc: '텍스트까지 포함해 창 전체로 데스크톱이 비쳐 보여요. 라이트와 다크 모드에 각각 맞춰져 있어요.',
      translucencyGlassDesc:
        '무광 유리: 데스크톱이 부드러운 흐림으로 비치면서 텍스트는 선명하게 유지돼요. 라이트와 다크 모드에 각각 맞춰져 있어요.',
      translucencyModeClear: '투명',
      translucencyModeGlass: '유리',
      translucencyTintTitle: '색조',
      translucencyFadeTitle: '페이드',
      translucencyFrostTitle: '프로스트',
      translucencyFrost: {
        'under-window': '깊음',
        popover: '부드러움',
        titlebar: '밝음',
        header: '광택'
      },
      translucencyScopeTitle: '영역',
      translucencyScope: {
        window: '창 전체',
        sidebar: '사이드바만'
      },
      backdropTitle: '채팅 배경',
      backdropDesc: '대화 뒤에 희미하게 보이는 조각상 이미지예요.',
      userBubbleTitle: '메시지 말풍선',
      userBubbleDesc: '내가 보낸 메시지의 투명도예요. 0이면 불투명하고, 100이면 윤곽선만 남아요.',
      textDirectionTitle: '텍스트 방향',
      textDirectionDesc:
        '채팅 메시지와 입력창이 방향을 정하는 방식이에요. 자동은 각 문단의 첫 글자를 따라가요. 섞인 텍스트가 반대 방향으로 정렬되면 방향을 직접 선택하세요. 코드는 항상 왼쪽에서 오른쪽으로 표시돼요.',
      textDirection: { auto: '자동', rtl: '오른쪽에서 왼쪽', ltr: '왼쪽에서 오른쪽' },
      introSplashTitle: '인트로 화면',
      introSplashDesc: '빈 채팅에 표시되는 워드마크와 프롬프트예요.',
      reactionsTitle: '메시지 반응',
      reactionsDesc: 'iMessage 스타일 이모지 탭백이에요 — 메시지에 반응을 남길 수 있고, Hermes도 내 메시지에 반응할 수 있어요.',
      tipsTitle: '앱 내 팁',
      tipsDesc:
        '앱과 Hermes가 가끔 보여주는 힌트예요. 각 팁은 한 번만 표시돼요. 처음 30일이 지나면 자동으로 꺼지지만, 다시 켤 수 있어요.',
      tipsReset: (count: number) => `팁 ${count}개 다시 표시`,
      toursTitle: '가이드 투어',
      toursDesc:
        'Hermes가 앱을 안내하면서 각 단계를 강조 표시해요. 처음 30일이 지나면 자동으로 꺼지지만, 다시 켤 수 있어요.',
      composerPopoutTitle: '플로팅 입력창',
      composerPopoutDesc: '입력창을 고정 위치에서 끌어낼 수 있게 해요. 이 기능을 끄면 항상 화면 아래에 고정돼요.',
      vibeHeartsTitle: 'Vibe 하트',
      vibeHeartsDesc:
        'thanks, ily, good bot이라고 하거나 하트를 보내면 하트가 떠다녀요. 위의 메시지 반응과는 별개예요.',
      embedsTitle: '인라인 임베드',
      embedsDesc:
        '서드파티 사이트(YouTube, X, …)에서 불러오는 리치 미리보기예요. 묻기는 허용할 때까지 플레이스홀더를 보여주고, 항상은 자동으로 불러오며, 끄기는 일반 링크로 표시해요.',
      embedsAsk: '묻기',
      embedsAlways: '항상',
      embedsOff: '끄기',
      embedsReset: (count: number) => `허용한 서비스 ${count}개 초기화`,
      resumeLastSessionTitle: '실행 시 마지막 채팅 다시 열기',
      resumeLastSessionDesc:
        '켜면 앱을 다시 시작할 때 가장 최근 채팅을 다시 열어요. 끄면 항상 새 채팅으로 시작해요.',
      product: '제품',
      productDesc: '간결한 요약과 함께 사람이 읽기 쉬운 도구 활동을 보여줘요.',
      technical: '기술',
      technicalDesc: '원시 도구 인수/결과와 저수준 세부 정보를 포함해요.',
      themeTitle: '테마',
      themeDesc: '데스크톱 팔레트만 해당돼요. 선택한 모드가 그 위에 적용돼요.',
      themeSearchPlaceholder: '내 테마 또는 VS Code Marketplace 검색…',
      themeProfileNote: profile => `${profile} 프로필에 저장됐어요 — 프로필마다 자체 테마를 사용해요.`,
      installTitle: 'VS Code에서 설치',
      installDesc:
        'Marketplace 확장 ID(예: dracula-theme.theme-dracula)를 붙여넣으면 색상 테마를 데스크톱 팔레트로 변환해요.',
      installPlaceholder: 'publisher.extension',
      installButton: '설치',
      installing: '설치 중…',
      installError: '테마를 설치할 수 없어요.',
      installed: name => `“${name}” 설치됨.`,
      removeTheme: '테마 제거',
      importedBadge: '가져옴',
      pet: {
        title: '펫',
        intro:
          '앱 위에 떠다니며 Hermes가 하는 일에 반응하는 애니메이션 petdex 마스코트를 입양해 보세요 — 도구가 실행될 때는 달리고, 성공하면 축하하고, 오류가 나면 시무룩해져요.',
        restartHint:
          '펫은 앱을 한 번 다시 시작해야 해요 — 지금 실행 중인 앱은 이 기능이 추가되기 전에 시작됐어요. Hermes를 종료하고 다시 연 뒤 이곳으로 돌아오세요.',
        scaleTitle: '크기',
        scaleDesc: '떠다니는 마스코트의 크기를 조절해요. 모든 곳에 즉시 적용돼요.',
        roamTitle: '돌아다니기',
        roamDesc: '대기 중일 때 펫이 창 안을 스스로 돌아다니게 해요.',
        chooseTitle: '펫 선택',
        chooseDesc: '하나를 고르면 (필요한 경우) 설치하고 활성화해요.',
        searchPlaceholder: '펫 검색…',
        unreachable: 'petdex 갤러리에 연결할 수 없어요. 연결을 확인하고 이 페이지를 다시 열어 주세요.',
        noMatch: query => `"${query}"에 해당하는 펫이 없어요.`,
        installedTag: '설치됨',
        generatedTag: '생성됨',
        countCapped: (cap, total) => `${total}개 중 ${cap}개 표시 — 입력해서 범위를 좁혀 보세요.`,
        count: n => `펫 ${n}마리.`,
        uninstall: name => `${name} 제거`,
        delete: name => `${name} 삭제`,
        deleteTitle: name => `${name} 삭제할까요?`,
        deleteBody: '이 펫을 영구히 삭제해요 — 다시 설치할 수 없어요.',
        deleteConfirm: '삭제',
        rename: name => `${name} 이름 바꾸기`,
        renameTitle: '펫 이름 바꾸기',
        renamePlaceholder: '펫 이름을 입력하세요',
        renameSave: '저장',
        exportPet: name => `${name} 내보내기`,
        adoptFailed: slug => `${slug} 펫을 입양할 수 없어요`,
        uninstallFailed: slug => `${slug} 펫을 제거할 수 없어요`,
        renameFailed: slug => `${slug} 펫의 이름을 바꿀 수 없어요`,
        exportFailed: slug => `${slug} 펫을 내보낼 수 없어요`,
        noneAvailable: '지금 켤 수 있는 펫이 없어요.',
        turnOnFailed: '펫을 켤 수 없어요.',
        turnOffFailed: '펫을 끌 수 없어요.'
      }
    },
fieldLabels: defineFieldCopy({
      model: '기본 모델',
      modelContextLength: '메인 모델 컨텍스트 창(재정의)',
      fallbackProviders: '대체 모델',
      toolsets: '사용 설정된 도구 세트',
      timezone: '시간대',
      display: {
        personality: '성격',
        showReasoning: '추론 블록'
      },
      desktop: {
        repoScanEnabled: '저장소 자동 검색',
        repoScanRoots: '저장소 검색 루트',
        repoScanExcludePaths: '제외된 저장소 경로'
      },
      agent: {
        maxTurns: '최대 에이전트 단계',
        imageInputMode: '이미지 첨부',
        apiMaxRetries: 'API 재시도',
        serviceTier: '서비스 등급',
        toolUseEnforcement: '도구 사용 강제'
      },
      terminal: {
        cwd: '작업 디렉터리',
        backend: '실행 백엔드',
        timeout: '명령 타임아웃',
        persistentShell: '영구 셸',
        envPassthrough: '환경 변수 전달',
        dockerImage: 'Docker 이미지',
        singularityImage: 'Singularity 이미지',
        modalImage: 'Modal 이미지',
        daytonaImage: 'Daytona 이미지'
      },
      fileReadMaxChars: '파일 읽기 한도',
      toolOutput: {
        maxBytes: '터미널 출력 한도',
        maxLines: '파일 페이지 한도',
        maxLineLength: '줄 길이 한도'
      },
      codeExecution: {
        mode: '코드 실행 모드'
      },
      approvals: {
        mode: '승인 모드',
        timeout: '승인 타임아웃',
        mcpReloadConfirm: 'MCP 다시 로드 확인'
      },
      commandAllowlist: '명령 허용 목록',
      security: {
        redactSecrets: '시크릿 가리기',
        allowPrivateUrls: '사설 URL 허용'
      },
      browser: {
        allowPrivateUrls: '브라우저 사설 URL',
        autoLocalForPrivateUrls: '사설 URL에 로컬 브라우저 사용',
        useRealProfile: '내 실제 브라우저 프로필 사용'
      },
      checkpoints: {
        enabled: '파일 체크포인트',
        maxSnapshots: '체크포인트 한도'
      },
      voice: {
        maxRecordingSeconds: '최대 녹음 길이',
        autoTts: '응답 소리 내어 읽기',
        voiceChatMode: '음성 채팅 모드',
        gptLive: {
          voice: 'GPT-Live 음성',
          instructions: 'GPT-Live 페르소나'
        }
      },
      stt: {
        enabled: '음성 텍스트 변환',
        echoTranscripts: '스크립트 다시 게시',
        provider: '음성 텍스트 변환 제공자',
        local: {
          model: '로컬 변환 모델',
          language: '변환 언어'
        },
        openai: {
          model: 'OpenAI STT 모델'
        },
        groq: {
          model: 'Groq STT 모델'
        },
        mistral: {
          model: 'Mistral STT 모델'
        },
        elevenlabs: {
          modelId: 'ElevenLabs STT 모델',
          languageCode: 'ElevenLabs 언어',
          tagAudioEvents: '오디오 이벤트 태그',
          diarize: '화자 분리'
        }
      },
      tts: {
        provider: '텍스트 음성 변환 제공자',
        edge: {
          voice: 'Edge 음성'
        },
        openai: {
          model: 'OpenAI TTS 모델',
          voice: 'OpenAI 음성'
        },
        elevenlabs: {
          voiceId: 'ElevenLabs 음성',
          modelId: 'ElevenLabs 모델'
        },
        xai: {
          voiceId: 'xAI(Grok) 음성',
          language: 'xAI 언어',
          speed: 'xAI 재생 속도',
          autoSpeechTags: 'xAI 자동 음성 태그',
          optimizeStreamingLatency: 'xAI 스트리밍 지연 최적화',
          sampleRate: 'xAI 샘플 레이트',
          bitRate: 'xAI 비트레이트'
        },
        minimax: {
          model: 'MiniMax TTS 모델',
          voiceId: 'MiniMax 음성'
        },
        mistral: {
          model: 'Mistral TTS 모델',
          voiceId: 'Mistral 음성'
        },
        gemini: {
          model: 'Gemini TTS 모델',
          voice: 'Gemini 음성'
        },
        neutts: {
          model: 'NeuTTS 모델',
          device: 'NeuTTS 장치'
        },
        kittentts: {
          model: 'KittenTTS 모델',
          voice: 'KittenTTS 음성'
        },
        piper: {
          voice: 'Piper 음성'
        },
        deepinfra: {
          model: 'DeepInfra TTS 모델',
          voice: 'DeepInfra 음성'
        }
      },
      memory: {
        memoryEnabled: '영구 메모리',
        userProfileEnabled: '사용자 프로필',
        memoryCharLimit: '메모리 예산',
        userCharLimit: '프로필 예산',
        provider: '메모리 제공자'
      },
      context: {
        engine: '컨텍스트 엔진'
      },
      compression: {
        enabled: '자동 압축',
        threshold: '압축 임계값',
        codexGpt55Autoraise: 'Codex 압축 자동 상향',
        targetRatio: '압축 목표',
        protectLastN: '보호할 최근 메시지'
      },
      auxiliary: {
        compression: {
          timeout: '압축 모델 타임아웃(초)'
        }
      },
      delegation: {
        model: '서브에이전트 모델',
        provider: '서브에이전트 제공자',
        maxIterations: '서브에이전트 턴 한도',
        maxConcurrentChildren: '병렬 서브에이전트',
        childTimeoutSeconds: '서브에이전트 타임아웃',
        reasoningEffort: '서브에이전트 추론 수준'
      },
      updates: {
        nonInteractiveLocalChanges: '앱 내 업데이트 시 로컬 변경'
      }
    }),
fieldDescriptions: defineFieldCopy({
      model: '입력창에서 다른 모델을 선택하지 않는 한 새 채팅에 사용돼요.',
      modelContextLength:
        '메인 채팅 모델에서 감지된 컨텍스트 창만 재정의해요(토큰). 0으로 두면 선택한 모델에서 감지된 값을 사용해요. 보조/MoA 모델에는 영향을 주지 않아요.',
      fallbackProviders: '기본 모델이 실패할 때 시도할 예비 제공자:모델 항목이에요.',
      display: {
        personality: '새 세션의 기본 어시스턴트 스타일이에요.',
        showReasoning: '백엔드가 제공할 때 추론 섹션을 표시해요.'
      },
      desktop: {
        repoScanEnabled: '프로젝트에 표시할 Git 저장소를 찾기 위해 로컬 폴더를 검사해요.',
        repoScanRoots: '검사할 폴더예요. 비워 두면 홈 디렉터리를 검사해요.',
        repoScanExcludePaths: '저장소를 찾는 동안 건너뛸 폴더와 그 하위 폴더예요.'
      },
      timezone: 'IANA 시간대 식별자예요. 비워 두면 시스템 시간대를 사용해요.',
      browser: {
        useRealProfile:
          '로컬 브라우징에 실제 로그인 정보를 사용해요. Hermes는 기본 브라우저의 프로필(쿠키, 로그인 정보, 환경설정)을 관리되는 스냅샷으로 복사한 뒤 함께 제공되는 Chromium으로 구동해요. 실제 프로필은 직접 열지 않으며, 실행할 때마다 복사본을 갱신해요. 클라우드 브라우저 백엔드가 설정돼 있어도 요청 시 에이전트가 로컬 실제 프로필 세션을 열 수 있어요. Chromium 브라우저(Chrome, Edge, Brave, Brave Origin, Chromium)만 지원하며, 기본 브라우저가 Chromium이 아니면 명확한 메시지와 함께 실패해요. 기본값은 꺼짐이에요.'
      },
      agent: {
        imageInputMode: '이미지 첨부를 모델에 보내는 방식을 제어해요.',
        maxTurns: 'Hermes가 실행을 중단하기 전까지의 도구 호출 턴 상한이에요.'
      },
      terminal: {
        cwd: '도구 및 터미널 작업에 사용할 기본 프로젝트 폴더예요.',
        persistentShell: '백엔드가 지원하면 명령 사이에 셸 상태를 유지해요.',
        envPassthrough: '도구 실행에 전달할 환경 변수예요.',
        dockerImage: '실행 백엔드가 Docker일 때 사용하는 컨테이너 이미지예요.',
        singularityImage: '실행 백엔드가 Singularity일 때 사용하는 이미지예요.',
        modalImage: '실행 백엔드가 Modal일 때 사용하는 이미지예요.',
        daytonaImage: '실행 백엔드가 Daytona일 때 사용하는 이미지예요.'
      },
      codeExecution: {
        mode: '코드 실행을 현재 프로젝트로 얼마나 엄격하게 제한할지 설정해요.'
      },
      fileReadMaxChars: 'Hermes가 파일 요청 한 번에 읽을 수 있는 최대 문자 수예요.',
      approvals: {
        mode: '명시적 승인이 필요한 명령을 Hermes가 처리하는 방식이에요.',
        timeout: '승인 요청이 타임아웃되기까지 기다리는 시간이에요.'
      },
      security: {
        redactSecrets: '가능하면 감지된 시크릿을 모델이 볼 수 있는 콘텐츠에서 숨겨요.'
      },
      checkpoints: {
        enabled: '파일을 편집하기 전에 롤백 스냅샷을 만들어요.'
      },
      memory: {
        memoryEnabled: '향후 세션에 도움이 되는 기억을 저장해요.',
        userProfileEnabled: '사용자 선호를 간결하게 정리한 프로필을 유지해요.'
      },
      context: {
        engine: '컨텍스트 한도에 가까워진 긴 대화를 관리하는 전략이에요.'
      },
      compression: {
        enabled: '대화가 커지면 오래된 컨텍스트를 요약해요.',
        codexGpt55Autoraise: '지원되는 ChatGPT Codex OAuth 모델에서 압축 임계값을 85%로 올려요.'
      },
      auxiliary: {
        compression: {
          timeout:
            '호출당 보조 압축 모델을 기다리는 시간(초)이에요(기본 120). 느린 로컬 모델에서는 값을 올리세요.'
        }
      },
      voice: {
        autoTts: '어시스턴트 응답을 자동으로 소리 내어 읽어요.',
        voiceChatMode:
          'chained: 아래 제공자로 음성→텍스트 → Hermes → 텍스트→음성 순서로 처리해요. gpt-live: 전이중 OpenAI 음성 모델(gpt-live-1) 하나가 듣고 말하며 모든 실제 요청을 Hermes에 넘겨요. 선택한 모델이 전체 도구 세트로 답변해요. OpenAI API 키가 필요하고, 음성 계층은 분당 $0.05가 청구돼요.',
        gptLive: {
          voice: 'GPT-Live 모드에 사용할 음성이에요. 커스텀 음성 ID도 사용할 수 있어요.',
          instructions:
            '실시간 음성 페르소나에 추가할 문장(말투, 속도, 언어)이에요. Hermes는 자체 시스템 프롬프트를 유지해요.'
        }
      },
      tts: {
        xai: {
          voiceId: 'xAI 음성 ID(예: eve) 또는 커스텀 음성 ID예요.',
          language: '음성 언어 코드(예: en, pt-BR) 또는 자동 감지를 위한 "auto"예요.',
          speed: '재생 속도예요. 0.7 = 느리게, 1.0 = 보통, 1.5 = 빠르게.',
          autoSpeechTags: '합성 전에 LLM이 표현력 있는 오디오 태그([laughing], [sighs])를 대본에 넣도록 해요.',
          optimizeStreamingLatency: '지연과 품질의 절충이에요. 0 = 최고 품질, 2 = 최저 지연.',
          sampleRate: '오디오 샘플 레이트(Hz)예요. 높을수록 품질이 좋고 파일이 커져요.',
          bitRate: 'MP3 비트레이트(bps)예요. 코덱이 mp3일 때만 적용돼요.'
        },
        neutts: {
          device: 'NeuTTS용 로컬 추론 장치예요.'
        }
      },
      stt: {
        enabled: '로컬 또는 제공자 기반 음성 변환을 사용 설정해요.',
        echoTranscripts: '음성 메시지의 원본 🎙️ 스크립트를 채팅에 다시 게시해요.',
        elevenlabs: {
          languageCode: '선택적인 ISO-639-3 언어 코드예요. 비워 두면 ElevenLabs가 자동 감지해요.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Hermes가 앱에서 스스로 업데이트할 때(터미널 프롬프트 없음) 로컬 소스 편집을 유지(stash)하거나 버려요(discard). 터미널 업데이트는 항상 물어봐요.'
      }
    }),
uninstallSection: {
      dangerZone: '위험 구역',
      checkingInstalled: '설치된 항목 확인 중…',
      uninstallHermes: 'Hermes 제거',
      chooseHowMuch:
        '얼마나 제거할지 선택하세요. 마무리를 위해 앱이 종료돼요. 설치 프로그램을 다시 열면 언제든 돌아올 수 있어요.',
      confirmUninstall: '제거 확인',
      confirmBody: what => `제거 항목: ${what}. 이 작업은 되돌릴 수 없어요.`,
      appLabel: '앱:',
      couldNotStart: '제거를 시작할 수 없어요.',
      uninstalling: '제거 중…',
      yesUninstall: '네, 제거합니다',
      options: {
        gui: {
          title: '채팅 GUI만 제거',
          description: '이 데스크톱 앱을 제거해요. Hermes 에이전트, 설정, 채팅은 모두 그대로 남아요.',
          consequence: '데스크톱 채팅 GUI(이 앱과 그 데이터)'
        },
        lite: {
          title: 'GUI + 에이전트 제거, 데이터는 유지',
          description:
            '앱과 Hermes 에이전트를 제거하지만 나중에 다시 설치할 수 있도록 설정, 채팅, 시크릿은 남겨 둬요.',
          consequence: '채팅 GUI와 Hermes 에이전트(설정, 채팅, 시크릿은 유지)'
        },
        full: {
          title: '모두 제거',
          description: '앱과 에이전트, 모든 사용자 데이터를 제거해요 — 설정, 채팅, 예약된 작업, 시크릿, 로그.',
          consequence: '전부 — 채팅 GUI, Hermes 에이전트, 그리고 모든 설정, 채팅, 시크릿, 로그'
        }
      }
    },
poolLimits: {
      warmBotBackendsAria: '봇 백엔드 예열',
      warmBotBackendsTitle: '봇 백엔드 예열',
      backendIdleTimeoutAria: '백엔드 유휴 타임아웃(밀리초)',
      backendIdleTimeoutTitle: '백엔드 유휴 타임아웃'
    },
  },
}
