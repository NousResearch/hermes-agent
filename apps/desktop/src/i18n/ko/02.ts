// ko/02.ts — Korean translation of the `connectorsPage` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko02: TranslationOverrides = {
connectorsPage: {
    title: '커넥터',
    searchPlaceholder: (count: number) => `${count}개 앱 검색`,
    filterCategory: '카테고리',
    categoryAll: '모든 카테고리',
    uncategorised: '미분류',

    residencyLocal: '이 기기에서',

    segment: {
      all: '전체',
      available: '사용 가능',
      connected: '연결됨',
      off: '꺼짐'
    },

    group: {
      connected: '연결됨',
      connectedNote: '끊긴 연결이 먼저 표시돼요.',
      available: '사용 가능',
      off: '꺼짐',
      offNote: '로그인 정보는 유지돼요.'
    },

    card: {
      kindManaged: '관리형',
      kindCatalog: 'MCP · 카탈로그',
      kindCustom: 'MCP · 사용자 정의',
      kindPlugin: (plugin: string) => `MCP · 플러그인 ${plugin}`,
      inCatalog: 'Hermes 카탈로그에 있음',
      hostedTwin: '관리형 버전 사용 가능',
      alsoLocal: '이 기기에서도 실행돼요',
      open: (name: string) => `${name} 열기`,
      turnServerOn: (name: string) => `${name} 켜기`,
      turnServerOff: (name: string) => `${name} 끄기`,
      state: {
        accessExpired: '액세스 만료됨',
        available: '사용 가능',
        connected: '연결됨',
        connecting: '연결 중',
        connectionUnknown: '상태 알 수 없음',
        couldNotConnect: '연결할 수 없음',
        offByYourOrganisation: '조직에서 꺼둠',
        offForYou: '내 계정에서 꺼짐',
        serverConnecting: '연결 중…',
        serverError: '오류',
        serverNeedsAuth: '인증 필요',
        serverOff: '꺼짐',
        serverOn: '켜짐',
        serverOnUnused: '켜짐, 사용 안 함'
      },
      fact: {
        tools: (count: number) => `${count}개 도구${count === 1 ? '' : ''}`,
        toolsOff: (count: number) => `${count}개 도구${count === 1 ? '' : ''} 꺼짐`,
        toolsOn: (count: number) => `${count}개 도구${count === 1 ? '' : ''} 켜짐`,
        toolsSomeOn: (total: number, on: number) => `도구 ${total}개 중 ${on}개 켜짐`
      },
      verb: {
        authenticate: '인증',
        connect: '연결',
        install: '설치',
        openLogs: '로그 열기',
        reconnect: '다시 연결',
        stopWaiting: '대기 중지',
        tryAgain: '다시 시도',
        turnBackOn: '다시 켜기'
      },
      reason: {
        finishSignIn: '브라우저에서 로그인을 마쳐 주세요.',
        reconnect: '이 앱을 계속 쓰려면 다시 연결하세요.',
        serverError: '서버가 연결을 거부했어요.',
        serverNeedsAuth: '서버가 응답하도록 로그인하세요.'
      }
    },

    page: {
      loading: '카탈로그와 이 컴퓨터의 서버를 읽는 중',
      emptyTitle: '아직 앱이 없어요. 이 컴퓨터에 서버를 추가해 시작하세요.',
      noMatchTitle: '일치하는 앱이 없어요',
      noMatchBody: '일치하는 항목이 없어요. 직접 만든 MCP 서버를 Hermes에 연결해 추가할 수 있어요.',
      clearSearch: '검색 지우기',
      hostedFailedTitle: '호스팅된 앱에 연결할 수 없어요.',
      hostedFailedBody: '이 컴퓨터의 서버는 영향을 받지 않고 계속 실행 중이에요. 꺼진 것은 없어요.',
      retry: '다시 시도',
      matchesElsewhere: (count: number) => `다른 그룹에 ${count}개${count === 1 ? '' : ''} 더 일치해요.`,
      showAllMatches: '모든 일치 항목 표시',
      segmentNoMatch: (segment: string) => `${segment}에 일치하는 항목이 없어 모든 일치 항목을 표시해요.`,
      freeTierNote: '로그인하기 전까지 연결은 이 컴퓨터에 남아요.',
      signInLine: '관리형 앱을 쓰려면 Nous에 로그인하세요.',
      signIn: '로그인',
      managedUnavailable: '이 계정에서는 아직 관리형 앱을 사용할 수 없어요.',
      writeFailed: '변경 사항이 저장되지 않았어요.',
      refreshFailed: '도구 목록을 새로 고치지 못했어요.',
      disconnectNoAccount: '여기서 연결을 끊을 계정이 Hermes에 없어요. 페이지를 새로 고친 뒤 다시 시도하세요.',
      disconnectRefused:
        '지금은 Nous에서 이 로그인을 제거할 수 없어요. 대신 스위치로 앱을 끄거나 나중에 다시 시도하세요.'
    },

    add: {
      action: '직접 추가',
      title: '사용자 정의 MCP 연결',
      hint: '이 기기의 mcp.json에 항목 하나 추가',
      pasteLabel: '명령이나 스니펫 붙여넣기',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /path/to/dir',
      pasteNoMatch: '서버로 인식되는 내용이 없어요. 대신 아래 항목을 채워 주세요.',
      name: '이름',
      nameTaken: '이미 사용 중인 이름이에요.',
      type: '유형',
      typeStdio: 'STDIO',
      typeHttp: 'Streamable HTTP',
      command: '실행할 명령',
      args: '인수',
      addArg: '+ 인수 추가',
      envVars: '환경 변수',
      addEnvVar: '+ 환경 변수 추가',
      passthrough: '환경 변수 전달',
      addPassthrough: '+ 변수 추가',
      cwd: '작업 디렉터리',
      url: 'URL',
      headers: '헤더',
      addHeader: '+ 헤더 추가',
      auth: '인증',
      authNone: '없음',
      authOauth: 'OAuth',
      authBearer: 'Bearer 토큰',
      keyPlaceholder: '키',
      valuePlaceholder: '값',
      removeRow: '이 행 제거',
      editJson: 'mcp.json 편집',
      saveFailed: '서버가 저장되지 않았어요.'
    },

    dialog: {
      disconnect: '연결 끊기',
      disconnectTitle: (name: string) => `${name} 연결을 끊을까요?`,
      disconnectBody: 'Hermes가 이 계정으로 동작하지 않게 돼요. 언제든 다시 연결할 수 있어요.',
      menuRefreshTools: '도구 새로 고침',
      moreActions: '추가 작업',
      removeServerTitle: (name: string) => `${name} 제거할까요?`,
      removeServerBody: '이 컴퓨터의 mcp.json에서 항목이 빠져요. 다른 것은 삭제되지 않아요.',
      appSwitch: (name: string) => `Hermes가 ${name}을 사용해도 됨`,
      waysTitle: (name: string) => `${name} 실행 위치`,
      wayNotConnected: (name: string) => `아직 연결되지 않았어요. 브라우저에서 ${name}에 로그인하세요.`,
      wayHosted: '관리형',
      bothOn: (name: string) => `둘 다 켜져 있어서 Hermes가 모든 ${name} 도구를 두 번 보게 돼요.`,
      turnOffLocal: '로컬 서버 끄기',
      providedByPlugin: (plugin: string) => `플러그인 ${plugin}에서 제공`,
      openPlugins: '플러그인 탭 열기',
      // Verbatim, by decision of the design of record.
      nousLine: 'Nous apps follow your account, not the profile.',
      rulesReadOnly: '지금은 규칙을 변경할 수 없어요.',
      rulesAppOff: (name: string) => `${name}을 켜야 도구를 바꿀 수 있어요.`,
      rulesSignIn: '여기서 Hermes가 할 수 있는 일을 바꾸려면 로그인하세요.',
      orgNote: (count: number) => `조직에서 도구 ${count}개를 껐어요.`,
      orgLink: '커넥터 관리자 열기',
      connectEnded: '로그인이 완료되지 않았어요.',
      connectOpenAgain: '링크 다시 열기',
      tokensPerCall: '호출당 토큰',
      usesPerMonth: '30일간 사용 횟수',
      advanced: '고급',
      advancedHint: 'mcp.json 항목과 로그'
    },

    tools: {
      title: '도구',
      notInstalledBody: '이 기기에 설치하면 제공되는 도구를 볼 수 있어요.',
      summaryTitle: (name: string) => `Hermes가 ${name}으로 할 수 있는 일`,
      summaryPreviewTitle: (name: string) => `연결하면 Hermes가 ${name}으로 할 수 있는 일`,
      summaryCount: (count: number) => `${count}개 도구${count === 1 ? '' : ''}`,
      summaryAllTools: '모든 도구',
      summaryOther: '기타',
      allToolsSwitch: '모든 도구 켜기 또는 끄기',
      summaryAllOn: '모두 켜짐',
      summarySomeOn: (on: number, total: number) => `도구 ${total}개 중 ${on}개 켜짐`,
      summaryOff: '꺼짐',
      showAllTools: (count: number) => `도구 ${count}개${count === 1 ? '' : ''} 모두 표시`,
      showSummary: '요약 표시',
      facetSwitch: (facet: string) => `${facet} 도구 켜기 또는 끄기`,
      moreHints: (count: number) => `+${count}`,
      staleSignIn: '최신 도구 목록을 읽으려면 로그인하세요.',
      searchCountPlaceholder: (count: number) => `도구 ${count}개 검색`,
      toolList: (name: string) => `${name} 도구`,
      categorySelect: (count: number) => `카테고리 ${count}개`,
      showDeprecated: (count: number) => `사용 중단된 도구 ${count}개 표시`,
      hideDeprecated: (count: number) => `사용 중단된 도구 ${count}개 숨기기`,
      quickReadOnly: '읽기 전용',
      quickNoDestructive: '파괴적 도구 끄기',
      quickEverythingOn: '모두 켜기',
      lockedHint: '조직에서 꺼둠',
      turnToolOn: (tool: string) => `${tool} 켜기`,
      turnToolOff: (tool: string) => `${tool} 끄기`,
      showDetails: (tool: string) => `${tool}이(가) 하는 일 표시`,
      hideDetails: (tool: string) => `${tool}이(가) 하는 일 숨기기`,
      noMatch: '이 필터에 맞는 도구가 없어요.',
      loading: '도구 목록을 읽는 중',
      unavailableLine: '도구 목록을 사용할 수 없어요.',
      needsAuthTitle: (name: string) => `${name}에 로그인해 도구를 읽어 오세요.`,
      needsAuthBody: '로그인 정보는 이 컴퓨터에 남아요. 아무것도 밖으로 나가지 않아요.',
      retry: '다시 시도',
      goneTitle: (name: string) => `${name}이(가) 카탈로그에서 빠졌어요.`,
      goneBody: 'Hermes가 더 이상 호출할 수 없어요. 제거할 때까지 항목은 남아 있으니 사라지는 것은 없어요.',
      remove: '제거',
      offTitle: (name: string) => `${name}이(가) 꺼져 있어요.`,
      offBody: '위 스위치로 켜면 제공되는 도구를 읽을 수 있어요.',
      signedOutTitle: '도구 목록을 읽으려면 Nous에 로그인하세요.',
      signedOutBody: '이 컴퓨터의 서버는 영향을 받지 않아요.',
      conflictTitle: '편집하는 동안 누군가 이 규칙을 바꿨어요.',
      // Two sentences at most, and the second says the work is still here.
      conflictBody: (theyOff: number, theyOn: number) => {
        const they = [
          theyOff > 0 ? `켜둔 도구 ${theyOff}개${theyOff === 1 ? '' : ''}를 껐어요` : '',
          theyOn > 0 ? `꺼둔 도구 ${theyOn}개${theyOn === 1 ? '' : ''}를 켜 두었어요` : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `상대방이 ${they.join(', ')}. ` : ''}편집 내용은 화면에 남아 있고, 아무것도 저장되지 않았어요.`
      },
      conflictReload: '상대방 버전 불러오기',
      conflictSave: '내 버전으로 저장',
      saveFailed: '도구 규칙이 저장되지 않았어요.',
      footerDirty: (off: number, backOn: number) =>
        `도구 ${off}개${off === 1 ? '' : ''} 꺼짐, 다시 켜짐 ${backOn === 0 ? '없음' : backOn}`,
      discard: '버리기',
      save: '변경 사항 저장',
      saving: '저장 중...'
    },

    // The label rides in every tool row, so it stays short enough not to widen one.
    vocabulary: {
      facetRead: { label: '읽기', long: '이 앱에서 데이터를 읽어요. 아무것도 바꾸지 않아요.' },
      facetWrite: { label: '쓰기', long: '이 앱에서 무언가를 만들거나 바꿔요.' },
      facetDestructive: { label: '파괴적', long: '이 앱의 무언가를 완전히 지울 수 있어요.' },
      facetUnclassified: { label: '효과 알 수 없음', long: '앱이 이 도구가 하는 일을 밝히지 않았어요.' },
      hintReadOnly: { label: '읽기 전용', long: '이 도구는 읽기만 한다고 밝혀요.' },
      hintCreate: { label: '생성', long: '새로운 것을 만들어요.' },
      hintUpdate: { label: '수정', long: '이미 있는 것을 바꿔요.' },
      hintDelete: { label: '삭제', long: '무언가를 지워요.' },
      hintDestructive: { label: '파괴적', long: '이 변경은 여기서 되돌릴 수 없어요.' },
      hintIdempotent: { label: '반복 가능', long: '두 번 실행해도 한 번 실행한 것과 같아요.' },
      hintOpenWorld: { label: '외부', long: '이 앱 밖의 무언가에 접근해요.' }
    }
  },
}
