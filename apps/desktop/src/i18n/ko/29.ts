// ko/29.ts — Korean translation of the `approval, clarify, catalogInstall, mcpSetup, tool, prompts` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko29: TranslationOverrides = {
  assistant: {
    approval: {
      gatewayDisconnected:
        '지금 Hermes가 오프라인 상태예요. 명령어는 승인 시간 초과 전까지 응답을 기다리고 있어요. 다시 연결한 후 다시 보내주세요.',
      sendFailed: '응답을 전송하지 못했어요',
      reconnect: '다시 연결',
      timedOutSystemLine:
        '승인 시간이 초과되어 명령어가 실행되지 않았어요. Hermes에게 다시 시도하도록 요청하거나 설정 → 보안 → 승인 시간 초과에서 제한 시간을 늘려주세요.',
      openSafetySettings: '보안 설정 열기',
      run: '실행',
      command: '명령어',
      moreOptions: '추가 승인 옵션',
      allowSession: '이 세션에서 허용',
      alwaysAllowMenu: '항상 허용…',
      jumpToApproval: '승인 필요',
      reject: '거부',
      alwaysTitle: '이 명령어를 항상 허용할까요?',
      alwaysDescription: pattern =>
        `“${pattern}” 패턴을 영구 허용 목록(~/.hermes/config.yaml)에 추가해요. 이번 세션이나 이후 세션에서 이와 같은 명령어에 대해 Hermes가 다시 묻지 않아요.`,
      alwaysAllow: '항상 허용'
    },
    clarify: {
      notReady: '확인 요청이 아직 준비되지 않았어요',
      gatewayDisconnected: '지금 Hermes가 오프라인 상태예요. 다시 연결한 후 다시 보내주세요.',
      sendFailed: '확인 응답을 전송하지 못했어요',
      loadingQuestion: '질문 불러오는 중…',
      other: '기타 (직접 입력)',
      placeholder: '답변을 입력하세요…',
      skip: '건너뛰기',
      skipped: '건너뜀',
      continueLabel: '계속',
      confirmAndContinueLabel: '확인 후 계속',
      answeredBadge: '답변 완료',
      questionProgress: (answered, total) => `${total}개 중 ${answered}개 답변 완료`,
      lateAnswer: (question, choice) => `Re: "${question}" — 내 답변: ${choice}`,
      lateAnswerTip: '이 답변을 후속 메시지로 작성',
      lateAnswerHint: '이 프롬프트는 더 이상 대기 중이지 않아요. 옵션을 선택하여 후속 메시지로 작성하세요.'
    },
    catalogInstall: {
      preparing: '설치 준비 중…',
      install: '설치',
      advanced: '고급',
      skip: '건너뛰기',
      installing: '설치 중…',
      installed: '설치됨',
      notInstalled: '설치되지 않음',
      failed: '실패',
      showNames: '이름 표시',
      hideNames: '이름 숨기기',
      skill: name => `스킬 ${name}`,
      kind: { plugin: '플러그인', skill: '스킬' },
      tier: { official: '공식', community: '커뮤니티' },
      targetProfile: profile => `${profile} 프로필에 설치돼요`,
      sendFailed: '응답을 전송하지 못했어요. 다시 시도해 주세요.',
      commitLabel: '커밋',
      subdirLabel: '폴더',
      securityHeading: '보안',
      scan: { passed: '검사 통과', warnings: '경고 발견됨', failed: '검사 실패' },
      requirementsLabel: '필수 요구사항',
      credentialsHeading: '자격 증명'
    },
    mcpSetup: {
      installTitle: 'MCP 서버 추가',
      enableTitle: 'MCP 서버 활성화',
      authorizeTitle: 'MCP 서버 승인',
      installAction: '설치',
      enableAction: '활성화',
      authorizeAction: '승인',
      installed: server => `${server} 설치됨`,
      enabled: server => `${server} 활성화됨`,
      authorized: server => `${server} 승인됨`,
      failed: server => `${server} 설정 실패`,
      toolCount: count => (count === 1 ? '도구 1개' : `도구 ${count}개`),
      envRequired: '필수 자격 증명을 먼저 입력해 주세요',
      sendFailed: 'MCP 설정 응답을 전송하지 못했어요',
      reloadFailed: '서버가 저장되었지만 MCP 도구를 다시 불러오지 못했어요 — 다음 세션에서 로드돼요',
      gatewayDisconnected: '지금 Hermes가 오프라인 상태예요. 다시 연결한 후 다시 보내주세요.'
    },
    tool: {
      copyCode: '코드 복사',
      renderingImage: '이미지 렌더링 중',
      copyOutput: '출력 복사',
      copyCommand: '명령어 복사',
      copyContent: '내용 복사',
      copyUrl: 'URL 복사',
      copyResults: '결과 복사',
      copyQuery: '쿼리 복사',
      copyFile: '파일 복사',
      copyPath: '경로 복사',
      failedCalls: (count: number) => `도구 호출 ${count}개 실패`,
      skillActivity: {
        loading: '스킬 불러오는 중',
        loaded: '스킬 불러옴',
        loadFailed: '스킬 불러오기 실패',
        readingResource: '스킬 리소스 읽는 중',
        readResource: '스킬 리소스 읽음',
        resourceFailed: '스킬 리소스 읽기 실패',
        listing: '스킬 목록 조회 중',
        listed: '스킬 목록 조회 완료',
        listFailed: '스킬 목록 조회 실패',
        unavailable: '스킬 결과 사용 불가'
      },
      outputAlt: '도구 출력',
      rawResponse: '원본 응답',
      copyActivity: '활동 복사',
      recoveredOne: '1개 실패 단계 후 복구됨',
      recoveredMany: count => `${count}개 실패 단계 후 복구됨`,
      failedOne: '1개 단계 실패',
      failedMany: count => `${count}개 단계 실패`,
      statusRunning: '실행 중',
      statusError: '오류',
      statusRecovered: '복구됨',
      statusDone: '완료',
      resultUnavailable: '결과 사용 불가',
      resultInterrupted: '중단됨',
      memoryWriteNoted: '기억 저장 완료',
      actions: {
        read: '읽음',
        reading: '읽는 중',
        opened: '엶',
        opening: '여는 중',
        failedToOpen: '열기 실패',
        searched: '검색함',
        searching: '검색 중',
        ran: '실행함',
        running: '실행 중',
        ranCode: '코드 실행함',
        runningCode: '스크립트 실행 중'
      },
      prefixes: {
        browser: '브라우저',
        web: '웹'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `${prefix} ${action} 실행 중`,
        runningTool: action => `${action} 실행 중`
      },
      titles: {
        browser_click: { done: '페이지 요소 클릭함', pending: '페이지 요소 클릭 중', pendingAction: '클릭 중' },
        browser_fill: { done: '폼 필드 입력함', pending: '폼 필드 입력 중', pendingAction: '입력 중' },
        browser_navigate: { done: '페이지 엶', pending: '페이지 여는 중', pendingAction: '여는 중' },
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
        browser_type: { done: '페이지에 텍스트 입력함', pending: '페이지에 텍스트 입력 중', pendingAction: '입력 중' },
        clarify: { done: '질문함', pending: '질문하는 중', pendingAction: '질문 중' },
        cronjob: { done: '크론 작업', pending: '크론 작업 예약 중', pendingAction: '예약 중' },
        edit_file: { done: '파일 수정함', pending: '파일 수정 중', pendingAction: '수정 중' },
        execute_code: { done: '코드 실행함', pending: '스크립트 실행 중', pendingAction: '스크립트 실행 중' },
        image_generate: { done: '이미지 생성함', pending: '이미지 생성 중', pendingAction: '생성 중' },
        list_files: { done: '파일 목록 조회함', pending: '파일 목록 조회 중', pendingAction: '목록 조회 중' },
        memory: { done: '기억에 저장함', pending: '기억에 저장 중', pendingAction: '저장 중' },
        patch: { done: '파일 패치함', pending: '파일 패치 중', pendingAction: '패치 중' },
        read_file: { done: '파일 읽음', pending: '파일 읽는 중', pendingAction: '읽는 중' },
        search_files: { done: '파일 검색함', pending: '파일 검색 중', pendingAction: '검색 중' },
        session_search_recall: {
          done: '세션 기록 검색함',
          pending: '세션 기록 검색 중',
          pendingAction: '검색 중'
        },
        terminal: { done: '명령어 실행함', pending: '명령어 실행 중', pendingAction: '실행 중' },
        todo: { done: '할 일 업데이트함', pending: '할 일 업데이트 중', pendingAction: '업데이트 중' },
        vision_analyze: { done: '이미지 분석함', pending: '이미지 분석 중', pendingAction: '분석 중' },
        web_extract: { done: '웹페이지 읽음', pending: '웹페이지 읽는 중', pendingAction: '읽는 중' },
        web_search: { done: '웹 검색함', pending: '웹 검색 중', pendingAction: '검색 중' },
        write_file: { done: '파일 수정함', pending: '파일 수정 중', pendingAction: '수정 중' }
      }
    },
  },
  prompts: {
    gatewayDisconnected: '지금 Hermes가 오프라인 상태예요. 다시 연결한 후 다시 보내주세요.',
    reconnect: '다시 연결',
    sudoSendFailed: 'sudo 비밀번호를 전송하지 못했어요',
    secretSendFailed: '시크릿을 전송하지 못했어요',
    sudoTitle: '관리자 비밀번호',
    sudoDesc:
      'sudo 비밀번호를 입력하기 전에 명령어를 검토해 주세요. 비밀번호는 실행 중인 에이전트에게 전달되며 이번 세션 동안 캐시돼요.',
    sudoCommandUnavailable:
      '이 에이전트가 명령어를 제공하지 않았어요. 대화에서 확인할 수 없다면 취소해 주세요.',
    sudoInstallDesc:
      'Hermes가 게이트웨이 호스트에 봇 화면 패키지(TigerVNC + Xfce)를 설치하려면 sudo 비밀번호가 필요해요. 해당 호스트로만 전송돼요.',
    sudoPlaceholder: 'sudo 비밀번호',
    secretTitle: '시크릿 필요',
    secretDesc: '계속하려면 Hermes에 자격 증명이 필요해요.',
    secretPlaceholder: '시크릿 값',
    vaultUnlockSendFailed: '마스터 비밀번호를 전송하지 못했어요',
    vaultUnlockTitle: name => `${name} 잠금 해제`,
    vaultUnlockDesc: name =>
      `에이전트가 ${name}에 저장된 로그인 정보로 사이트에 로그인하려고 해요. 이번 세션 동안 잠금을 해제하려면 마스터 비밀번호를 입력하세요. 이 기기의 ${name}에 직접 전달되며 에이전트에게 저장되거나 표시되지 않아요.`,
    vaultUnlockPlaceholder: '마스터 비밀번호',
    vaultUnlockKeepLocked: '잠긴 상태 유지',
    vaultUnlockConfirm: '잠금 해제',
    vaultSaveSendFailed: '로그인 정보를 저장하지 못했어요',
    vaultSaveTitle: site => `${site} 로그인 정보를 저장할까요?`,
    vaultSaveDesc: origin =>
      `Hermes가 ${origin}의 로그인 페이지에 도달했으나 저장된 로그인 정보가 없어요. 여기에 한 번 입력하면 이 기기에서 암호화되어 페이지에 자동으로 입력되며 모델은 비밀번호를 전혀 볼 수 없어요.`,
    vaultSaveIdentifierLabel: '이메일 또는 사용자 이름',
    vaultSaveIdentifierPlaceholder: 'you@example.com',
    vaultSavePasswordPlaceholder: '비밀번호',
    vaultSaveFootnote: '설정 → 비밀번호 및 로그인에서 저장된 로그인을 관리하세요.',
    vaultSaveDecline: '저장 안 함',
    vaultSaveConfirm: '저장 및 로그인',
    vaultCodeSendFailed: '코드를 전송하지 못했어요',
    vaultCodeTitle: site => `${site} 인증 코드`,
    vaultCodeDesc: site =>
      `${site}에서 일회용 코드(문자 메시지, 이메일 또는 OTP 앱)를 요청하고 있어요. 여기에 입력하면 Hermes가 페이지에 입력하며 모델은 코드를 전혀 볼 수 없어요.`,
    vaultCodeLabel: '코드',
    vaultCodeFootnote:
      '팁: 설정 → 비밀번호 및 로그인에서 이 로그인에 OTP 키를 함께 저장하면 Hermes가 코드를 자동으로 입력해 줘요.',
    vaultCodeSkip: '건너뛰기',
    vaultCodeConfirm: '코드 입력'
  },
}
