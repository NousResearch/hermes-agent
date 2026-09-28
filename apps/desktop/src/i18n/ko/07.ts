// ko/07.ts — Korean translation of the `vault, notifications, sections, searchPlaceholder, modeOptions` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko07: TranslationOverrides = {
  settings: {
vault: {
      title: '비밀번호 및 로그인',
      blurb:
        '"GitHub에 로그인해 줘"라고 말하면 에이전트가 대신 로그인해 줘요. 처음 마주친 로그인 페이지에서는 그 자리에서 로그인 정보를 물어보고, 그다음부터는 알아서 처리해요. 비밀번호는 이 컴퓨터에서 암호화되어 페이지에 바로 입력되며, 모델은 절대 볼 수 없어요.',
      count: n => `${n}개 저장됨`,
      loadFailed: '보관함 항목을 불러올 수 없어요',
      empty: '아직 저장된 항목이 없어요',
      emptyDesc:
        "여기에 직접 추가하지 않아도 괜찮아요. 에이전트에게 사이트 로그인을 요청하면 그 자리에서 로그인 정보를 한 번만 물어봐요. 미리 입력해 두고 싶다면 추가를 사용하세요.",
      add: '추가',
      addTitle: '로그인, 카드 또는 주소 추가',
      addDescription: '이 컴퓨터에 암호화되어 저장돼요. 에이전트는 비밀번호를 볼 수 없어요.',
      added: '저장했어요.',
      adding: '저장 중…',
      addConfirm: '저장',
      kindField: '종류',
      kinds: { login: '로그인', payment: '결제 카드', address: '주소' },
      labelField: '이름',
      labelPlaceholder: '예: GitHub 업무 계정',
      labelRequired: '이름을 입력해 주세요.',
      originField: '사이트 주소',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: 'https://example.com 형식의 올바른 URL을 입력해 주세요.',
      identifierTypeField: '식별자 유형',
      identifierTypes: { email: '이메일', phone: '전화번호', username: '사용자 이름' },
      identifierField: '식별자',
      identifierShown: identifier => identifier,
      passwordField: '비밀번호',
      loginFieldsRequired: '식별자와 비밀번호를 입력해 주세요.',
      cardNumberField: '카드 번호',
      cardNameField: '카드에 표시된 이름',
      expMonthField: '만료 월',
      expYearField: '만료 연도',
      cvcField: 'CVC',
      postalField: '우편번호',
      addressLine1Field: '주소 1',
      addressLine2Field: '주소 2',
      cityField: '도시',
      stateField: '주 / 지역',
      countryField: '국가',
      optional: '(선택 사항)',
      createdOn: date => `${date}에 추가함`,
      deleteAction: '저장된 항목 제거',
      otpField: '인증 앱 키',
      otpPlaceholder: 'Base32 시크릿 또는 otpauth:// 링크',
      otpHint: '사이트에서 2단계 인증을 켤 때 표시되는 "설정 키"예요. 이 키를 저장해 두면 Hermes가 인증 코드를 직접 생성해요.',
      twoFactorBadge: '2FA 자동',
      deleteTitle: '이 항목을 삭제할까요?',
      deleteDescription: label => `"${label}" 항목이 제거돼요. 되돌릴 수 없어요.`,
      deleteConfirm: '삭제',
      sources: {
        title: '비밀번호 관리자',
        blurb:
          '설치된 비밀번호 관리자는 자동으로 인식돼요. 에이전트가 처음 해당 관리자의 로그인 정보가 필요할 때 잠금 해제를 요청해요(세션당 한 번). 세션 토큰만 메모리에 남고, 에이전트는 마스터 비밀번호나 로그인 정보를 볼 수 없어요.',
        toggleFailed: '비밀번호 관리자를 업데이트할 수 없어요',
        notInstalled: name =>
          `${name} 명령줄 도구가 감지되지 않았어요. 설치하고 로그인하면 Hermes가 자동으로 인식해요.`,
        disabledDesc: '감지되었지만 Hermes에서 꺼져 있어요.',
        lockedDesc: '감지되었어요. 로그인 정보가 필요할 때 에이전트가 잠금 해제를 요청해요. 지금 잠금 해제할 수도 있어요.',
        unlockedDesc: '이번 세션 동안 잠금 해제됨. 30분 동안 사용하지 않거나 Hermes를 닫으면 자동으로 잠겨요.',
        statusLocked: '잠김',
        statusNotDetected: '감지되지 않음',
        statusOff: '꺼짐',
        statusUnlocked: '잠금 해제됨',
        unlock: '잠금 해제',
        unlocking: '잠금 해제 중…',
        lock: '잠금',
        unlocked: name => `${name} 잠금을 이번 세션 동안 해제했어요.`,
        unlockTitle: name => `${name} 잠금 해제`,
        unlockDescription:
          '마스터 비밀번호를 입력하세요. 이 컴퓨터의 비밀번호 관리자에게 전달된 뒤 폐기되며, 저장되거나 기록되지 않고 에이전트에게도 표시되지 않아요.',
        masterPasswordPlaceholder: '마스터 비밀번호'
      }
    },
notifications: {
      title: '알림',
      intro: 'OS 알림이에요(앱 내 토스트 아님). 기기별로 설정돼요.',
      enableAll: '알림 사용',
      enableAllDesc: '끄면 아래 모든 알림이 울리지 않아요.',
      focusedHint: '완료 알림은 Hermes가 백그라운드에 있을 때만 울려요.',
      kinds: {
        approval: {
          label: '승인 필요',
          description: '명령이 승인 또는 거부를 기다리고 있어요.'
        },
        input: {
          label: '입력 필요',
          description: 'Hermes가 질문을 했거나 비밀번호 또는 시크릿이 필요해요.'
        },
        turnDone: {
          label: '응답 준비됨',
          description: 'Hermes가 백그라운드에 있는 동안 턴이 끝났어요.'
        },
        turnError: {
          label: '턴 실패',
          description: '백그라운드 턴 오류예요.'
        },
        backgroundDone: {
          label: '백그라운드 작업 완료',
          description: '백그라운드로 실행한 터미널 명령이 완료됐어요.'
        },
        credits: {
          label: '크레딧 알림',
          description: '크레딧 접근이 일시 중지되거나 복구됐어요.'
        },
        plugin: {
          label: '플러그인 알림',
          description: 'Hermes가 백그라운드에 있는 동안 데스크톱 플러그인이 알림을 보냈어요.'
        }
      },
      test: '테스트 알림 보내기',
      testTitle: 'Hermes',
      testBody: '알림이 정상 작동해요.',
      testSent: '테스트를 보냈어요. 아무것도 표시되지 않으면 OS 알림 권한과 집중 모드/방해 금지 설정을 확인해 주세요.',
      testUnsupported: '이 시스템은 기본 알림을 지원하지 않아요.',
      completionSoundTitle: '완료음',
      completionSoundDesc: '에이전트 턴이 끝나면 재생돼요. 프리셋을 고르고 여기서 미리 들어보세요.',
      completionSoundPreview: '미리 듣기'
    },
sections: {
      model: '모델',
      chat: '채팅',
      appearance: '모양',
      workspace: '작업 공간',
      safety: '안전',
      memory: '메모리 및 컨텍스트',
      voice: '음성',
      advanced: '고급'
    },
searchPlaceholder: {
      about: 'Hermes 데스크톱 정보',
      config: '설정 검색...',
      gateway: '게이트웨이 연결...',
      keys: 'API 키 검색...',
      mcp: 'MCP 서버 검색...',
      sessions: '보관된 세션 검색...'
    },
modeOptions: {
      light: { label: '라이트', description: '밝은 데스크톱 화면' },
      dark: { label: '다크', description: '눈부심이 적은 작업 공간' },
      system: { label: '시스템', description: 'OS 모양 설정 따르기' }
    },
  },
}
