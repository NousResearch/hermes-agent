// ko/09.ts — Korean translation of the `customEndpoints, computerUse, about, config, hudModifier, screenshot` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko09: TranslationOverrides = {
  settings: {
customEndpoints: {
      active: '사용 중',
      apiKeySet: 'API 키 설정됨',
      use: '사용',
      editTitle: '엔드포인트 편집',
      addTitle: '엔드포인트 추가',
      fields: {
        name: '이름',
        providerId: '제공자 ID',
        endpointUrl: '엔드포인트 URL',
        defaultModel: '기본 모델',
        context: '컨텍스트',
        apiKey: 'API 키',
        apiKeyNewPlaceholder: '비워 두면 현재 키를 유지해요',
        apiKeyPlaceholder: '선택 사항',
        useNewChats: '새 채팅에 사용',
        discoverModels: '모델 찾기'
      },
      test: '테스트',
      save: '저장',
      newEndpoint: '새 엔드포인트',
      apiMode: 'API 모드',
      autoDetect: '자동 감지',
      couldNotLoad: '사용자 지정 엔드포인트를 불러올 수 없어요',
      endpointSaved: '사용자 지정 엔드포인트를 저장했어요.',
      saveFailed: '저장 실패',
      endpointReachable: '엔드포인트에 연결할 수 있어요.',
      endpointReachableTransport: transport => `엔드포인트에 연결할 수 있어요 (${transport} 경로로 응답).`,
      endpointReachableModels: (reachable, count) => `${reachable} 모델 ${count}개를 찾았어요.`,
      endpointValidationFailed: '엔드포인트 검증에 실패했어요.',
      validationFailed: '검증 실패',
      activationFailed: '활성화 실패',
      deleteConfirm: name => `${name}을(를) 삭제할까요?`,
      deleteFailed: '삭제 실패',
      title: '사용자 지정 엔드포인트',
      deleteEndpoint: '엔드포인트 삭제',
      emptyDescription: '아래에서 OpenAI 호환 엔드포인트를 추가할 수 있어요.',
      emptyTitle: '사용자 지정 엔드포인트가 없어요',
      namePlaceholder: 'Axet Proxy',
      contextPlaceholder: '자동'
    },
computerUse: {
      accessibility: '손쉬운 사용',
      screenRecording: '화면 기록',
      driverHealth: '드라이버 상태'
    },
about: {
      updates: '업데이트'
    },
config: {
      minimizeToTrayTitle: '트레이로 최소화',
      minimizeToTrayDesc:
        '창을 최소화하거나 기본 창을 닫으면 시스템 트레이(macOS에서는 메뉴 막대)에 숨기고 Hermes를 계속 실행해요. 종료하려면 트레이 메뉴의 Hermes 종료를 사용하거나 Cmd+Q를 누르세요. 기본값은 꺼짐이며 이 기기에만 적용돼요.',
      minimizeToTrayUnavailable:
        '시스템 트레이를 사용할 수 없어요. 창은 평소처럼 최소화되고 닫혀요. 다시 시도하려면 이 설정을 껐다가 켜세요.',
      none: '없음',
      noneParen: '(없음)',
      builtinOnly: '내장만',
      notSet: '설정 안 됨',
      commaSeparated: '쉼표로 구분한 값',
      searchPlaceholder: '검색…',
      noResults: '검색 결과가 없어요',
      systemDefault: '시스템 기본값',
      loading: 'Hermes 구성을 불러오는 중...',
      emptyTitle: '설정할 항목이 없어요',
      emptyDesc: '이 섹션에는 조정할 수 있는 설정이 없어요.',
      failedLoad: '설정을 불러오지 못했어요',
      autosaveFailed: '자동 저장 실패',
      imported: '구성을 가져왔어요',
      invalidJson: '잘못된 구성 JSON',
      toolsetsWipeConfirm:
        '활성화된 도구 세트를 모두 제거할까요? 그러면 메모리, 터미널, 웹 검색, 위임 및 대부분의 다른 도구가 다시 활성화할 때까지 비활성화돼요.',
      keepAwakeTitle: '컴퓨터를 깨어 있게 유지',
      keepAwakeDesc: '오래 걸리거나 밤새 실행되는 작업이 계속되도록 이 기기가 잠들지 않게 해요. 화면은 계속 어두워질 수 있어요.',
      disableF12Title: 'F12 개발자 도구 사용 안 함',
      disableF12Desc: 'F12로 개발자 도구를 여는 것을 막아요. Ctrl+Shift+I (Mac에서는 Cmd+Opt+I)는 계속 작동해요.',
      alwaysExternalLinksTitle: '항상 외부 브라우저에서 링크 열기',
      alwaysExternalLinksDesc:
        '클릭하는 모든 링크를 앱 내 브라우저 대신 시스템 브라우저에서 열어요. 마우스 오른쪽 버튼 메뉴의 \"앱 내 브라우저에서 열기\"는 계속 작동해요.',
      attachmentSizeTitle: '최대 미리보기 / 이미지 불러오기 크기',
      attachmentSizeDesc:
        '데스크톱이 미리보기와 이미지 첨부를 위해 로컬 파일을 불러오는 최대 크기(MB)예요. 기본값은 16이에요. 원격 비이미지 첨부에는 별도의 256MB 제한이 적용돼요. 이 값을 너무 높게 설정하면 파일 전체가 메모리에 올라가 앱이 멈추거나 종료될 수 있어요.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: '최대 미리보기 / 이미지 불러오기 크기(MB)',
      voiceShortcutHintTitle: '음성 녹음 단축키',
      voiceShortcutHintDesc:
        '설정 → 키보드 단축키에서 음성 녹음 단축키(\"음성 대화 시작 / 중지\")를 설정하세요. voice.record_key 구성 값은 CLI와 TUI에만 적용돼요.',
      showOptions: '옵션 표시'
    },
hudModifier: {
      title: '탭해서 HUD 열기',
      description:
        'Mac에서는 ⌘ + Option을, Windows/Linux에서는 Ctrl + Alt를 눌렀다 떼면 어떤 앱에서든 HUD를 앞으로 불러와요. 기본값은 꺼짐이며 이 기기에만 적용돼요.',
      permission:
        '시스템 설정 → 개인 정보 보호 및 보안 → 입력 모니터링에서 Hermes를 허용한 뒤 다시 시도하세요. 이 제스처는 키 입력을 기록하거나 화면을 캡처하지 않아요.',
      unavailable:
        'HUD 제스처 도우미를 시작하지 못했거나 예기치 않게 중지됐어요. 다시 시도하거나 Hermes를 재시작하세요. 기존 HUD 단축키는 Hermes 안에서 계속 작동해요.',
      missingHelper:
        '이 Hermes 설치에는 HUD 제스처 도우미가 없어요. Hermes를 업데이트하거나 다시 설치한 뒤 다시 시도하세요.',
      unsupportedSession:
        '이 데스크톱 세션은 전역 보조 키 탭을 지원하지 않아요. Linux는 X11이 필요하며 Wayland는 지원하지 않아요.'
    },
screenshot: {
      enabledTitle: '스크린샷 단축키',
      enabledDesc:
        '어떤 앱에서든 Command 키 두 개를 함께 누르면 맨 앞 창을 캡처해 현재 Hermes 초안에 첨부해요. 자동으로 전송하지 않아요. 기본값은 꺼짐이며 이 Mac에만 적용돼요. 창 내용이 민감할 수 있으니 보내기 전에 첨부 파일을 확인하세요.',
      statusTitle: '스크린샷 단축키 상태',
      checking: '스크린샷 단축키를 확인하는 중…',
      disabled: '스크린샷 단축키가 꺼져 있어요.',
      starting: '단축키 감지기를 시작하는 중이에요. 아직 준비되지 않았어요.',
      ready: '단축키가 준비됐어요. 스크린샷은 전송하지 않고 현재 초안에 첨부돼요.',
      inputPermission:
        '입력 모니터링 권한이 있으면 다른 앱이 활성화된 상태에서도 Hermes가 Command 키 두 개를 감지할 수 있어요. 시스템 설정 → 개인 정보 보호 및 보안 → 입력 모니터링에서 Hermes를 허용한 뒤 여기로 돌아와 다시 시도하세요.',
      screenPermission:
        '화면 기록 권한이 있으면 이 단축키를 사용할 때 Hermes가 맨 앞 앱 창을 캡처할 수 있어요. 시스템 설정 → 개인 정보 보호 및 보안 → 화면 기록에서 Hermes를 허용한 뒤 여기로 돌아와 다시 시도하세요. macOS가 요청하면 Hermes를 재시작하세요.',
      openSettings: '시스템 설정 열기',
      retry: '다시 시도',
      unavailable: '스크린샷 단축키를 사용할 수 없어요. 다시 시도하거나 꺼 주세요.',
      errorTitle: '스크린샷 단축키 오류',
      loadFailed: '단축키 상태를 읽을 수 없어요. 다시 시도해서 현재 설정을 확인하세요.',
      saveFailed: '단축키 변경을 확인할 수 없어요. 다시 시도해서 현재 설정을 확인하세요.',
      permissionFailed: '시스템 설정을 열 수 없어요. 개인 정보 보호 및 보안을 직접 연 뒤 다시 시도하세요.',
      captureFailed: '맨 앞 창을 캡처할 수 없어요. 첨부되거나 전송된 것은 없어요.',
      contextChanged: '캡처하는 동안 현재 초안이 변경됐어요. 스크린샷은 첨부되거나 전송되지 않았어요.'
    },
quickEntry: {
      enabledTitle: '빠른 입력',
      enabledDesc:
        '전역 단축키로 어디서든 작은 입력창을 불러와 Hermes를 열지 않고 프롬프트를 보낼 수 있어요.',
      shortcutTitle: '빠른 입력 단축키',
      shortcutDesc: '수정 키가 하나 이상 필요해요. 예: CommandOrControl+Shift+Space.',
      active: '단축키가 활성화됐어요.',
      takenBy: '다른 앱이 이미 이 단축키를 사용하고 있어요 — 다른 단축키를 선택하세요.',
      invalidShortcut: '유효한 단축키가 아니에요. 수정 키를 하나 이상 포함하세요.'
    },
credentials: {
      pasteKey: '키 붙여넣기',
      pasteLabelKey: label => `${label} 키 붙여넣기`,
      optional: '선택 사항',
      enterValueFirst: '먼저 값을 입력하세요.',
      couldNotSave: '자격 증명을 저장할 수 없어요.',
      remove: '제거',
      getKey: '키 발급받기',
      saving: '저장 중'
    },
envActions: {
      actions: '작업',
      manageInKeys: 'API 키에서 관리',
      docs: '문서',
      hideValue: '값 숨기기',
      revealValue: '값 표시',
      replace: '교체',
      set: '설정',
      clear: '지우기'
    },
  },
}
