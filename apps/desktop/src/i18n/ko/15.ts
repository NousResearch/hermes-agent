// ko/15.ts — Korean translation of the `providers, sessions, toolsets` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko15: TranslationOverrides = {
  settings: {
    providers: {
      connectAccount: '계정 연결',
      haveApiKey: '대신 API 키를 사용하시겠어요?',
      intro:
        '구독 계정으로 로그인하세요 — 복사할 API 키가 필요 없어요. Hermes가 앱 내에서 바로 브라우저 로그인을 진행해 줘요.',
      connected: '연결됨',
      collapse: '접기',
      connectAnother: '다른 제공자 연결',
      otherProviders: '기타 제공자',
      disconnect: '연결 해제',
      disconnectInTerminal: '연결 해제 (터미널에서 제거 명령 실행)',
      removeConfirm: provider => `${provider}을(를) 제거할까요?`,
      removeExternalGeneric: provider => `${provider}은(는) 자체 CLI로 관리돼요 — 해당 CLI에서 제거해 주세요.`,
      removeKeyManaged: provider => `${provider}은(는) API 키로 설정되어 있어요. API 키 설정에서 제거해 주세요.`,
      removeTerminalConfirm: (provider, command) =>
        `${provider} 연결을 해제할까요? 터미널에서 "${command}"을(를) 실행하여 인증 정보를 삭제해요.`,
      removeTerminalRunning: provider => `터미널에서 ${provider} 연결 해제 실행 중…`,
      removedTitle: '계정 제거됨',
      removedMessage: provider => `${provider}이(가) 제거되었어요.`,
      failedRemove: provider => `${provider}을(를) 제거할 수 없어요`,
      noProviderKeys: '사용 가능한 제공자 API 키가 없어요.',
      searchKeys: '제공자 검색…',
      noKeysMatch: '검색과 일치하는 제공자가 없어요.',
      localEndpoint: {
        title: '로컬 / 커스텀 엔드포인트',
        description: 'Hermes가 OpenAI 호환 엔드포인트(Zyphra, vLLM, llama.cpp, Ollama 등)를 가리키도록 설정하세요.'
      },
      loading: '제공자 불러오는 중...'
    },
    sessions: {
      loading: '보관된 세션 불러오는 중…',
      archivedTitle: '보관된 세션',
      archivedIntro:
        '보관된 대화는 사이드바에서 숨겨지지만 모든 메시지는 유지돼요. 사이드바에서 Alt/⌥+Shift를 누른 채 대화를 클릭하면 보관할 수 있어요.',
      emptyArchivedTitle: '보관된 항목 없음',
      emptyArchivedDesc: '대화를 보관하면 여기에 표시돼요.',
      unarchive: '보관 해제',
      deletePermanently: '영구 삭제',
      messages: count => `메시지 ${count}개`,
      restored: '복원됨',
      deleteConfirm: title => `"${title}"을(를) 영구적으로 삭제할까요? 이 작업은 되돌릴 수 없어요.`,
      autoArchiveTitle: '오래된 대화 자동 보관',
      autoArchiveDesc:
        '한동안 사용하지 않은 대화를 자동으로 보관해요. 고정된 대화는 절대 보관되지 않으며, 아무것도 삭제되지 않고 보관된 대화가 이곳으로 이동할 뿐이에요.',
      autoArchiveDaysLabel: '보관 기준',
      autoArchiveDaysUnit: '일 동안 미사용 시',
      autoArchiveFailed: '자동 보관 설정을 업데이트할 수 없어요',
      defaultDirTitle: '기본 프로젝트 디렉터리',
      defaultDirDesc:
        '다른 폴더를 선택하지 않는 한 새 세션이 이 폴더에서 시작돼요. 설정하지 않으면 홈 디렉터리를 사용해요.',
      defaultDirUpdated: '기본 프로젝트 디렉터리가 업데이트되었어요 — 새 대화(Ctrl/⌘+N)를 시작하면 적용돼요',
      defaultsTo: label => `기본값: ${label}.`,
      change: '변경',
      choose: '선택',
      clear: '지우기',
      notSet: '설정되지 않음',
      failedLoad: '보관된 세션을 불러올 수 없어요',
      unarchiveFailed: '보관 해제 실패',
      deleteFailed: '삭제 실패',
      updateDirFailed: '기본 디렉터리를 업데이트할 수 없어요',
      clearDirFailed: '기본 디렉터리를 지울 수 없어요'
    },
    toolsets: {
      loadingConfig: '설정 불러오는 중',
      savedTitle: '인증 정보 저장됨',
      savedMessage: key => `${key}이(가) 업데이트되었어요.`,
      removedTitle: '인증 정보 제거됨',
      removedMessage: key => `${key}이(가) 제거되었어요.`,
      failedSave: key => `${key} 저장 실패`,
      failedRemove: key => `${key} 제거 실패`,
      failedReveal: key => `${key} 표시 실패`,
      removeConfirm: key => `.env에서 ${key}을(를) 제거할까요?`,
      set: '설정됨',
      notSet: '설정되지 않음',
      selectedTitle: '제공자 선택됨',
      selectedMessage: provider => `이제 ${provider}이(가) 활성화되었어요.`,
      failedSelect: provider => `${provider} 선택 실패`,
      failedLoad: '도구 설정을 불러오지 못했어요',
      noProviderOptions: '이 도구 세트에는 제공자 옵션이 없어요 — 활성화하면 현재 설정으로 바로 작동해요.',
      noProviders: '현재 이 도구 세트에 사용할 수 있는 제공자가 없어요.',
      ready: '준비 완료',
      needsSignIn: '로그인 필요',
      needsSetup: '설정 필요',
      activeBackend: '활성',
      activeBackendHint: '현재 활성화된 백엔드예요',
      useBackend: '이 백엔드 사용',
      nousIncluded: 'Nous 구독에 포함되어 있어요 — Nous 계정으로 로그인하여 활성화하세요.',
      nousAuthNeededTitle: 'Nous 계정으로 로그인',
      nousAuthNeededMessage: provider =>
        `${provider}이(가) 저장되었지만 Nous 계정으로 로그인해야 작동해요.`,
      nousAuthSignIn: '로그인',
      nousAuthDoneTitle: 'Nous 계정 연결됨',
      nousAuthDoneMessage: '구독 백엔드가 이제 활성화되었어요.',
      nousAuthFailed: 'Nous 로그인을 완료하지 못했어요',
      nousAuthFailedMessage: '다시 시도해 주세요.',
      nousAuthTryAgain: '다시 시도',
      noApiKeyRequired: 'API 키가 필요하지 않아요.',
      postSetupHint: step =>
        `이 백엔드는 최초 1회 설치(${step})가 필요해요. 이 컴퓨터에서 실행되며 몇 분 정도 걸릴 수 있어요.`,
      postSetupInstalledHint: '설치되었어요. 문제가 발생한 경우에만 설정을 다시 실행하세요.',
      postSetupRun: '설정 실행',
      postSetupRerun: '설정 다시 실행',
      postSetupInstalled: '설치됨',
      postSetupRunning: '설치 중…',
      postSetupStarting: '시작 중…',
      postSetupCompleteTitle: '설정 완료',
      postSetupCompleteMessage: step => `${step}이(가) 설치되었어요.`,
      postSetupErrorTitle: '오류와 함께 설정 완료됨',
      postSetupErrorMessage: step =>
        `${step} 설정을 완료하지 못했어요. 로그를 열어 원인을 확인한 후 설정을 다시 실행해 주세요.`,
      postSetupOpenLogs: '로그 열기',
      postSetupRunAgain: '다시 실행',
      postSetupFailed: step => `${step} 설정 실행 실패`,
      webSearchActive: backend => `검색: ${backend}`,
      webExtractActive: backend => `추출: ${backend}`,
      webCapabilityUnset: '설정되지 않음',
      webUseForSearch: '검색에 사용',
      webUseForExtract: '추출에 사용',
      webUsedForSearch: '검색 백엔드',
      webUsedForExtract: '추출 백엔드',
      webCapabilitySelectedMessage: (provider, capability) => `이제 ${provider}이(가) 웹 ${capability}을(를) 처리해요.`,
      failedSelectCapability: provider => `${provider} 설정 실패`,
      loadingModels: '모델 카탈로그 불러오는 중...',
      modelSectionTitle: '모델',
      modelCount: count => `모델 ${count}개`,
      modelInUse: '사용 중',
      modelDefault: '기본값',
      modelInactiveHint: '모델을 변경하려면 먼저 이 백엔드를 선택하세요.',
      modelSelectedTitle: '모델 선택됨',
      modelSelectedMessage: model => `새 세션에 ${model}이(가) 적용돼요.`,
      failedSelectModel: model => `${model} 선택 실패`,
      terminalBackend: {
        sectionTitle: '실행 백엔드',
        loading: '실행 백엔드 확인 중…',
        failedLoad: '터미널 백엔드를 불러올 수 없어요',
        ready: '준비 완료',
        needsSetup: '설정 필요',
        unavailable: '사용 불가',
        inUse: '사용 중',
        selectedTitle: '백엔드 선택됨',
        selectedMessage: backend => `터미널 명령어가 이제 ${backend}을(를) 통해 실행돼요. 새 세션에 적용돼요.`,
        failedSelect: backend => `${backend} 선택 실패`,
        needsSetupHint:
          '이 백엔드는 전체 설정 없이 현재 선택되어 있어요 — 설정이 완료될 때까지 명령이 실패해요.',
        needsSetupConfirmTitle: backend => `그래도 ${backend}을(를) 선택할까요?`,
        needsSetupConfirmDescription: detail =>
          `${detail} 이 변경 이후 시작되는 세션은 설정이 완료될 때까지 터미널이나 파일 도구를 사용할 수 없어요.`,
        needsSetupConfirmDescriptionGeneric:
          '이 백엔드는 아직 설정되지 않았어요. 이 변경 이후 시작되는 세션은 설정이 완료될 때까지 터미널이나 파일 도구를 사용할 수 없어요.',
        needsSetupConfirmAction: '그래도 선택',
        unavailableTitle: '터미널 명령을 사용할 수 없어요',
        unavailableMessage: backend =>
          `현재 Hermes에서 셸 명령을 실행할 수 없어요: ${backend}이(가) 준비되지 않았어요. Local로 전환하거나 ${backend} 설정을 완료한 후 다시 시도해 주세요.`,
        openBackendSettings: '터미널 설정 열기',
        useLocal: 'Local 사용',
        switchedToLocal: '터미널 명령이 이제 로컬에서 실행돼요. 새 세션에 적용돼요.'
      },
      browserRealProfile: {
        label: '실제 브라우저 프로필 사용',
        description:
          '기본 브라우저의 로그인 정보와 쿠키를 관리형 스냅샷으로 복사하여 에이전트가 탐색할 때 사용해요. 실제 프로필을 직접 열지 않아요. 새 세션에 적용돼요.',
        enabledTitle: '실제 프로필 탐색 켜짐',
        enabledMessage: '새 세션에서는 기본 브라우저 프로필의 스냅샷을 사용하여 탐색해요.',
        disabledTitle: '실제 프로필 탐색 꺼짐',
        disabledMessage: '프로필 스냅샷이 삭제되며, 새 세션에서는 깨끗한 브라우저를 사용해요.',
        failedSave: '실제 프로필 설정을 저장할 수 없어요',
        prompt: {
          title: '사이트 로그인 상태 유지',
          body: 'Hermes가 기본 브라우저 프로필의 스냅샷으로 탐색하도록 하여 사이트가 이미 로그인된 상태로 열리도록 해요.',
          bulletSnapshot: '쿠키와 로그인 정보가 관리형 스냅샷으로 복사돼요.',
          bulletLiveProfile: '실제 브라우저 프로필은 직접 열리지 않아요.',
          bulletLocal: '어떤 정보도 이 컴퓨터 외부로 전송되지 않아요.',
          dontShowAgain: '다시 보지 않기',
          notNow: '나중에',
          enable: '내 프로필 사용'
        }
      }
    },
  },
}
