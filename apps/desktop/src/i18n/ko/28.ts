// ko/28.ts — Korean translation of the `thread` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko28: TranslationOverrides = {
  assistant: {
    thread: {
      loadingSession: '세션 불러오는 중',
      showEarlier: '이전 메시지 표시',
      loadingResponse: 'Hermes가 응답을 불러오는 중이에요',
      loadingLocalModel: model => `${model} 모델을 메모리로 불러오는 중`,
      processingPrompt: '프롬프트 처리 중',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? '백그라운드 작업이 완료되면 다시 시작해요'
          : `백그라운드 작업 ${count}개가 완료되면 다시 시작해요`,
      thinking: '생각하는 중',
      thought: '생각 완료',
      thoughtBriefly: '잠시 생각함',
      thoughtFor: duration => `${duration} 동안 생각함`,
      turnDuration: duration => `이 턴에 ${duration} 걸렸어요`,
      today: time => `오늘, ${time}`,
      yesterday: time => `어제, ${time}`,
      copy: '복사',
      refresh: '새로 고침',
      moreActions: '더 많은 작업',
      branchNewChat: '새 대화로 분기',
      react: '반응',
      dismissError: '오류 닫기',
      errorLayers: {
        auth: '로그인 문제',
        billing: '크레딧 부족',
        disk: '디스크 공간 부족',
        endpoint: '모델 서버에 연결할 수 없어요',
        gateway: 'Hermes에 문제가 발생했어요',
        generic: 'Hermes가 응답을 완료하지 못했어요',
        provider: 'AI 서비스에서 오류를 반환했어요',
        runtime: 'Hermes에 문제가 발생했어요',
        streaming: '응답이 중단되었어요'
      },
      errorLayerBodies: {
        auth: 'AI 서비스에서 로그인을 거부했어요. 해당 제공자의 인증 정보를 확인한 후 메시지를 다시 보내주세요.',
        billing: '해당 제공자에 사용할 수 있는 계정 크레딧이 없어요. 충전하거나 다른 제공자로 전환한 후 다시 보내주세요.',
        disk: '디스크가 가득 차서 Hermes가 이 대화를 저장할 수 없었어요. 디스크 공간을 확보한 후 다시 시도해 주세요.',
        endpoint:
          'Hermes가 커스텀 모델 서버에 연결할 수 없어요. 서버가 실행 중인지 확인한 후 메시지를 다시 보내주세요.',
        gateway:
          'Hermes가 응답을 시작하는 중에 내부 문제가 발생했어요. 메시지를 다시 보내주세요. 문제가 계속되면 진단 정보를 보내주세요.',
        generic: 'Hermes가 응답하는 중에 문제가 발생했어요. 다시 시도하거나, 문제가 계속되면 세부 정보를 복사해 주세요.',
        provider: 'AI 서비스에서 요청을 완료할 수 없었어요. 잠시 후 다시 시도하거나 다른 제공자로 전환해 주세요.',
        runtime:
          'Hermes가 응답을 시작하는 중에 내부 문제가 발생했어요. 메시지를 다시 보내주세요. 문제가 계속되면 진단 정보를 보내주세요.',
        streaming: '응답이 완료되기 전에 연결이 끊어졌어요. 다시 시도해 메시지를 보내주세요.'
      },
      errorCodes: {
        auth: {
          title: provider => `${provider}에서 로그인을 거부했어요`,
          body: provider =>
            `${provider}에 저장된 인증 정보가 승인되지 않았어요. 설정에서 수정하거나 다른 제공자로 전환한 후 메시지를 다시 보내주세요.`
        },
        auth_permanent: {
          title: provider => `${provider}에서 로그인을 거부했어요`,
          body: provider =>
            `${provider}에 저장된 인증 정보가 유효하지 않거나 취소되었어요. 정보를 업데이트하거나 다른 제공자로 전환한 후 메시지를 다시 보내주세요.`
        },
        billing: {
          title: '크레딧 부족',
          body: provider => `${provider} 계정의 남은 크레딧이 없어요. 충전하거나 다른 제공자로 전환한 후 다시 보내주세요.`
        },
        rate_limit: {
          title: 'AI 서비스가 혼잡해요',
          body: provider => `현재 ${provider}에서 요청을 제한하고 있어요. 1분 정도 기다린 후 다시 시도해 주세요.`
        },
        upstream_rate_limit: {
          title: 'AI 서비스가 혼잡해요',
          body: provider => `현재 ${provider}에서 요청을 제한하고 있어요. 1분 정도 기다린 후 다시 시도해 주세요.`
        },
        overloaded: {
          title: 'AI 서비스가 과부하 상태예요',
          body: provider => `현재 ${provider}에 문제가 발생했어요. 잠시 후 다시 시도하거나 다른 제공자로 전환해 주세요.`
        },
        server_error: {
          title: 'AI 서비스에 문제가 발생했어요',
          body: provider => `${provider}에서 서버 오류를 반환했어요. 잠시 후 다시 시도하거나 다른 제공자로 전환해 주세요.`
        },
        timeout: {
          title: '응답 시간이 초과되었어요',
          body: provider => `${provider}에서 제시간에 응답하지 않았어요. 다시 시도해 보내주세요.`
        },
        stream_drop: {
          title: '응답이 중단되었어요',
          body: '응답이 완료되기 전에 연결이 끊어졌어요. 다시 시도해 메시지를 보내주세요.'
        },
        upstream_blocked: {
          title: '방화벽에서 요청을 차단했어요',
          body: provider =>
            `${provider} 앞단의 방화벽이나 CDN이 모델에 도달하기 전에 요청을 차단했어요. 키 자체에는 문제가 없을 가능성이 높아요. 설정의 제공자 extra_headers에서 User-Agent 헤더를 설정하거나, 다른 제공자로 전환한 후 메시지를 다시 보내주세요.`
        },
        ssl_cert_verification: {
          title: '보안 연결에 실패했어요',
          body: provider =>
            `Hermes가 ${provider} 보안 연결을 확인할 수 없었어요. 네트워크나 프록시 설정을 확인하거나, 다른 제공자로 전환한 후 메시지를 다시 보내주세요.`
        },
        context_overflow: {
          title: '대화 내용이 너무 길어요',
          body: '대화가 모델의 허용 크기를 초과했어요. 대화를 압축하거나 새 대화를 시작한 후 다시 보내주세요.'
        },
        payload_too_large: {
          title: '메시지가 너무 커요',
          body: '요청 크기가 모델의 허용 크기보다 커요. 대화를 압축하거나 새 대화를 시작한 후 다시 보내주세요.'
        },
        model_not_found: {
          title: '이 모델을 사용할 수 없어요',
          body: provider =>
            `사용자 계정에서는 ${provider}의 이 모델을 지원하지 않아요. 다른 모델을 선택한 후 메시지를 다시 보내주세요.`
        },
        provider_policy_blocked: {
          title: '계정 설정에 의해 이 모델이 차단되었어요',
          body: provider =>
            `계정의 데이터 또는 개인정보 보호 설정으로 인해 ${provider}에서 이 요청을 라우팅하지 않았어요. 다른 모델을 선택하거나 다른 제공자로 전환해 주세요.`
        },
        content_policy_blocked: {
          title: 'AI 서비스에서 이 요청을 거부했어요',
          body: provider => `${provider}에서 이 메시지에 응답하지 않았어요. 내용을 수정한 후 다시 보내주세요.`
        },
        format_error: {
          title: 'AI 서비스에서 요청을 거부했어요',
          body: provider =>
            `${provider}에서 요청 형식을 승인하지 않았어요. 다른 제공자로 전환하거나 진단 정보를 보내주시면 원인을 확인해 볼게요.`
        },
        truncated: {
          title: '응답이 도중에 끊겼어요',
          body: '모델이 완료하기 전에 중단되었어요. 전체 응답을 받으려면 다시 시도해 주세요.'
        },
        invalid_response: {
          title: 'AI 서비스에서 읽을 수 없는 응답을 보냈어요',
          body: provider => `${provider}에서 Hermes가 읽을 수 없는 형식의 응답을 반환했어요. 잠시 후 다시 시도해 주세요.`
        },
        empty_response: {
          title: 'AI 서비스에서 빈 응답을 보냈어요',
          body: provider => `${provider}에서 이 메시지에 아무런 응답도 반환하지 않았어요. 잠시 후 다시 시도해 주세요.`
        },
        loop_error: {
          title: 'Hermes가 루프에 빠졌어요',
          body: '응답이 동일한 단계를 계속 반복하여 Hermes가 중단했어요. 다시 시도하거나, 계속 발생하면 새 대화를 시작해 주세요.'
        },
        SESSION_NOT_OWNED: {
          title: '이 대화가 다른 곳에서 열려 있어요',
          body: '이 대화가 현재 다른 Hermes 창이나 터미널에서 열려 있어요. 해당 창을 닫고 메시지를 다시 보내거나, 여기서 새 대화를 시작해 주세요.'
        },
        disk_full: {
          title: '디스크 공간 부족',
          body: '디스크가 가득 차서 Hermes가 이 대화를 저장할 수 없었어요. 디스크 공간을 확보한 후 다시 시도해 주세요.'
        },
        // Nous free tier. The body is normally the backend's own sentence (it names the wait
        // and the way forward); these bodies stand in for an older backend that sent none.
        free_tier_disabled: {
          title: '현재 로그인 없이 Hermes를 사용하는 기능이 꺼져 있어요',
          body: "계속 대화하려면 Nous 계정으로 로그인해 주세요. 무료입니다."
        },
        free_tier_rate_limited: {
          title: '비로그인 대화 한도를 모두 사용했어요',
          body: "잠시 후 한도가 초기화돼요. 더 많은 한도를 사용하려면 Nous 계정으로 로그인해 주세요. 무료입니다."
        },
        free_tier_at_capacity: {
          title: '현재 비로그인 대화 이용량이 많아 혼잡해요',
          body: "대기 없이 이용하려면 Nous 계정으로 로그인해 주세요. 무료이며, 잠시 후 다시 시도하셔도 됩니다."
        },
        free_tier_model_not_free: {
          title: '로그인하지 않고는 해당 모델을 사용할 수 없어요',
          body: "현재 Hermes는 무료 모델을 사용하고 있어요. 더 많은 모델을 사용하려면 Nous 계정으로 로그인해 주세요. 무료입니다."
        },
        free_tier_route: {
          title: 'Hermes가 이 경로의 무료 모델에 연결할 수 없었어요',
          body: '무료인 Nous 계정으로 로그인하거나, NOUS_INFERENCE_BASE_URL 설정을 확인해 주세요.'
        },
        free_tier_outage: {
          title: '현재 무료 모델 응답에 문제가 발생했어요',
          body: '1분 후에 메시지를 다시 보내보세요.'
        },
        free_tier_refused: {
          title: '로그인하지 않고는 전송할 수 없어요',
          body: 'Nous 계정 로그인은 무료입니다.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider}에서 API 키를 거부했어요`,
          body: provider => `${provider}에 저장된 키가 유효하지 않거나 취소되었어요. 키를 업데이트한 후 다시 시도해 주세요.`
        },
        oauth: {
          title: provider => `${provider} 로그인 세션이 만료되었어요`
        }
      },
      errorDetails: '세부 정보',
      errorGenericProvider: 'AI 서비스',
      errorToastTitle: 'Hermes가 응답을 완료하지 못했어요',
      errorRetry: '다시 시도',
      errorLimitResets: time => `제한 초기화 시간: ${time}`,
      errorRetryAtReset: time => `제한 초기화 시 다시 시도 (${time})`,
      errorRetryScheduled: (time, wait) => `${time}에 다시 시도 — ${wait} 후`,
      errorRetryScheduledCancel: '취소',
      errorStartNewSession: '새 세션 시작',
      errorSwitchProvider: '제공자 전환',
      errorChooseModel: '모델 선택',
      errorCompressConversation: '대화 압축',
      errorCompressFailed: '대화를 압축할 수 없어요',
      errorOpenHermesFolder: 'Hermes 폴더 열기',
      errorOpenHermesFolderFailed: 'Hermes 폴더를 열 수 없어요',
      errorUpdateApiKey: 'API 키 업데이트',
      errorSignInAgain: provider => `${provider}에 다시 로그인`,
      errorSignInFreeTier: 'Nous 계정으로 로그인',
      errorOauthExpired: provider =>
        `${provider} 로그인 세션이 만료되었거나 취소되었어요. 계속 대화하려면 다시 로그인해 주세요.`,
      errorOpenLogs: '로그 열기',
      errorOpenLogsFailed: '로그 폴더를 열 수 없어요',
      errorOpenDesktopLogs: '데스크톱 로그 열기',
      errorCopyDiagnostics: '오류 세부 정보 복사',
      errorSendDiagnostics: '진단 정보 보내기',
      filesChanged: count => (count === 1 ? '파일 1개 변경됨' : `파일 ${count}개 변경됨`),
      reviewChanges: '검토',
      readAloudFailed: '음성 읽기 실패',
      preparingAudio: '오디오 준비 중...',
      stopReading: '읽기 중지',
      readAloud: '음성으로 읽기',
      editMessage: '메시지 수정',
      expandMessage: '메시지 펼치기',
      scrollToBottom: '맨 아래로 스크롤',
      stop: '중지',
      restorePrevious: '이전 체크포인트로 복원',
      restoreCheckpoint: '체크포인트 복원',
      restoreFromHere: '체크포인트 복원 — 이 프롬프트부터 다시 실행',
      restoreTitle: '이 체크포인트로 복원할까요?',
      restoreBody:
        '이 프롬프트 이후의 모든 내용이 대화에서 삭제되며, 여기서부터 프롬프트가 다시 실행돼요.',
      restoreConfirm: '복원 및 다시 실행',
      restoreNext: '다음 체크포인트로 복원',
      goForward: '앞으로 이동',
      sendEdited: '수정한 메시지 전송',
      attachingFile: '첨부 중…'
    },
  },
}
