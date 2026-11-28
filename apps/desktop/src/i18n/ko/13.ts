// ko/13.ts — Korean translation of the `localModels` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko13: TranslationOverrides = {
  settings: {
localModels: {
      connectionChanged: '로컬 모델 연결이 변경됐어요',
      title: '로컬 모델',
      runtimeTitle: '로컬 런타임',
      runtimeReady: backend => `준비됨 · ${backend}`,
      serverRunning: '실행 중',
      runtimeInstalled: 'llama.cpp 런타임 설치됨',
      runtimeInstalledDetail: (tag, backend) =>
        `빌드 ${tag}, ${backend} 백엔드. Hermes가 서버를 시작하고 관리해 드려요.`,
      installTitle: '로컬 런타임 설치',
      installDetail:
        'llama.cpp 추론 엔진을 다운로드해요(수백 MB). 다운로드한 모델은 전부 이 컴퓨터에서 실행돼요 — 계정도 필요 없고, 어디에도 전송되지 않아요.',
      installAction: '런타임 설치',
      installing: '런타임 설치 중…',
      installFailed: '런타임 설치에 실패했어요',
      hardwareTitle: '이 컴퓨터',
      hardwareLoading: '하드웨어를 확인하는 중…',
      vram: label => `${label} GPU 메모리`,
      ram: label => `${label} RAM`,
      unifiedMemory: '통합 메모리',
      modelsTitle: '모델',
      recommended: '추천',
      /* The Recommended badge's tooltip, keyed by the resolver branch that
         made the pick. Qualitative on purpose: predictions order candidates,
         they are not promises to print. */
      recommendedReason: {
        'best-quality-resident':
          'GPU에서 최대 속도로 완전히 실행되는 가장 품질 좋은 모델이에요. 품질과 이 하드웨어에서의 예상 속도를 함께 고려해 선택해요.',
        'speed-gated-quality':
          '더 품질 좋은 모델도 이 컴퓨터에 맞지만 메모리 대역폭 때문에 응답이 너무 느려요 — 이 모델이 속도를 유지하는 최선의 선택이에요.',
        'fastest-resident':
          '이 하드웨어에서 최대 속도에 도달하는 모델은 없어요. 이 모델은 GPU 메모리에 완전히 올라간 상태에서 가장 근접해요.'
      } as Record<string, string>,
      noRecommendationTitle: '이 컴퓨터에 대한 자동 추천이 없어요',
      noRecommendationDetail:
        '자동 설정에는 GPU 또는 통합 메모리에 완전히 들어가는 검증된 모델이 필요해요. 아래에서 모델을 직접 고르거나 더 많은 모델을 찾아볼 수 있어요.',
      noRecommendationAction: '모델 찾아보기',
      downloaded: '다운로드됨',
      downloadAction: size => `다운로드 · ${size}`,
      downloadProgress: (done, total) => `${total} 중 ${done}`,
      downloadStatusRunning: '다운로드 중',
      downloadSpeed: rate => `${rate}`,
      downloadEta: time => `~${time} 남음`,
      downloadEtaSeconds: count => `${count}초`,
      downloadEtaMinutes: count => `${count}분`,
      downloadEtaHours: (hours, minutes) => (minutes ? `${hours}시간 ${minutes}분` : `${hours}시간`),
      downloadPausedLabel: '일시 정지됨',
      downloadPauseAction: '일시 정지',
      downloadResumeAction: '계속',
      downloadDoneToast: model => `${model} 다운로드가 완료됐어요.`,
      installDoneToast: '로컬 런타임 설치가 완료됐어요.',
      quickstartTitle: '이 컴퓨터에서 모델 실행',
      quickstartDetail: (model, size) =>
        `클릭 한 번이면 로컬 엔진, ${model}(${size} 다운로드), 새 대화의 기본 모델까지 모두 설정돼요. 어디에도 전송되지 않아요.`,
      quickstartDetailReady: model =>
        `클릭 한 번이면 ${model}이 새 대화의 기본 모델이 돼요. 모든 것이 이 컴퓨터에서 실행돼요.`,
      quickstartAction: '자동으로 설정',
      quickstartConfigure: '직접 선택',
      quickstartDoneToast: model => `${model} 설정이 끝났어요 — 새 대화는 이 컴퓨터에서 실행돼요.`,
      quickstartFailed: '로컬 모델 설정에 실패했어요',
      quickstartStageEngine: '엔진',
      quickstartStageModel: '모델',
      quickstartStageFinish: '완료',
      useAction: '사용',
      activePill: '기본',
      updateTitle: '엔진 업데이트 있음',
      updateDetail: (next, current) =>
        `더 새로운 llama.cpp 빌드(${next})를 설치할 수 있어요 — 현재는 ${current}예요. 다운로드 중에도 모델은 계속 작동해요.`,
      updateAction: '엔진 업데이트',
      updating: '엔진 업데이트 중…',
      upToDateTitle: '엔진 최신 상태',
      upToDateDetail: (tag, backend) => `llama.cpp ${tag}(${backend}) 실행 중이에요.`,
      activeDetail: '새 대화에서 이 모델을 사용해요 — 첫 메시지를 보낼 때 로드돼요',
      activeNotLoaded: '첫 메시지에서 로드돼요',
      loadedPill: '메모리에 로드됨',
      placementResident: '전부 GPU',
      placementSpilled: '일부 RAM',
      placementResidentTip: '이 컨텍스트 창에서 GPU 메모리에 완전히 올라가 최대 속도로 실행돼요.',
      placementSpilledTip:
        '이 모델의 일부는 시스템 RAM에서 실행돼요 — 작동은 하지만 느려요. 더 작은 빌드나 더 짧은 컨텍스트를 쓰면 전부 올릴 수 있어요.',
      loadingPill: '로드 중…',
      ejectTip: 'GPU 메모리 확보(다음 메시지에서 다시 로드돼요)',
      ejected: '모델을 내렸어요 — GPU 메모리를 확보했어요.',
      ejectFailed: '모델을 내리지 못했어요',
      stopServer: '끄기',
      startServer: '켜기',
      runtimeRunningDetail:
        '로컬 서버가 실행 중이에요. 끄면 GPU 메모리를 모두 확보하고, 다시 켤 때까지 새 대화에서 로컬 모델을 사용할 수 없어요.',
      serverStopped: '로컬 서버를 중지했어요 — GPU 메모리를 확보했어요.',
      serverStarted: '로컬 서버 실행 중이에요.',
      serverStopFailed: '로컬 서버를 중지하지 못했어요',
      serverStartFailed: '로컬 서버를 시작하지 못했어요',
      activating: '시작 중…',
      activateFailed: model => `${model} 모델로 전환하지 못했어요`,
      activateDoneToast: model => `새 대화에서 ${model}을 사용해요.`,
      downloadFailed: model => `${model} 다운로드에 실패했어요`,
      downloadPauseFailed: model => `${model} 다운로드를 일시 정지하지 못했어요`,
      downloadResumeFailed: model => `${model} 다운로드를 계속하지 못했어요`,
      pillFitsGpu: 'GPU에 맞음',
      pillUsesRam: '시스템 RAM 사용',
      pillTooBig: '이 컴퓨터에 너무 큼',
      browseTitle: '더 많은 모델 찾기',
      browseHint:
        'Hugging Face 전체를 검색해요. 여기서 다운로드하는 모델은 컴퓨터 사양에 맞게 자동으로 조정되지만, 저희가 테스트한 것은 아니에요.',
      browsePlaceholder: '이름이나 작성자로 모델 검색…',
      browseSearching: 'Hugging Face 검색 중',
      browseListing: '모델 파일 읽는 중',
      browseShowFiles: '파일 보기',
      browseRefresh: '새로 고침',
      browseDownloads: '다운로드',
      browseLikes: '좋아요',
      browseGated: 'Hugging Face 로그인 필요',
      browseNoGguf: '호환되는 모델 파일을 찾지 못했어요.',
      browseFitUnknown: '호환 여부 알 수 없음',
      browseAlreadyDownloaded: '이미 다운로드됨.',
      addedByYou: '직접 추가함',
      browseDownloadStarted: '{name} 다운로드 중',
      browseDownloadAria: '{name} 다운로드',
      sideloadButton: '모델 파일 추가',
      sideloadTitle: 'GGUF 모델 파일 선택',
      sideloadDone: '{name}을 추가했어요.',
      sideloadAlreadyPresent: '이미 라이브러리에 있어요.',
      pillFullContext: max => `전체 ${max} 컨텍스트`,
      pillFullContextTip: '처음부터 모델의 전체 컨텍스트 창으로 실행돼요',
      pillUpTo: max => `최대 ${max} 컨텍스트`,
      pillGrowsTip: '대화에 더 많은 공간이 필요하면 자동으로 늘어나요',
      pillVision: '이미지 인식',
      deleteAction: '모델 삭제',
      deleteConfirm: model => `${model}을 디스크에서 삭제할까요?`,
      deleted: model => `${model}을 삭제했어요.`,
      deleteFailed: '삭제하지 못했어요'
    },
  },
}
