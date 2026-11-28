// ko/10.ts — Korean translation of the `connections, managedUpdates` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko10: TranslationOverrides = {
  settings: {
connections: {
      title: '등록된 게이트웨이',
      intro: '이 기기와, 원격 · SSH · Cloud 연결을 통해 접근할 수 있는 모든 Hermes 게이트웨이를 관리해요.',
      stagedNote:
        '세션에서 게이트웨이를 전환해요. 프로필, 채팅, 메시징, 크론 작업은 해당 게이트웨이에 그대로 남고, 다른 게이트웨이의 작업은 계속 실행돼요.',
      launchModeTitle: '시작할 때 마지막으로 사용한 게이트웨이의 세션으로 돌아가기',
      launchModeDesc: '꺼져 있으면 세션이 기본 게이트웨이에서 열려요.',
      searchPlaceholder: '게이트웨이 검색…',
      noSearchResults: '검색과 일치하는 게이트웨이가 없어요.',
      loadFailed: '연결을 불러올 수 없어요',
      currentPill: '현재',
      primaryPill: '기본',
      managedPill: '앱 관리',
      addConnection: '연결 추가',
      editConnection: '편집',
      removeConnection: '제거',
      removeConfirmTitle: '이 연결을 제거할까요?',
      removeConfirmDesc: (label: string) =>
        `“${label}” 연결이 이 앱에서 제거돼요. 인스턴스 자체는 그대로라 언제든 다시 추가할 수 있어요.`,
      makePrimary: '기본으로 설정',
      testConnection: '테스트',
      testOk: '연결 가능',
      testFailed: '연결 테스트 실패',
      saveFailed: '연결을 저장할 수 없어요',
      removeFailed: '연결을 제거할 수 없어요',
      updateAll: '모든 인스턴스 업데이트',
      updateAllRunning: '모든 인스턴스 업데이트 중…',
      updateAllDone: '업데이트 전송됨',
      updateAllFailed: '업데이트 배포 실패',
      updateSkippedCloud: 'Hermes Cloud에서 관리 중',
      kindLocal: '로컬',
      kindRemote: '원격 게이트웨이',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: '이 앱이 관리하는 Hermes 런타임이에요.',
      kindRemoteDesc: 'HTTP(S)로 접근할 수 있는 Hermes 게이트웨이예요 — LAN, Tailscale 또는 인터넷.',
      kindCloudDesc: 'Hermes Cloud 계정을 통해 검색된 호스팅 인스턴스예요.',
      kindSshDesc: 'SSH로 접근하는 Hermes 설치예요.',
      labelTitle: '이름',
      labelDesc: '필수예요. 이 인스턴스가 표시되는 모든 곳에 나타나며 고유해야 해요(예: “Homelab”, “회사 노트북”).',
      labelPlaceholder: 'Homelab',
      urlTitle: '게이트웨이 URL',
      sshHostTitle: 'SSH 호스트',
      headersTitle: '추가 게이트웨이 헤더',
      headersDesc:
        '이 게이트웨이로 보내는 모든 HTTP 및 WebSocket 요청에 함께 전송돼요 — Cloudflare Access(CF-Access-Client-Id / CF-Access-Client-Secret) 같은 접근 프록시용이에요. 값은 암호화되어 저장돼요. Hermes가 관리하는 헤더(Authorization, Cookie, Host…)는 무시돼요.',
      headerValuePlaceholder: '값',
      headerValueSaved: '저장됨 — 비워 두면 유지돼요',
      headerAdd: '헤더 추가',
      headerRemove: '제거',
      duplicateLocal: '이 앱이 이미 로컬 연결을 관리하고 있어요 — 하나만 있을 수 있어요.',
      duplicateUrl: (label: string) => `이 게이트웨이 URL에 대한 연결이 이미 있어요(“${label}”).`,
      duplicateSsh: (label: string) => `이 SSH 호스트에 대한 연결이 이미 있어요(“${label}”).`,
      sameBackendHint: (label: string) => `“${label}”과(와) 같은 백엔드`,
      localAddHint: '로컬을 사용할 수 없어요: 관리되는 로컬 연결이 이미 있어요(하나뿐이에요).',
      cloudAddHint:
        '팁: 위 Hermes Cloud에서 로그인하면 에이전트가 자동으로 검색돼요 — 이 양식은 알려진 인스턴스 URL을 직접 등록할 때만 사용해요.',
      save: '연결 저장',
      saving: '저장 중…',
      cancel: '취소',
      empty: '아직 등록된 연결이 없어요.'
    },
managedUpdates: {
      title: '관리형 업데이트',
      intro:
        'Desktop이 관리하는 SSH 설치를 트랜잭션 방식으로 업데이트해요: 세션이 정리되고, 원격 체크아웃이 업데이트되며, 모든 프로필이 대응하는 영수증과 함께 복원돼요.',
      sshConnection: 'Desktop이 관리하는 SSH 설치',
      update: '업데이트',
      updating: '업데이트 중…',
      progress: '세션을 정리하고 원격 설치를 업데이트한 뒤 프로필을 복원하는 중…',
      updated: '업데이트됨',
      partial: '업데이트됨 — 복원 실패',
      refused: '거부됨',
      failed: '업데이트 실패',
      alreadyRunning: '업데이트가 이미 진행 중이에요',
      receipt: (id: string, outcome: string) => `영수증 ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `복원된 프로필: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `프로필 “${profile}”을(를) 복원하지 못했어요: ${error}`
    },
  },
}
