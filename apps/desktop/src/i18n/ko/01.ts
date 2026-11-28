// ko/01.ts — Korean translation of the `externalOpenFailed, connectors` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko01: TranslationOverrides = {
externalOpenFailed: {
    title: '링크를 열 수 없어요',
    message: '이 주소를 열 수 있는 브라우저가 없어요. 링크를 복사해 직접 열어 주세요.',
    copyUrl: '링크 복사',
    close: '닫기'
  },
connectors: {
  title: '앱 연결하기',
  connect: '연결',
  skip: '나중에',
  cancel: '대기 중지',
  retry: '다시 시도',
  grant: '다시 연결',
  connected: '연결됨',
  checking: '앱을 확인하는 중…',
  notConnected: '연결되지 않음',
  skipped: '건너뜀',
  disabled: '사용할 수 없음',
  failed: '연결하지 못했어요',
  needsAuth: '액세스 만료됨',
  opening: '로그인 여는 중…',
  waiting: '브라우저를 기다리는 중…',
  timeout: '아직 승인을 기다리고 있어요.',
  refresh: '상태 새로 고침',
  connectError: '승인을 시작하지 못했어요. 다시 시도해 주세요.',
  connectErrorFor: (app: string) => `${app} 승인을 시작하지 못했어요.`,
  unavailable: '이 세션에서는 커넥터를 사용할 수 없어요.',
  ownerMissing: '이 대화의 연결을 관리하려면 이 대화를 다시 열어 주세요.',
  search: '앱 찾기',
  empty: '일치하는 앱이 없어요',
  disclaimer: '연결은 선택 사항이에요. Hermes가 사용하기를 원하는 앱만 승인해 주세요.',
  execution: '커넥터 도구',
  setup: server => `${server} 설정`,
  openInBrowser: '브라우저에서 열기',
  setupCancel: '취소',
  authorizedToolsUnavailable: '승인됨. 도구를 사용할 수 없어요.',
  required: '필수'
},
}
