import { defineLocale } from './define-locale'

/** Korean Desktop copy. Unlisted future keys intentionally fall back to English. */
export const ko = defineLocale({
  common: {
    apply: '적용', back: '뒤로', save: '저장', saving: '저장 중…', cancel: '취소', change: '변경',
    choose: '선택', clear: '지우기', close: '닫기', confirm: '확인', connect: '연결', connecting: '연결 중',
    continue: '계속', copied: '복사됨', copy: '복사', copyFailed: '복사하지 못했습니다', delete: '삭제',
    docs: '문서', done: '완료', error: '오류', expand: '펼치기', collapse: '접기', failed: '실패',
    free: '무료', loading: '로드 중…', notSet: '설정되지 않음', refresh: '새로 고침', remove: '제거',
    replace: '바꾸기', retry: '다시 시도', run: '실행', send: '전송', set: '설정', skip: '건너뛰기',
    update: '업데이트', on: '켜짐', off: '꺼짐', tryHint: term => `${term} 시도`
  },
  language: { switchTo: '언어 변경' },
  boot: {
    ready: 'Hermes Desktop을 사용할 준비가 되었습니다',
    steps: {
      connectingGateway: 'Desktop 게이트웨이에 연결 중', loadingSettings: 'Hermes 설정을 불러오는 중',
      loadingSessions: '최근 세션을 불러오는 중', retryingRemoteBackend: '원격 Hermes 백엔드에 다시 연결하는 중…',
      startingDesktopConnection: 'Desktop 연결을 시작하는 중', startingHermesDesktop: 'Hermes Desktop을 시작하는 중…'
    }
  },
  composer: { message: '메시지', stop: '중지', send: '전송', thinking: '생각 중…', listening: '듣는 중' }
})
