import type { FirstRunCopy } from './types_first_run'

export const koFirstRun: FirstRunCopy = {
  beats: {
    name: '이름',
    theme: '강조 색상',
    layout: '화면 구성',
    connectors: '앱',
    plugins: '플러그인',
    tour: '둘러보기',
    build: '첫 작업'
  },
  namePlaceholder: '사용할 이름',
  useName: name => `“${name}” 사용`,
  install: '설치',
  installed: '설치됨',
  loading: '불러오는 중…',
  noPlugins: '현재 표시할 플러그인이 없습니다.',
  noneChosen: '선택 없음',
  noAnswer: '답변 없음',
  answerRefused: '답변이 수락되지 않았습니다',
  tours: {
    quick: '간단히 둘러보기',
    full: '모든 기능 둘러보기',
    skip: '건너뛰고 작업 시작'
  },
  skippedAll: '초기 설정을 건너뛰었습니다',
  skip: '초기 설정 건너뛰기',
  skipTip: '초기 설정 안내를 종료하고 새 채팅을 엽니다.',
  skipFailed: '초기 설정을 건너뛰지 못했습니다',
  localModel: {
    title: '이 컴퓨터에서 모델을 직접 실행할 수 있습니다',
    text: (model, size) =>
      `${model} (${size})을(를) 이 컴퓨터에서 실행할 수 있습니다. 무료로 실행되며 대화는 이 컴퓨터에 보관됩니다.`,
    setUp: '설정하기',
    notNow: '나중에'
  },
  intro: {
    greeting: '안녕하세요, Hermes입니다.',
    plan: '사용 환경을 먼저 설정하고, 함께 작업을 시작해 보세요.'
  },
  rerun: {
    title: '초기 설정 안내 다시 시작',
    description: '현재 모델로 초기 설정 단계를 안내하는 새 채팅을 엽니다.',
    action: '다시 시작',
    failed: '초기 설정 안내를 시작하지 못했습니다'
  }
}
