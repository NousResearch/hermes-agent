// ko/11.ts — Korean translation of the `gateway, keys, search, profileScope` section(s) of en.ts.
// Translate ONLY the user-visible English strings into natural Korean (존댓말, concise UI tone).
// Keep every key, every function's parameters/arity, array lengths and placeholders exactly as-is.
// Do not add, remove or reorder keys, and keep trailing commas/quoting style intact.

import type { TranslationOverrides } from '../define-locale'

export const ko11: TranslationOverrides = {
  settings: {
    gateway: {
      loading: '게이트웨이 설정을 불러오는 중...',
      unavailableTitle: '게이트웨이 설정을 사용할 수 없어요',
      unavailableDesc:
        '연결 설정은 Hermes를 실행 중인 컴퓨터의 Hermes Desktop 앱에서만 변경할 수 있어요.',
      title: '게이트웨이 연결',
      envOverride: 'env 재정의',
      intro:
        '기본값은 로컬이에요. 이 앱이 다른 곳의 Hermes 백엔드를 제어해야 할 때 원격을 사용해요. 게이트웨이 연결은 컴퓨터 단위이고, 프로필은 연결한 게이트웨이에서 검색돼요.',
      envOverrideTitle: 'Hermes를 실행한 방식 때문에 이 연결이 고정됐어요.',
      envOverrideDesc:
        '앱 외부의 시작 설정이 이 연결을 선택해서 아래 옵션은 읽기 전용이에요. 그 설정 없이 Hermes를 다시 시작하거나, 설정한 사람에게 요청해서 여기서 변경해요.',
      modeTitle: '연결 모드',
      localTitle: '로컬 게이트웨이',
      localDesc: 'localhost에서 개인 Hermes 백엔드를 시작해요. 기본값이며 오프라인에서도 동작해요.',
      remoteTitle: '원격 게이트웨이',
      remoteDesc: '이 데스크톱 셸을 원격 Hermes 백엔드에 연결해요.',
      remoteAuthHint: '호스팅 게이트웨이는 OAuth나 사용자 이름과 비밀번호를 사용하고, 자체 호스팅 게이트웨이는 세션 토큰을 사용할 수 있어요.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc: 'Hermes Cloud에 한 번 로그인하고 계정의 에이전트 중에서 선택해요 — 붙여넣을 URL이 없어요.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Hermes Cloud에 로그인',
      cloudSignedIn: 'Hermes Cloud에 로그인됨',
      cloudNeedsSignIn: '계정의 에이전트를 확인하려면 Hermes Cloud에 로그인해 주세요.',
      cloudSignedInDesc: '로그인됐어요. 아래에서 에이전트를 선택해요. 세션은 자동으로 갱신돼요.',
      cloudAgentsTitle: '내 에이전트',
      cloudOrgPickerTitle: '조직 선택',
      cloudOrgSelect: '선택',
      cloudOrgChange: '조직 변경',
      cloudOrgRole: role => `역할: ${role}`,
      cloudLoadingAgents: '에이전트를 불러오는 중…',
      cloudNoAgents: {
        before: '이 계정에서 에이전트를 찾지 못했어요. ',
        linkText: 'Nous 포털',
        after: '에서 만들고 새로 고침해 주세요.'
      },
      cloudRefresh: '새로 고침',
      cloudConnect: '연결',
      cloudSavedTitle: '저장된 Cloud 게이트웨이',
      cloudSavedDesc:
        '기본값을 바꾸지 않고 저장된 게이트웨이를 사용해요. 아래에서 로그인해 인스턴스를 추가해요. 이름과 로그인은 저장된 연결 목록에서 관리해요.',
      cloudUseSaved: '게이트웨이 사용',
      cloudActive: '이 창에서 활성',
      cloudConnecting: '연결 중…',
      cloudDiscoverFailed: 'Hermes Cloud 에이전트를 불러올 수 없어요',
      cloudConnectFailed: '해당 에이전트에 연결할 수 없어요',
      cloudSignInFailed: 'Hermes Cloud 로그인에 실패했어요',
      cloudSignedOutTitle: 'Hermes Cloud에서 로그아웃했어요',
      cloudSignedOutMessage: 'Hermes Cloud 세션을 지웠어요.',
      cloudConnectedTitle: '연결됨',
      cloudConnectedPill: '연결됨',
      cloudConnectedTo: name => `${name}에 연결됐어요.`,
      cloudAgentProvisioning: '프로비저닝 중…',
      cloudStatusLabel: status => `상태: ${status}`,
      remoteUrlTitle: '원격 URL',
      remoteUrlDesc: '원격 대시보드 백엔드의 기본 URL이에요. 경로 접두사도 지원해요(예: /hermes).',
      probing: '이 게이트웨이의 인증 방식을 확인하는 중…',
      probeError:
        '해당 주소에 연결할 수 없어요. URL과 다른 컴퓨터에서 Hermes가 실행 중인지 확인해 주세요 — 응답이 오면 로그인 옵션이 나타나요.',
      signedIn: '로그인됨',
      signIn: '로그인',
      signOut: '로그아웃',
      signInWith: provider => `${provider}(으)로 로그인`,
      authTitle: '인증',
      authSignedInPassword:
        '이 게이트웨이는 사용자 이름과 비밀번호를 사용해요. 로그인된 상태이며 세션은 자동으로 갱신돼요.',
      authSignedInOauth: '이 게이트웨이는 OAuth를 사용해요. 로그인된 상태이며 세션은 자동으로 갱신돼요.',
      authNeedsPassword: '이 게이트웨이는 사용자 이름과 비밀번호를 사용해요. 이 데스크톱 앱을 승인하려면 로그인해 주세요.',
      authNeedsOauth: provider => `이 게이트웨이는 OAuth를 사용해요. 이 데스크톱 앱을 승인하려면 ${provider}(으)로 로그인해 주세요.`,
      tokenTitle: '세션 토큰',
      tokenDesc: 'REST 및 WebSocket 접근에 사용하는 대시보드 세션 토큰이에요. 비워 두면 저장된 토큰이 유지돼요.',
      existingToken: value => `기존 토큰 ${value}`,
      savedToken: '저장됨',
      pasteSessionToken: '세션 토큰 붙여넣기',
      plainTextConfirmTitle: '게이트웨이 토큰을 일반 텍스트로 저장할까요?',
      plainTextConfirmDesc:
        '이 컴퓨터에서 OS 키링 서비스를 찾지 못해서 토큰이 앱의 연결 설정 파일에 암호화되지 않은 채 저장되고, 이 사용자로 실행되는 모든 프로세스가 읽을 수 있어요. 암호화 저장을 사용하려면 시스템 키체인(Linux에서는 GNOME Keyring 또는 KWallet)을 설치하거나 활성화해 주세요.',
      plainTextConfirmAction: '일반 텍스트로 저장',
      plainTextStoredTitle: '토큰이 일반 텍스트로 저장됐어요',
      plainTextStoredDesc:
        '보안 저장소를 사용할 수 없어 저장된 토큰이 이 컴퓨터의 앱 연결 설정 파일에 암호화되지 않은 채 저장돼요. 암호화하려면 시스템 키체인(Linux에서는 GNOME Keyring 또는 KWallet)을 설치하거나 활성화해 주세요.',
      keychainEncryptionTitle: '저장된 비밀 정보를 OS 키체인으로 암호화',
      keychainEncryptionDesc:
        '기본값은 꺼짐이에요. 켜면 게이트웨이 토큰과 로그인 자격 증명이 시스템 키체인(Keychain Access, GNOME Keyring 또는 Windows DPAPI)으로 암호화돼요 — 시스템이 권한이나 비밀번호를 요청할 수 있어요. 끄면 이 사용자 계정만 읽을 수 있는 일반 파일로 저장돼요.',
      keychainEncryptionFailed: '비밀 정보 암호화를 변경할 수 없어요',
      testRemote: '원격 테스트',
      saveForRestart: '다음 재시작에 저장',
      saveAndReconnect: '저장 후 다시 연결',
      diagnostics: '진단',
      diagnosticsDesc: '파일 관리자에서 desktop.log를 열어요 — 게이트웨이가 시작되지 않을 때 유용해요.',
      openLogs: '로그 열기',
      incompleteTitle: '원격 게이트웨이 설정이 불완전해요',
      incompleteSignIn: '원격으로 전환하려면 원격 URL을 입력하고 로그인해 주세요.',
      incompleteToken: '원격으로 전환하려면 원격 URL과 세션 토큰을 입력해 주세요.',
      incompleteSignInTest: '테스트하려면 원격 URL을 입력하고 로그인해 주세요.',
      incompleteTokenTest: '테스트하려면 원격 URL과 세션 토큰을 입력해 주세요.',
      enterUrlFirst: '먼저 원격 URL을 입력해 주세요.',
      restartingTitle: '게이트웨이 연결 다시 시작 중',
      savedTitle: '게이트웨이 설정 저장됨',
      restartingMessage: 'Hermes Desktop이 저장된 설정으로 다시 연결해요 — 셸은 열린 상태로 유지돼요.',
      savedMessage: '다음 재시작에 적용되도록 저장했어요.',
      connectedTo: (baseUrl, version) => `${baseUrl}에 연결됐어요${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: '원격 게이트웨이에 연결할 수 있어요',
      signedOutTitle: '로그아웃됨',
      signedOutMessage: '원격 게이트웨이 세션을 지웠어요.',
      failedLoad: '게이트웨이 설정을 불러오지 못했어요',
      signInFailed: '로그인 실패',
      signOutFailed: '로그아웃 실패',
      testFailed: '원격 게이트웨이 테스트 실패',
      applyFailed: '게이트웨이 설정을 적용할 수 없어요',
      saveFailed: '게이트웨이 설정을 저장할 수 없어요',
      sshTitle: 'SSH로 연결',
      sshDesc:
        'SSH를 통해 원격에서 Hermes를 실행하고 이 앱으로 터널링해요 — 직접 시작하거나 노출할 것은 없어요. 호스트에 대한 키 기반 SSH 접근이 필요해요.',
      sshTrustHint: '처음 제시된 호스트 키를 신뢰하고 고정하며, 이후에 변경되면 연결이 거부돼요.',
      sshHostTitle: '호스트',
      sshHostDesc: 'user@host 또는 ~/.ssh/config의 Host 별칭이에요.',
      sshHostPick: '호스트 선택…',
      sshHostPickTitle: '호스트',
      sshHostPickDesc: '~/.ssh/config의 Host 별칭이에요. 직접 입력하려면 Custom을 선택해요.',
      sshHostCustom: 'Custom(직접 입력)…',
      sshUserTitle: '사용자',
      sshUserDesc: '비워 두면 ~/.ssh/config 또는 현재 사용자를 사용해요.',
      sshUserPlaceholder: '~/.ssh/config에서',
      sshPortTitle: '포트',
      sshPortDesc: '비워 두면 22 또는 ~/.ssh/config의 포트를 사용해요.',
      sshKeyTitle: 'ID 파일',
      sshKeyDesc: '개인 키 경로예요. 비워 두면 ssh-agent 또는 ~/.ssh/config를 사용해요.',
      sshHermesPathTitle: 'Hermes 경로(선택)',
      sshHermesPathDesc: '원격 hermes 바이너리의 전체 경로예요. 비워 두면 자동으로 감지해요.',
      sshHermesPathPlaceholder: '자동 감지',
      sshTestConnection: 'SSH 테스트',
      sshConnect: '연결',
      sshButtonsHint: '저장은 다음 실행에 적용되고, 연결은 지금 다시 연결해요.',
      sshReachable: (host, platform) => `연결 가능: ${host} (${platform}) — Hermes 발견`,
      sshIncompleteHost: '연결하려면 SSH 호스트를 입력해 주세요.',
      sshErrUnreachable: 'SSH로 해당 호스트에 연결할 수 없어요. 호스트, 포트, 네트워크를 확인해 주세요.',
      sshErrAuth:
        'SSH 인증에 실패했어요. 키를 ssh-agent에 등록하거나(ssh-add) ~/.ssh/config에 IdentityFile을 설정해 주세요 — Hermes는 ssh를 비대화식으로 실행해요.',
      sshErrHostKey:
        '마지막으로 연결한 이후 호스트 키가 변경됐어요. 예상된 변경인지 확인한 뒤 ssh-keygen -R <host>를 실행하고 다시 연결해 주세요.',
      sshErrNotInstalled:
        '원격 호스트에 Hermes가 설치되어 있지 않아요. 그곳에 설치하거나(curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) Hermes 경로를 설정해 주세요.',
      sshErrPlatform:
        '지원하지 않는 원격 플랫폼이에요. Hermes Desktop SSH 모드는 Linux, macOS, Windows 원격 호스트를 지원해요.',
      sshErrTimeout: 'SSH 연결 시간이 초과됐어요. 호스트가 응답하지 않거나 절전 상태일 수 있어요.',
      sshErrUpdateRequired: 'Desktop SSH로 연결하려면 원격 호스트의 Hermes를 업데이트해 주세요.',
      sshErrUnknown: 'SSH 연결에 실패했어요.'
    },
    keys: {
      loading: 'API 키와 자격 증명을 불러오는 중...',
      failedLoad: 'API 키를 불러오지 못했어요',
      empty: '이 범주에는 아직 설정된 항목이 없어요.'
    },
    search: {
      placeholder: '모든 설정 검색…',
      pill: '검색'
    },
    profileScope: {
      appliesTo: '적용 대상',
      editsProfile: profile => `이 페이지의 변경 사항은 “${profile}” 프로필에 적용돼요.`
    },
  },
}
