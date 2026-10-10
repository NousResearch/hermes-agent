/** Copy composed into the regular settings catalog; locale selection stays in useI18n. */
export interface UninstallRiskCopy {
  options: Record<'gui' | 'lite' | 'full', { title: string; description: string; consequence: string }>
  checking: string
  confirmRemoval: (scope: string) => string
  appPath: (path: string) => string
  running: string
  confirmAction: string
  intro: string
  startFailed: string
}

export interface CustomEndpointsRiskCopy {
  loadFailed: string
  saved: string
  saveFailed: string
  reachable: string
  modelsFound: (count: number) => string
  validationFailed: string
  activationFailed: string
  deleteFailed: string
  deleteConfirm: (name: string) => string
  deleteDescription: string
  active: string
  use: string
  keySet: string
  edit: string
  add: string
  name: string
  providerId: string
  endpointUrl: string
  defaultModel: string
  contextLength: string
  apiKey: string
  keepCurrentKey: string
  optionalKey: string
  useForNewChats: string
  discoverModels: string
  test: string
  testHint: string
  newEndpoint: string
}

export interface SettingsRiskCopy {
  uninstallSection: UninstallRiskCopy
  customEndpoints: CustomEndpointsRiskCopy
}

export const settingsRiskCopyEn: SettingsRiskCopy = {
  uninstallSection: {
    options: {
      gui: {
        title: 'Uninstall the desktop app only',
        description:
          'Remove this app and its desktop settings. Keep the Hermes agent, agent configuration, chats, and secrets.',
        consequence:
          'the desktop app and its desktop settings; the Hermes agent, agent configuration, chats, and secrets are kept'
      },
      lite: {
        title: 'Uninstall the app and agent, keep my data',
        description:
          'Remove the app and Hermes agent. Keep agent configuration, chats, and secrets for a future reinstall.',
        consequence: 'the desktop app and Hermes agent; agent configuration, chats, and secrets are kept'
      },
      full: {
        title: 'Uninstall everything',
        description:
          'Remove the app, agent, and local Hermes data: configuration, chats, scheduled jobs, secrets, and logs.',
        consequence:
          'the desktop app, Hermes agent, and local Hermes data, including configuration, chats, scheduled jobs, secrets, and logs'
      }
    },
    checking: 'Checking what is installed…',
    confirmRemoval: scope => `This removes ${scope}. This cannot be undone.`,
    appPath: path => `App: ${path}`,
    running: 'Uninstalling…',
    confirmAction: 'Yes, uninstall',
    intro:
      'Choose what to remove. The app closes to finish uninstalling. You can reinstall it later, but deleted data is not restored.',
    startFailed: 'Uninstall could not start.'
  },
  customEndpoints: {
    loadFailed: 'Could not load custom endpoints',
    saved: 'Custom endpoint saved.',
    saveFailed: 'Could not save the endpoint',
    reachable: 'The endpoint is reachable.',
    modelsFound: count => `The endpoint is reachable. Found ${count} models.`,
    validationFailed: 'The endpoint check failed.',
    activationFailed: 'Could not activate the endpoint',
    deleteFailed: 'Could not delete the endpoint',
    deleteConfirm: name => `Delete endpoint “${name}”?`,
    deleteDescription:
      'Remove this endpoint’s configuration from Hermes. This does not delete the external service or revoke its API key.',
    active: 'Active',
    use: 'Use',
    keySet: 'API key set',
    edit: 'Edit endpoint',
    add: 'Add endpoint',
    name: 'Name',
    providerId: 'Provider ID',
    endpointUrl: 'Endpoint URL',
    defaultModel: 'Default model',
    contextLength: 'Context length (tokens)',
    apiKey: 'API key',
    keepCurrentKey: 'Leave blank to keep the current key when saving',
    optionalKey: 'Optional',
    useForNewChats: 'Use for new chats',
    discoverModels: 'Discover models',
    test: 'Check connection',
    testHint:
      'Checks the model list with the key entered here. A blank field does not use the saved key for this check. This does not send a chat request.',
    newEndpoint: 'New endpoint'
  }
}

export const settingsRiskCopyKo: SettingsRiskCopy = {
  uninstallSection: {
    options: {
      gui: {
        title: '데스크톱 앱만 제거',
        description: '이 앱과 앱 자체 설정을 제거합니다. Hermes 에이전트, 에이전트 설정, 대화, 인증 정보는 유지합니다.',
        consequence: '데스크톱 앱과 앱 자체 설정을 제거하고, Hermes 에이전트·에이전트 설정·대화·인증 정보는 유지합니다'
      },
      lite: {
        title: '앱과 에이전트 제거, 내 데이터 유지',
        description:
          '앱과 Hermes 에이전트를 제거합니다. 나중에 다시 설치할 수 있도록 에이전트 설정, 대화, 인증 정보는 유지합니다.',
        consequence: '데스크톱 앱과 Hermes 에이전트를 제거하고, 에이전트 설정·대화·인증 정보는 유지합니다'
      },
      full: {
        title: '모두 제거',
        description: '앱, 에이전트와 로컬 Hermes 데이터(설정, 대화, 예약 작업, 인증 정보, 로그)를 모두 제거합니다.',
        consequence:
          '데스크톱 앱, Hermes 에이전트와 로컬 Hermes 데이터(설정·대화·예약 작업·인증 정보·로그)를 모두 제거합니다'
      }
    },
    checking: '설치된 항목 확인 중…',
    confirmRemoval: scope => `${scope}. 이 작업은 되돌릴 수 없습니다.`,
    appPath: path => `앱: ${path}`,
    running: '제거 중…',
    confirmAction: '제거 실행',
    intro:
      '제거할 범위를 선택하세요. 제거를 마치기 위해 앱이 종료됩니다. 나중에 다시 설치할 수 있지만 삭제된 데이터는 복원되지 않습니다.',
    startFailed: '제거를 시작하지 못했습니다.'
  },
  customEndpoints: {
    loadFailed: '사용자 지정 엔드포인트를 불러오지 못했습니다',
    saved: '사용자 지정 엔드포인트를 저장했습니다.',
    saveFailed: '엔드포인트를 저장하지 못했습니다',
    reachable: '엔드포인트에 연결할 수 있습니다.',
    modelsFound: count => `엔드포인트에 연결할 수 있습니다. 모델 ${count}개를 찾았습니다.`,
    validationFailed: '엔드포인트 연결 검사에 실패했습니다.',
    activationFailed: '엔드포인트를 활성화하지 못했습니다',
    deleteFailed: '엔드포인트를 삭제하지 못했습니다',
    deleteConfirm: name => `엔드포인트 “${name}”을(를) 삭제할까요?`,
    deleteDescription:
      'Hermes에서 이 엔드포인트의 설정을 삭제합니다. 외부 서비스가 삭제되거나 해당 API 키가 폐기되는 것은 아닙니다.',
    active: '사용 중',
    use: '사용',
    keySet: 'API 키 설정됨',
    edit: '엔드포인트 편집',
    add: '엔드포인트 추가',
    name: '이름',
    providerId: '제공자 ID',
    endpointUrl: '엔드포인트 URL',
    defaultModel: '기본 모델',
    contextLength: '컨텍스트 길이(토큰)',
    apiKey: 'API 키',
    keepCurrentKey: '비워 두면 저장 시 기존 키 유지',
    optionalKey: '선택 사항',
    useForNewChats: '새 대화에 사용',
    discoverModels: '모델 검색',
    test: '연결 검사',
    testHint:
      '여기에 입력한 키로 모델 목록을 확인합니다. 입력란이 비어 있으면 이 검사에 저장된 키를 사용하지 않습니다. 대화 요청은 보내지 않습니다.',
    newEndpoint: '새 엔드포인트'
  }
}

export const settingsRiskCopyJa: SettingsRiskCopy = {
  uninstallSection: {
    options: {
      gui: {
        title: 'デスクトップアプリのみをアンインストール',
        description:
          'このアプリとアプリ自体の設定を削除します。Hermes エージェント、エージェント設定、チャット、認証情報は保持します。',
        consequence:
          'デスクトップアプリとアプリ自体の設定を削除します。Hermes エージェント、エージェント設定、チャット、認証情報は保持します'
      },
      lite: {
        title: 'アプリとエージェントを削除し、データは保持',
        description:
          'アプリと Hermes エージェントを削除します。再インストールに備えて、エージェント設定、チャット、認証情報は保持します。',
        consequence:
          'デスクトップアプリと Hermes エージェントを削除します。エージェント設定、チャット、認証情報は保持します'
      },
      full: {
        title: 'すべてアンインストール',
        description:
          'アプリ、エージェント、ローカルの Hermes データ（設定、チャット、スケジュール済みジョブ、認証情報、ログ）を削除します。',
        consequence:
          'デスクトップアプリ、Hermes エージェント、ローカルの Hermes データ（設定、チャット、スケジュール済みジョブ、認証情報、ログ）をすべて削除します'
      }
    },
    checking: 'インストール済みの項目を確認中…',
    confirmRemoval: scope => `${scope}。この操作は取り消せません。`,
    appPath: path => `アプリ: ${path}`,
    running: 'アンインストール中…',
    confirmAction: 'アンインストールを実行',
    intro:
      '削除する範囲を選択してください。処理を完了するためにアプリが終了します。再インストールはできますが、削除されたデータは復元されません。',
    startFailed: 'アンインストールを開始できませんでした。'
  },
  customEndpoints: {
    loadFailed: 'カスタムエンドポイントを読み込めませんでした',
    saved: 'カスタムエンドポイントを保存しました。',
    saveFailed: 'エンドポイントを保存できませんでした',
    reachable: 'エンドポイントに接続できます。',
    modelsFound: count => `エンドポイントに接続できます。${count} 件のモデルが見つかりました。`,
    validationFailed: 'エンドポイントの接続確認に失敗しました。',
    activationFailed: 'エンドポイントを有効にできませんでした',
    deleteFailed: 'エンドポイントを削除できませんでした',
    deleteConfirm: name => `エンドポイント「${name}」を削除しますか？`,
    deleteDescription:
      'Hermes からこのエンドポイントの設定を削除します。外部サービスの削除や API キーの失効は行いません。',
    active: '使用中',
    use: '使用',
    keySet: 'API キー設定済み',
    edit: 'エンドポイントを編集',
    add: 'エンドポイントを追加',
    name: '名前',
    providerId: 'プロバイダー ID',
    endpointUrl: 'エンドポイント URL',
    defaultModel: 'デフォルトモデル',
    contextLength: 'コンテキスト長（トークン）',
    apiKey: 'API キー',
    keepCurrentKey: '空欄の場合、保存時に現在のキーを保持',
    optionalKey: '任意',
    useForNewChats: '新しいチャットで使用',
    discoverModels: 'モデルを検出',
    test: '接続を確認',
    testHint:
      'ここに入力したキーでモデル一覧を確認します。空欄の場合、この確認には保存済みのキーを使用しません。チャットのリクエストは送信しません。',
    newEndpoint: '新しいエンドポイント'
  }
}

export const settingsRiskCopyZh: SettingsRiskCopy = {
  uninstallSection: {
    options: {
      gui: {
        title: '仅卸载桌面应用',
        description: '删除此应用及其桌面设置。保留 Hermes 智能体、智能体配置、聊天和凭据。',
        consequence: '桌面应用及其桌面设置。Hermes 智能体、智能体配置、聊天和凭据将保留'
      },
      lite: {
        title: '卸载应用和智能体，保留数据',
        description: '删除应用和 Hermes 智能体。保留智能体配置、聊天和凭据，供以后重新安装时使用。',
        consequence: '桌面应用和 Hermes 智能体。智能体配置、聊天和凭据将保留'
      },
      full: {
        title: '全部卸载',
        description: '删除应用、智能体及本地 Hermes 数据：配置、聊天、定时任务、凭据和日志。',
        consequence: '桌面应用、Hermes 智能体以及本地 Hermes 数据，包括配置、聊天、定时任务、凭据和日志'
      }
    },
    checking: '正在检查已安装的项目…',
    confirmRemoval: scope => `将删除${scope}。此操作无法撤销。`,
    appPath: path => `应用：${path}`,
    running: '正在卸载…',
    confirmAction: '确认卸载',
    intro: '选择要删除的范围。应用将关闭以完成卸载。可以稍后重新安装，但已删除的数据不会恢复。',
    startFailed: '无法开始卸载。'
  },
  customEndpoints: {
    loadFailed: '无法加载自定义端点',
    saved: '自定义端点已保存。',
    saveFailed: '无法保存端点',
    reachable: '可以连接到端点。',
    modelsFound: count => `可以连接到端点。已找到 ${count} 个模型。`,
    validationFailed: '端点连接检查失败。',
    activationFailed: '无法启用端点',
    deleteFailed: '无法删除端点',
    deleteConfirm: name => `删除端点“${name}”？`,
    deleteDescription: '从 Hermes 中删除此端点的配置。这不会删除外部服务或撤销其 API 密钥。',
    active: '使用中',
    use: '使用',
    keySet: '已设置 API 密钥',
    edit: '编辑端点',
    add: '添加端点',
    name: '名称',
    providerId: '提供方 ID',
    endpointUrl: '端点 URL',
    defaultModel: '默认模型',
    contextLength: '上下文长度（token）',
    apiKey: 'API 密钥',
    keepCurrentKey: '留空将在保存时保留当前密钥',
    optionalKey: '可选',
    useForNewChats: '用于新聊天',
    discoverModels: '发现模型',
    test: '检查连接',
    testHint: '使用此处输入的密钥检查模型列表。留空时，此检查不会使用已保存的密钥。不会发送聊天请求。',
    newEndpoint: '新端点'
  }
}

export const settingsRiskCopyZhHant: SettingsRiskCopy = {
  uninstallSection: {
    options: {
      gui: {
        title: '僅解除安裝桌面應用程式',
        description: '刪除此應用程式及其桌面設定。保留 Hermes 代理、代理設定、聊天和憑證。',
        consequence: '桌面應用程式及其桌面設定。Hermes 代理、代理設定、聊天和憑證將保留'
      },
      lite: {
        title: '解除安裝應用程式和代理，保留資料',
        description: '刪除應用程式和 Hermes 代理。保留代理設定、聊天和憑證，供日後重新安裝時使用。',
        consequence: '桌面應用程式和 Hermes 代理。代理設定、聊天和憑證將保留'
      },
      full: {
        title: '全部解除安裝',
        description: '刪除應用程式、代理及本機 Hermes 資料：設定、聊天、排程工作、憑證和日誌。',
        consequence: '桌面應用程式、Hermes 代理以及本機 Hermes 資料，包括設定、聊天、排程工作、憑證和日誌'
      }
    },
    checking: '正在檢查已安裝的項目…',
    confirmRemoval: scope => `將刪除${scope}。此操作無法復原。`,
    appPath: path => `應用程式：${path}`,
    running: '正在解除安裝…',
    confirmAction: '確認解除安裝',
    intro: '選擇要刪除的範圍。應用程式將關閉以完成解除安裝。可以稍後重新安裝，但已刪除的資料不會還原。',
    startFailed: '無法開始解除安裝。'
  },
  customEndpoints: {
    loadFailed: '無法載入自訂端點',
    saved: '自訂端點已儲存。',
    saveFailed: '無法儲存端點',
    reachable: '可以連線至端點。',
    modelsFound: count => `可以連線至端點。已找到 ${count} 個模型。`,
    validationFailed: '端點連線檢查失敗。',
    activationFailed: '無法啟用端點',
    deleteFailed: '無法刪除端點',
    deleteConfirm: name => `刪除端點「${name}」？`,
    deleteDescription: '從 Hermes 中刪除此端點的設定。這不會刪除外部服務或撤銷其 API 金鑰。',
    active: '使用中',
    use: '使用',
    keySet: '已設定 API 金鑰',
    edit: '編輯端點',
    add: '新增端點',
    name: '名稱',
    providerId: '提供方 ID',
    endpointUrl: '端點 URL',
    defaultModel: '預設模型',
    contextLength: '上下文長度（token）',
    apiKey: 'API 金鑰',
    keepCurrentKey: '留空會在儲存時保留目前的金鑰',
    optionalKey: '選填',
    useForNewChats: '用於新聊天',
    discoverModels: '探索模型',
    test: '檢查連線',
    testHint: '使用此處輸入的金鑰檢查模型清單。留空時，此檢查不會使用已儲存的金鑰。不會傳送聊天請求。',
    newEndpoint: '新端點'
  }
}
