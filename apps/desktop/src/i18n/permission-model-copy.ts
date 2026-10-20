/** Renderer-owned permission guidance and guarded model-switch summaries.
 * Driver diagnostics and gateway cost/data-policy warnings remain verbatim. */
export interface PermissionModelCopy {
  computerUse: {
    granted: string
    notGranted: string
    unknown: string
    readFailed: string
    requestFailed: string
    approveTitle: string
    approveMessage: string
    checking: string
    unsupported: (platform: string) => string
    install: string
    installPermissions: string
    permissionIdentity: string
    recheck: string
    accessibilityHint: string
    screenRecordingHint: string
    ready: string
    notReady: string
    readyHint: string
    waitingApproval: string
    grantPermissions: string
    platformNote: { linux: string; win32: string }
  }
  modelSwitch: {
    title: string
    summary: string
  }
}

export const en: PermissionModelCopy = {
  computerUse: {
    granted: 'Granted',
    notGranted: 'Not granted',
    unknown: 'Unknown',
    readFailed: 'Could not read Computer Use status',
    requestFailed: 'Could not request permissions',
    approveTitle: 'Approve in System Settings',
    approveMessage: 'macOS will show a permission dialog attributed to CuaDriver. Approve it, then return here.',
    checking: 'Checking Computer Use status…',
    unsupported: platform => `Computer Use isn’t supported on this platform (${platform}).`,
    install: 'Install the cua-driver backend below to drive this machine.',
    installPermissions: ' Then grant Accessibility and Screen Recording here.',
    permissionIdentity:
      'Grants attach to CuaDriver’s own identity (com.trycua.driver), not Hermes — so the dialog is attributed to the process that drives your Mac.',
    recheck: 'Recheck',
    accessibilityHint: 'Lets cua-driver post clicks, keystrokes, and read the accessibility tree.',
    screenRecordingHint: 'Lets cua-driver capture screenshots of app windows.',
    ready: 'Ready',
    notReady: 'Not ready',
    readyHint: 'Computer Use is ready. Ask the agent to capture an app and click around.',
    waitingApproval: 'Waiting for approval…',
    grantPermissions: 'Grant permissions',
    platformNote: {
      linux: 'Drives your desktop via the X11/XWayland accessibility stack — no permission prompt.',
      win32: 'First run may trigger a Windows SmartScreen prompt for the cua-driver UIAccess worker — allow it.'
    }
  },
  modelSwitch: {
    title: 'Confirm model switch',
    summary: 'Review this model’s cost and data-use conditions before approving the switch.'
  }
}

export const ko: PermissionModelCopy = {
  computerUse: {
    granted: '허용됨',
    notGranted: '허용되지 않음',
    unknown: '알 수 없음',
    readFailed: '컴퓨터 사용 상태를 확인하지 못했습니다',
    requestFailed: '권한을 요청하지 못했습니다',
    approveTitle: '시스템 설정에서 승인하세요',
    approveMessage: 'macOS에 CuaDriver의 권한 요청 창이 표시됩니다. 승인한 뒤 이 화면으로 돌아오세요.',
    checking: '컴퓨터 사용 상태 확인 중…',
    unsupported: platform => `이 플랫폼(${platform})에서는 컴퓨터 사용 기능을 지원하지 않습니다.`,
    install: '이 컴퓨터를 조작하려면 아래에서 cua-driver 백엔드를 설치하세요.',
    installPermissions: ' 그런 다음 여기에서 손쉬운 사용 및 화면 기록 권한을 허용하세요.',
    permissionIdentity:
      '권한은 CuaDriver 자체 식별자(com.trycua.driver)에 부여됩니다. 따라서 Mac을 조작하는 프로세스인 CuaDriver의 이름으로 권한 요청이 표시됩니다.',
    recheck: '다시 확인',
    accessibilityHint: 'cua-driver가 클릭과 키 입력을 수행하고 접근성 트리를 읽도록 허용합니다.',
    screenRecordingHint: 'cua-driver가 앱 창의 스크린샷을 캡처하도록 허용합니다.',
    ready: '준비 완료',
    notReady: '준비되지 않음',
    readyHint: '컴퓨터를 사용할 준비가 되었습니다. 에이전트에게 앱 화면 캡처나 클릭을 요청해 보세요.',
    waitingApproval: '승인 대기 중…',
    grantPermissions: '권한 요청',
    platformNote: {
      linux: 'X11/XWayland 접근성 기능으로 데스크톱을 조작합니다. 별도의 권한 요청 창은 표시되지 않습니다.',
      win32:
        '처음 실행할 때 cua-driver UIAccess 작업 프로세스에 대한 Windows SmartScreen 알림이 나타날 수 있습니다. 허용해 주세요.'
    }
  },
  modelSwitch: {
    title: '모델 변경 확인',
    summary: '이 모델의 비용 및 데이터 사용 조건을 확인한 뒤 변경을 승인해 주세요.'
  }
}

export const ja: PermissionModelCopy = {
  computerUse: {
    granted: '許可済み',
    notGranted: '未許可',
    unknown: '不明',
    readFailed: 'コンピューター操作の状態を取得できませんでした',
    requestFailed: '権限を要求できませんでした',
    approveTitle: 'システム設定で許可してください',
    approveMessage: 'macOS に CuaDriver の権限ダイアログが表示されます。許可してからここに戻ってください。',
    checking: 'コンピューター操作の状態を確認中…',
    unsupported: platform => `このプラットフォーム（${platform}）ではコンピューター操作を利用できません。`,
    install: 'このマシンを操作するには、下で cua-driver バックエンドをインストールしてください。',
    installPermissions: ' その後、ここでアクセシビリティと画面収録の権限を許可してください。',
    permissionIdentity:
      '権限は Hermes ではなく CuaDriver 自身の識別子（com.trycua.driver）に付与されます。ダイアログには Mac を操作するプロセスの名前が表示されます。',
    recheck: '再確認',
    accessibilityHint: 'cua-driver にクリック、キー入力、アクセシビリティツリーの読み取りを許可します。',
    screenRecordingHint: 'cua-driver にアプリウィンドウのスクリーンショット撮影を許可します。',
    ready: '準備完了',
    notReady: '未準備',
    readyHint: 'コンピューター操作の準備ができました。エージェントにアプリの撮影やクリックを依頼できます。',
    waitingApproval: '許可を待っています…',
    grantPermissions: '権限を要求',
    platformNote: {
      linux: 'X11/XWayland のアクセシビリティ機能でデスクトップを操作します。権限ダイアログは表示されません。',
      win32:
        '初回実行時に cua-driver UIAccess ワーカーの Windows SmartScreen 通知が表示される場合があります。許可してください。'
    }
  },
  modelSwitch: {
    title: 'モデル変更の確認',
    summary: 'このモデルの料金とデータ利用条件を確認してから、変更を承認してください。'
  }
}

export const zh: PermissionModelCopy = {
  computerUse: {
    granted: '已授权',
    notGranted: '未授权',
    unknown: '未知',
    readFailed: '无法读取电脑操作状态',
    requestFailed: '无法请求权限',
    approveTitle: '请在系统设置中授权',
    approveMessage: 'macOS 将显示 CuaDriver 的权限请求。授权后请返回此处。',
    checking: '正在检查电脑操作状态…',
    unsupported: platform => `此平台（${platform}）不支持电脑操作功能。`,
    install: '请在下方安装 cua-driver 后端，以操作此电脑。',
    installPermissions: ' 然后在此处授予辅助功能和屏幕录制权限。',
    permissionIdentity:
      '权限授予 CuaDriver 自身的标识（com.trycua.driver），而非 Hermes，因此对话框会显示实际操作 Mac 的进程名称。',
    recheck: '重新检查',
    accessibilityHint: '允许 cua-driver 执行点击、按键并读取辅助功能树。',
    screenRecordingHint: '允许 cua-driver 截取应用窗口画面。',
    ready: '已就绪',
    notReady: '未就绪',
    readyHint: '电脑操作已就绪。可以让代理截取应用画面并进行点击。',
    waitingApproval: '等待授权…',
    grantPermissions: '请求权限',
    platformNote: {
      linux: '通过 X11/XWayland 辅助功能操作桌面，不会弹出权限请求。',
      win32: '首次运行时，Windows SmartScreen 可能提示授权 cua-driver UIAccess 工作进程，请允许。'
    }
  },
  modelSwitch: {
    title: '确认切换模型',
    summary: '请先查看此模型的费用和数据使用条件，再批准切换。'
  }
}

export const zhHant: PermissionModelCopy = {
  computerUse: {
    granted: '已授權',
    notGranted: '未授權',
    unknown: '未知',
    readFailed: '無法讀取電腦操作狀態',
    requestFailed: '無法請求權限',
    approveTitle: '請在系統設定中授權',
    approveMessage: 'macOS 將顯示 CuaDriver 的權限請求。授權後請返回此處。',
    checking: '正在檢查電腦操作狀態…',
    unsupported: platform => `此平台（${platform}）不支援電腦操作功能。`,
    install: '請在下方安裝 cua-driver 後端，以操作此電腦。',
    installPermissions: ' 然後在此處授予輔助使用和螢幕錄製權限。',
    permissionIdentity:
      '權限授予 CuaDriver 自身的識別碼（com.trycua.driver），而非 Hermes，因此對話框會顯示實際操作 Mac 的程序名稱。',
    recheck: '重新檢查',
    accessibilityHint: '允許 cua-driver 執行點擊、按鍵並讀取輔助使用樹。',
    screenRecordingHint: '允許 cua-driver 擷取應用程式視窗畫面。',
    ready: '已就緒',
    notReady: '未就緒',
    readyHint: '電腦操作已就緒。可以讓代理擷取應用程式畫面並進行點擊。',
    waitingApproval: '等待授權…',
    grantPermissions: '請求權限',
    platformNote: {
      linux: '透過 X11/XWayland 輔助使用功能操作桌面，不會顯示權限請求。',
      win32: '首次執行時，Windows SmartScreen 可能提示授權 cua-driver UIAccess 工作程序，請允許。'
    }
  },
  modelSwitch: {
    title: '確認切換模型',
    summary: '請先查看此模型的費用和資料使用條件，再核准切換。'
  }
}
