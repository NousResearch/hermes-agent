import type { ScreenshotTranslations } from './types_screenshot'

export const zh_hantScreenshot: ScreenshotTranslations = {
  leftCommand: '左 ⌘',
  rightCommand: '右 ⌘',
  enabledTitle: '螢幕截圖快速鍵',
  enabledDesc:
    '在任何應用程式中同時按下左右兩個 Command 鍵，即可擷取最前方的視窗並附加到目前的 Hermes 草稿。絕不會自動傳送。預設關閉，僅適用於這台 Mac。視窗可能包含敏感內容，請在傳送前檢查附件。',
  statusTitle: '螢幕截圖快速鍵狀態',
  checking: '正在檢查螢幕截圖快速鍵…',
  disabled: '螢幕截圖快速鍵已關閉。',
  starting: '正在啟動快速鍵偵測，尚未就緒。',
  ready: '快速鍵已就緒。截圖會附加到目前的草稿，不會傳送。',
  inputPermission:
    '輸入監控權限可讓 Hermes 在其他應用程式使用中時偵測左右兩個 Command 鍵。請在系統設定 → 隱私權與安全性 → 輸入監控中允許 Hermes，然後返回此處重試。',
  screenPermission:
    '螢幕錄製權限可讓 Hermes 在你使用此快速鍵時擷取最前方的應用程式視窗。請在系統設定 → 隱私權與安全性 → 螢幕錄製中允許 Hermes，然後返回此處重試。如果 macOS 提示，請重新啟動 Hermes。',
  openSettings: '開啟系統設定',
  retry: '重試',
  unavailable: '螢幕截圖快速鍵無法使用。請重試或將其關閉。',
  errorTitle: '螢幕截圖快速鍵錯誤',
  loadFailed: '無法讀取快速鍵狀態。請重試以檢查目前的設定。',
  saveFailed: '無法確認快速鍵變更。請重試以檢查目前的設定。',
  permissionFailed: '無法開啟系統設定。請手動開啟「隱私權與安全性」，然後重試。',
  captureFailed: '無法擷取最前方的視窗。未附加或傳送任何內容。',
  contextChanged: '截圖期間目前的草稿已變更。截圖未附加或傳送。'
}
