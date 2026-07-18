$kill = @(
  'chrome','claude','codex','ChatGPT','streamlit','WhatsApp.Root','ms-teams',
  'SpotifyWidgetProvider','HPCC.Bg.BackgroundApp','HpSystemManagement','SDXHelper',
  'OneDrive','OneDrive.Sync.Service','ExpressVPNNotificationService','PhoneExperienceHost',
  'M365Copilot','mscopilot_proxy','WidgetBoard','MicrosoftStartFeedProvider',
  'DesktopExtension','TouchpointAnalyticsClientService','ReconsentNotification',
  'SysInfoCap','AppVShNotify','BackgroundTaskHost','SearchProtocolHost','SearchIndexer',
  'CrossDeviceService','CrossDeviceResume','AppleMobileDeviceService','McUICnt',
  'hp-one-agent-service','HPCommRecovery','HP.HPX','HPCC.Bg.BackgroundSys',
  'ModuleCoreService','BridgeCommunication','AggregatorHost',
  'gamingservices','gamingservicesnet','GameInputSvc','GameInputRedistService',
  'TextInputHost','TabTip','StartMenuExperienceHost','ApplicationFrameHost',
  'SecurityHealthSystray','RuntimeBroker','conhost','cmd','bash','OpenConsole'
)
foreach ($p in $kill) {
  Get-Process -Name $p -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
