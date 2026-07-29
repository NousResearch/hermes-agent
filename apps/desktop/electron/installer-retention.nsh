Var retainedInstaller
Var retainedInstallerTemp
Var retainedInstallerBackup

!macro customInstall
  ; Retain this exact signed installer so a newly installed candidate can roll
  ; back to the last renderer-confirmed version. The app hashes the retained
  ; file before promotion and verifies that SHA-512 again before execution.
  CreateDirectory "$LOCALAPPDATA\${APP_PACKAGE_NAME}-rollback"
  CreateDirectory "$LOCALAPPDATA\${APP_PACKAGE_NAME}-rollback\installers"
  StrCpy $retainedInstaller "$LOCALAPPDATA\${APP_PACKAGE_NAME}-rollback\installers\${VERSION}.exe"
  StrCpy $retainedInstallerTemp "$LOCALAPPDATA\${APP_PACKAGE_NAME}-rollback\installers\${VERSION}.tmp"
  StrCpy $retainedInstallerBackup "$LOCALAPPDATA\${APP_PACKAGE_NAME}-rollback\installers\${VERSION}.bak"

  Delete "$retainedInstallerTemp"
  ClearErrors
  CopyFiles /SILENT "$EXEPATH" "$retainedInstallerTemp"
  IfErrors retain_installer_cleanup_temp

  IfFileExists "$retainedInstaller" 0 retain_installer_promote
  Delete "$retainedInstallerBackup"
  ClearErrors
  CopyFiles /SILENT "$retainedInstaller" "$retainedInstallerBackup"
  IfErrors retain_installer_cleanup_temp

  ClearErrors
  Delete "$retainedInstaller"
  IfErrors retain_installer_cleanup_temp

retain_installer_promote:
  ClearErrors
  Rename "$retainedInstallerTemp" "$retainedInstaller"
  IfErrors retain_installer_restore_backup
  Delete "$retainedInstallerBackup"
  Goto retain_installer_done

retain_installer_restore_backup:
  ; Promotion failed after the original moved aside. Keep the backup unless a
  ; byte-for-byte copy back to the canonical path succeeds.
  Delete "$retainedInstallerTemp"
  ClearErrors
  CopyFiles /SILENT "$retainedInstallerBackup" "$retainedInstaller"
  IfErrors retain_installer_done
  Delete "$retainedInstallerBackup"
  Goto retain_installer_done

retain_installer_cleanup_temp:
  Delete "$retainedInstallerTemp"

retain_installer_done:
!macroend
