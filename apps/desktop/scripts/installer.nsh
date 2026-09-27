; Launch the installed binary, independent of shortcut migration and shell context.
!macro customInstall
  StrCpy $launchLink "$INSTDIR\${APP_EXECUTABLE_FILENAME}"
!macroend
