@rem Set local directory if not set
@setlocal

@rem Get the real directory the script is in
@set BIN_DIR=%~dp0
@set PROJECT_DIR=%BIN_DIR%..

@rem Set APP_HOME
@set DEFAULT_APP_HOME=%PROJECT_DIR%
@set APP_HOME=%DEFAULT_APP_HOME%

@rem Resolve any "." or ".." in APP_HOME to make it shorter
@for %%i in ("%APP_HOME%") do @set APP_HOME=%%~fi

@rem Check if we have JAVA_HOME set
@if "%JAVA_HOME%"=="" (
    @rem Try to find Java in common locations
    @if exist "%PROGRAMFILES%\Java\jdk-17\bin\java.exe" set JAVA_HOME=%PROGRAMFILES%\Java\jdk-17
    @if exist "%PROGRAMFILES(X86)%\Java\jdk-17\bin\java.exe" set JAVA_HOME=%PROGRAMFILES(X86)%\Java\jdk-17
)

@rem Set JAVA_EXE
@if "%JAVA_HOME%"=="" (
    @set JAVA_EXE=java
) else (
    @set JAVA_EXE=%JAVA_HOME%\bin\java
)

@rem Check if java exists
@where %JAVA_EXE% >nul 2>&1
@if %ERRORLEVEL% neq 0 (
    @echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
    @echo Please set the JAVA_HOME variable in your environment to match the
    @echo location of your Java installation.
    @goto :EOF
)

@rem Setup the command line
@set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar

:run
@rem Execute Gradle
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %GRADLE_OPTS% -Dorg.gradle.appname="%APP_BASE_NAME%" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*

@endlocal
