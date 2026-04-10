# Hermes Android APK + CI Release Implementation Plan
> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

## Goal
Port Hermes from the current Termux-oriented branch to a releasable Android APK without rewriting the shared agent core.

The first public milestone is a chat-first Android app which:
- installs and runs without Termux
- embeds a Hermes Python runtime inside the APK
- uses `gateway/platforms/api_server.py` as the app/backend seam
- keeps Hermes state in app-private storage
- ships a reduced mobile-safe tool profile by default
- builds Android artifacts in CI
- signs and uploads release artifacts from CI on GitHub releases

## Architecture
- App shell: native Android app in `android/` using Kotlin + Jetpack Compose.
- Embedded runtime: Python runs inside the app process.
- Backend seam: the native app talks to a local loopback HTTP server powered by `gateway/platforms/api_server.py`.
- Shared Hermes core to preserve and reuse:
  - `run_agent.py`
  - `agent/*`
  - `hermes_constants.py`
  - `hermes_state.py`
  - `hermes_cli/auth.py`
  - `hermes_cli/config.py`
- App-private Hermes home:
  - set `HERMES_HOME` to something like `<filesDir>/hermes-home`
  - reuse current config/auth/state stores through that path
- Conversation contract:
  - MVP chat transport uses `POST /v1/chat/completions` with `stream=true`
  - the app provides a stable `X-Hermes-Session-Id` so the backend owns multi-turn state server-side
  - keep `/v1/responses` as a later enhancement if the app needs response retrieval or `previous_response_id` chaining
  - keep the Android UI as a thin client over the local API
- Tooling contract:
  - add a new default mobile toolset for the app
  - do not make shell-heavy tools default in the APK MVP
- Release contract:
  - debug APK on PRs
  - signed release APK + AAB on GitHub releases

## Tech Stack
- Android Gradle project in `android/`
- Kotlin + Jetpack Compose + coroutines + OkHttp
- Embedded Python via Chaquopy
- Local HTTP/SSE streaming to the embedded API server
- GitHub Actions for Android build, test, signing, checksum, and release upload

---

## Repo-Grounded Starting Point
- Reusable shared core already exists in:
  - `run_agent.py`
  - `agent/*`
  - `hermes_constants.py`
  - `hermes_state.py`
  - `hermes_cli/auth.py`
  - `hermes_cli/config.py`
- Best backend seam already exists in:
  - `gateway/platforms/api_server.py`
- API server docs already exist in:
  - `website/docs/user-guide/features/api-server.md`
- Current Android support is Termux-only in:
  - `scripts/install.sh`
  - `setup-hermes.sh`
  - `constraints-termux.txt`
  - `website/docs/getting-started/termux.md`
- Shell-heavy or non-mobile-first pieces that should not be MVP defaults:
  - `cli.py`
  - `hermes_cli/setup.py`
  - `tools/terminal_tool.py`
  - `tools/file_operations.py`
  - `tools/browser_tool.py`
  - `tools/voice_mode.py`
  - `tools/transcription_tools.py`
- CI today has no Android lane. Existing workflows are:
  - `.github/workflows/tests.yml`
  - `.github/workflows/docker-publish.yml`
  - `.github/workflows/deploy-site.yml`
  - `.github/workflows/nix.yml`
  - `.github/workflows/supply-chain-audit.yml`
  - `.github/workflows/docs-site-checks.yml`
- Release today is still local/manual via:
  - `scripts/release.py`

## Explicit MVP Non-Goals
- Do not port `cli.py` or prompt_toolkit to Android.
- Do not ship a terminal emulator inside the app.
- Do not make `terminal`, `process`, browser automation, voice, transcription, or direct POSIX file tools default in the APK MVP.
- Do not tie the app to Termux.
- Do not block the first APK release on full local coding/workspace support.
- Do not add Play Console upload before GitHub release artifacts work end-to-end.

## MVP Definition
The first releasable APK is done when all of the following are true:
- `android/` exists and builds on CI.
- The app boots an embedded Hermes runtime locally.
- The runtime exposes a healthy local API server using `gateway/platforms/api_server.py`.
- The app has a native onboarding/settings flow for provider/model/base URL/API key.
- The app streams chat responses from the local API server.
- Hermes state persists in app-private storage across restarts.
- The default tool profile is mobile-safe.
- CI produces a debug APK for PRs.
- CI produces signed release artifacts on GitHub releases.
- Release artifacts are attached to the GitHub release with checksums.

## Default Mobile-Safe Tool Profile
Create a new toolset named `hermes-android-app` in `toolsets.py`.

MVP default allowlist:
- `web_search`
- `web_extract`
- `vision_analyze`
- `image_generate`
- `skills_list`
- `skill_view`
- `skill_manage`
- `todo`
- `memory`
- `session_search`

MVP default denylist:
- `terminal`
- `process`
- `read_file`
- `write_file`
- `patch`
- `search_files`
- all `browser_*` tools
- `execute_code`
- `delegate_task`
- `cronjob`
- `send_message`
- `text_to_speech`
- voice/transcription tools

Notes:
- This is intentionally smaller than `hermes-api-server`.
- Local workspace editing returns later as an explicit post-MVP phase.
- The app should seed `platform_toolsets.api_server` to `['hermes-android-app']` on first run.

## Stage 0 — Create the Android lane and prove packaging viability
Do not proceed past Stage 0 until the app can boot Python code from a Gradle-built debug APK.

### [ ] 0.0 Lock the Android support matrix and artifact policy
Files:
- `android/README.md`
- `docs/plans/2026-04-10-android-apk-ci-port-plan.md`
Do:
- decide and record the exact MVP support matrix before deeper implementation:
  - min SDK
  - target SDK
  - CI emulator API level
  - Chaquopy + Python version pairing
  - artifact policy: universal debug APK on PRs; universal release APK + release AAB on GitHub releases; no ABI splits in MVP
- keep the first public release narrow and explicit
Verify:
- `read_file android/README.md`
- the recorded support matrix matches the Android build config once `android/app/build.gradle.kts` exists

### [ ] 0.1 Create the Android project scaffold
Files:
- `android/settings.gradle.kts`
- `android/build.gradle.kts`
- `android/gradle.properties`
- `android/app/build.gradle.kts`
- `android/app/proguard-rules.pro`
- `android/gradlew`
- `android/gradlew.bat`
- `android/gradle/wrapper/gradle-wrapper.properties`
- `android/gradle/wrapper/gradle-wrapper.jar`
Do:
- create a standard single-app Android project rooted at `android/`
- set the application id to `com.nousresearch.hermesagent`
- set min/target SDKs explicitly
Verify:
- `cd android && ./gradlew :app:tasks`

### [ ] 0.2 Add a minimal app shell and localhost network policy
Files:
- `android/app/src/main/AndroidManifest.xml`
- `android/app/src/main/java/com/nousresearch/hermesagent/MainActivity.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/HermesApplication.kt`
- `android/app/src/main/res/values/strings.xml`
- `android/app/src/main/res/xml/network_security_config.xml`
Do:
- create a minimal Compose activity
- add `android.permission.INTERNET` explicitly in the manifest
- allow cleartext only for `127.0.0.1` / `localhost`
- keep storage in app-private paths only
Verify:
- `cd android && ./gradlew :app:assembleDebug`

### [ ] 0.2a Add the first Android CI smoke lane
Files:
- `.github/workflows/android.yml`
Do:
- add a minimal Android workflow early instead of waiting until release work
- trigger it on `pull_request`, `workflow_dispatch`, and pushes to the current Android development branch while the lane is maturing
- install JDK + Android SDK, cache Gradle, and run the earliest safe command for the current stage:
  - first `cd android && ./gradlew :app:tasks`
  - then upgrade it to `cd android && ./gradlew :app:assembleDebug` as soon as the project scaffolding is stable
Verify:
- `gh run list --workflow android.yml --limit 1`
- `gh run view <run-id> --log | cat`

### [ ] 0.3 Add a dedicated Android Python dependency lane
Files:
- `pyproject.toml`
- `constraints-android.txt`
- `MANIFEST.in`
Do:
- add an `android` optional dependency set
- do not reuse `.[termux]` for the APK build
- explicitly include API-server runtime dependencies required by `gateway/platforms/api_server.py`, including `aiohttp`
- include the new Android Python package in setuptools discovery
- keep Termux install paths intact
Verify:
- `python -m pip install -e '.[android]' -c constraints-android.txt`
- `python -c "from gateway.platforms.api_server import APIServerAdapter; print('api-server import ok')"`

### [ ] 0.4 Add the Android Python package and packaging probe
Files:
- `hermes_android/__init__.py`
- `hermes_android/boot_probe.py`
- `android/app/src/main/java/com/nousresearch/hermesagent/backend/PythonBootProbe.kt`
- `android/app/build.gradle.kts`
Do:
- wire Chaquopy into the app module
- package the repo's Python code into the app
- add a trivial Python probe callable from Kotlin
Verify:
- `cd android && ./gradlew :app:assembleDebug`
- debug app shows the probe result in logcat or a temporary status view

### [ ] 0.5 Trim or isolate desktop-only Python deps only if they block APK packaging
Files:
- `pyproject.toml`
- `scripts/install.sh`
- `setup-hermes.sh`
- `constraints-android.txt`
Do:
- if APK packaging fails on desktop-only deps, move those deps behind extras instead of carrying them into the mobile runtime
- keep the current desktop and Termux paths working
Verify:
- `python -m pip install -e '.[termux]' -c constraints-termux.txt`
- `python -m pip install -e '.[android]' -c constraints-android.txt`

## Stage 1 — Boot Hermes locally through the API-server seam
The goal of this stage is a healthy embedded backend, not a polished UI.

### [ ] 1.1 Map app-private Hermes paths and runtime env
Files:
- `hermes_android/runtime_env.py`
- `hermes_android/bootstrap.py`
- `tests/hermes_android/test_runtime_env.py`
Do:
- create a bootstrap entrypoint that accepts app-private paths from Kotlin
- set `HERMES_HOME`, API server host/port/key, and any runtime flags needed for mobile boot
- keep config/state under app-private storage
Verify:
- `python -m pytest tests/hermes_android/test_runtime_env.py -q`

### [ ] 1.2 Add a local server runner around `gateway/platforms/api_server.py`
Files:
- `hermes_android/server.py`
- `hermes_android/bootstrap.py`
- `tests/hermes_android/test_server.py`
Do:
- start only the local API server adapter instead of the full gateway multi-platform runner
- keep the host loopback-only
- generate an internal bearer key even for local use
Verify:
- `python -m pytest tests/hermes_android/test_server.py -q`

### [ ] 1.2a Bundle built-in skills and sync them into app-private storage
Files:
- `hermes_android/bootstrap.py`
- `hermes_android/bundled_assets.py`
- `tools/skills_sync.py`
- `MANIFEST.in`
- `tests/hermes_android/test_bundled_skills.py`
Do:
- ensure `skills/` and `optional-skills/` are packaged into the Android Python payload
- extract bundled skills into the app runtime in a deterministic location
- set `HERMES_BUNDLED_SKILLS` for bundled-skill sync and `HERMES_OPTIONAL_SKILLS` for optional-skill discovery when the Android bootstrap path is active
- call the existing sync/seed path during Android bootstrap so bundled skills appear under app-private `HERMES_HOME`
- keep non-Android behavior unchanged
Verify:
- `python -m pytest tests/hermes_android/test_bundled_skills.py -q`
- verify built-in skills exist under the Android app-private Hermes home after bootstrap
- verify optional skills are discoverable from the extracted Android runtime path

### [ ] 1.3 Seed mobile defaults and harden the app tool profile fallback
Files:
- `toolsets.py`
- `hermes_android/mobile_defaults.py`
- `gateway/platforms/api_server.py`
- `tests/hermes_android/test_mobile_defaults.py`
- `tests/gateway/test_api_server_android_toolset.py`
Do:
- add `hermes-android-app`
- seed `platform_toolsets.api_server` to the new toolset on first run
- if Android bootstrap detects missing or invalid API-server toolset config, hard-force `hermes-android-app` instead of falling back to `hermes-api-server`
- keep `hermes-api-server` unchanged for non-mobile users
Verify:
- `python -m pytest tests/hermes_android/test_mobile_defaults.py tests/gateway/test_api_server_android_toolset.py -q`
- include a missing-config regression case proving Android does not leak into the full `hermes-api-server` toolset

### [ ] 1.4 Add Android-side runtime lifecycle management
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeManager.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/HermesApplication.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/backend/HermesRuntimeManagerTest.kt`
Do:
- start Python once per app process
- boot the local server once
- expose boot state, port, and auth key to the rest of the app
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 1.5 Add a boot-status screen with `/health` validation
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/boot/BootViewModel.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/boot/BootScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/MainActivity.kt`
Do:
- poll `/health` before entering chat UI
- show actionable boot errors instead of a blank screen
Verify:
- `cd android && ./gradlew :app:installDebug`
- `adb shell am start -n com.nousresearch.hermesagent/.MainActivity`
- confirm the UI shows a ready state only after `/health` returns 200, or shows an explicit boot error state instead of a blank screen

## Stage 2 — Build the native chat client over the local API
Keep the Android UI thin. Do not reimplement Hermes logic in Kotlin.

### [ ] 2.1 Define local API DTOs and client primitives
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/api/HermesApiModels.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/api/HermesApiClient.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/api/HermesApiClientTest.kt`
Do:
- model `/health`, `/v1/models`, and `/v1/chat/completions`
- centralize loopback base URL + bearer auth handling
- centralize `X-Hermes-Session-Id` handling for multi-turn continuity
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 2.2 Implement SSE chat streaming over chat-completions
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/api/HermesSseClient.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/api/HermesSseClientTest.kt`
Do:
- consume OpenAI chat-completion SSE chunks from `POST /v1/chat/completions` with `stream=true`
- treat `chat.completion.chunk` deltas and the terminal `[DONE]` marker as the MVP streaming contract
- treat tool progress as inline content/tool deltas from the current API-server behavior, not as a guaranteed separate event class
- support token streaming and final completion events
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 2.3 Add chat state and ViewModel
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatState.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/ui/chat/ChatViewModelTest.kt`
Do:
- keep UI state native-side
- treat the Python backend as the source of truth for conversation/tool state
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 2.4 Add the Compose chat screen
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/theme/Color.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/theme/Theme.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/theme/Type.kt`
Do:
- render messages, streaming partials, loading state, and failure state
- keep the UI mobile-native instead of terminal-like
Verify:
- `cd android && ./gradlew :app:assembleDebug`

### [ ] 2.5 Persist chat session identity using `X-Hermes-Session-Id`
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/data/ConversationStore.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatViewModel.kt`
- `android/app/src/androidTest/java/com/nousresearch/hermesagent/ConversationResumeTest.kt`
Do:
- generate and store a stable per-conversation `X-Hermes-Session-Id`
- reuse that session id after app restarts so backend conversation history stays server-side
- avoid duplicating long conversation history in native code
Verify:
- `cd android && ./gradlew :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.nousresearch.hermesagent.ConversationResumeTest`

## Stage 3 — Replace CLI setup/auth flows with native mobile settings
Do not attempt to reuse `hermes setup` or any prompt-driven CLI UI in the APK.

### [ ] 3.1 Add native app settings storage
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/data/AppSettingsStore.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/data/AppSettingsStoreTest.kt`
Do:
- store non-secret settings natively
- keep UI configuration separate from backend boot state
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 3.2 Add native secret storage for provider keys
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/data/SecureSecretsStore.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/data/SecureSecretsStoreTest.kt`
Do:
- store API keys in Android encrypted storage
- inject secrets into Python at boot time instead of hard-requiring plaintext `.env`
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 3.3 Add Python bridges for config/auth compatibility
Files:
- `hermes_android/config_bridge.py`
- `hermes_android/auth_bridge.py`
- `tests/hermes_android/test_config_bridge.py`
Do:
- reuse `hermes_cli/config.py` and `hermes_cli/auth.py` semantics where practical
- expose small helper functions for the Android app instead of calling CLI commands
Verify:
- `python -m pytest tests/hermes_android/test_config_bridge.py -q`

### [ ] 3.4 Add onboarding and settings screens
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt`
Do:
- support provider preset, base URL, model, and API key entry
- include safe defaults for Nous, OpenAI, OpenRouter, and custom OpenAI-compatible endpoints
Verify:
- `cd android && ./gradlew :app:installDebug`
- `adb shell am start -n com.nousresearch.hermesagent/.MainActivity`
- enter a provider preset, base URL, model, and API key, then confirm the values survive process restart

### [ ] 3.5 Support runtime restart after settings changes
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeManager.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt`
Do:
- restart the embedded backend cleanly when config changes require it
- keep the UI responsive during restart
Verify:
- `cd android && ./gradlew :app:installDebug`
- `adb shell am start -n com.nousresearch.hermesagent/.MainActivity`
- change model/base URL/key and confirm the app returns to a healthy `/health` state after restart

## Stage 4 — Lock in the mobile-safe tool contract
Keep the first APK honest about what it can and cannot do.

### [ ] 4.1 Document the app tool profile in code and docs
Files:
- `toolsets.py`
- `website/docs/getting-started/android-app.md`
- `website/sidebars.ts`
Do:
- document the MVP allowlist and denylist
- clearly distinguish APK support from Termux support
Verify:
- `cd website && npm run build`

### [ ] 4.2 Show the active tool profile in the settings UI
Files:
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/ToolProfileCard.kt`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt`
Do:
- show users which tools are enabled in the APK MVP
- explicitly label shell-heavy tools as not included in the first mobile release
Verify:
- `cd android && ./gradlew :app:installDebug`
- `adb shell am start -n com.nousresearch.hermesagent/.MainActivity`
- confirm the settings screen shows the MVP allowlist and labels blocked tool classes clearly

### [ ] 4.3 Add regression tests so blocked tools do not leak into mobile defaults
Files:
- `tests/gateway/test_api_server_android_toolset.py`
- `tests/hermes_android/test_mobile_defaults.py`
Do:
- assert that terminal/process/file/browser/voice tools stay out of the mobile default toolset
Verify:
- `python -m pytest tests/gateway/test_api_server_android_toolset.py tests/hermes_android/test_mobile_defaults.py -q`

## Stage 5 — Test, harden, and make the APK release-worthy

### [ ] 5.1 Add Python unit tests for Android bootstrap paths
Files:
- `tests/hermes_android/test_runtime_env.py`
- `tests/hermes_android/test_server.py`
- `tests/hermes_android/test_config_bridge.py`
Do:
- cover startup env, local server boot, and config bridge behavior
Verify:
- `python -m pytest tests/hermes_android -q`

### [ ] 5.2 Add Android unit tests for runtime, API, and chat state
Files:
- `android/app/src/test/java/com/nousresearch/hermesagent/backend/HermesRuntimeManagerTest.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/api/HermesApiClientTest.kt`
- `android/app/src/test/java/com/nousresearch/hermesagent/ui/chat/ChatViewModelTest.kt`
Do:
- keep fast feedback for most Android changes
Verify:
- `cd android && ./gradlew :app:testDebugUnitTest`

### [ ] 5.3 Add one instrumentation smoke test
Files:
- `android/app/src/androidTest/java/com/nousresearch/hermesagent/BootSmokeTest.kt`
Do:
- launch the app
- wait for backend health
- verify the app reaches chat-ready state
Verify:
- `cd android && ./gradlew :app:connectedDebugAndroidTest`

### [ ] 5.4 Add release build hardening
Files:
- `android/app/build.gradle.kts`
- `android/app/proguard-rules.pro`
- `android/app/src/main/res/xml/backup_rules.xml`
- `android/app/src/main/res/xml/data_extraction_rules.xml`
Do:
- define release build type
- keep backup/export rules explicit
- make release builds reproducible in CI
Verify:
- `cd android && ./gradlew :app:assembleRelease`
- the same release build command is exercised by Android CI without relying on local-only files

### [ ] 5.5 Add version and branding wiring for release artifacts
Files:
- `android/app/build.gradle.kts`
- `android/app/src/main/res/values/strings.xml`
- `scripts/release.py`
- `README.md`
Do:
- wire Android versioning to the existing repo release process
- use Python package/release metadata as the single source of truth for `versionName`
- derive Android `versionCode` deterministically from the existing CalVer release tag format:
  - `vYYYY.M.D` -> `YYYYMMDD00`
  - `vYYYY.M.D.N` -> `YYYYMMDDNN`
- add a short README note that Android APK is now supported separately from Termux
Verify:
- `python scripts/release.py --help`
- `cd android && ./gradlew :app:assembleRelease`

## Stage 6 — Add CI build, signing, and GitHub release upload
This is the release cut line for the first public APK.

### [ ] 6.1 Expand the early Android workflow into a full PR/push gate
Files:
- `.github/workflows/android.yml`
Do:
- extend the Stage 0 CI smoke lane instead of replacing it
- build debug APK
- run Android unit tests
- upload the debug APK as a workflow artifact
Verify:
- `gh run list --workflow android.yml --limit 1`
- `gh run view <run-id> --log | cat`
- the run uploads a debug APK artifact

### [ ] 6.2 Add a signed release workflow for Android artifacts
Files:
- `.github/workflows/android-release.yml`
Do:
- trigger on GitHub release publish
- build release APK + AAB
- sign artifacts from GitHub secrets
- upload artifacts to the GitHub release
Verify:
- publish a test prerelease
- `gh run list --workflow android-release.yml --limit 1`
- `gh run view <run-id> --log | cat`
- `gh release download <tag> -p '*.apk' -p '*.aab' -p '*.sha256'`

### [ ] 6.3 Add signing templates and ignore rules
Files:
- `android/keystore.properties.example`
- `.gitignore`
- `website/docs/developer-guide/android-release.md`
- `website/sidebars.ts`
Do:
- document the required secrets:
  - `ANDROID_KEYSTORE_BASE64`
  - `ANDROID_KEY_ALIAS`
  - `ANDROID_KEYSTORE_PASSWORD`
  - `ANDROID_KEY_PASSWORD`
- ignore local signing files and Android-local machine files such as `local.properties`
Verify:
- `cd website && npm run build`
- `git check-ignore -v android/local.properties android/keystore.properties || true`

### [ ] 6.4 Add artifact renaming + checksum generation
Files:
- `scripts/android_release_manifest.py`
- `.github/workflows/android-release.yml`
Do:
- rename artifacts to stable release-friendly names
- emit SHA256 checksums for every Android artifact
Verify:
- `python scripts/android_release_manifest.py --help`
- release workflow uploads APK/AAB plus `.sha256` files

### [ ] 6.5 Update the existing release script to acknowledge Android CI artifacts
Files:
- `scripts/release.py`
- `README.md`
Do:
- keep the current release flow intact
- make release notes/output mention that Android artifacts are generated by CI after the GitHub release is published
Verify:
- `python scripts/release.py --help`
- dry-run output references Android release artifacts clearly

## Stage 7 — Post-MVP expansion toward fuller mobile coding support
Do not block the first APK release on this stage.

### [ ] 7.1 Extract a pluggable file-backend interface from desktop file tools
Files:
- `tools/file_backends/base.py`
- `tools/file_backends/local_fs.py`
- `tools/file_operations.py`
Do:
- separate filesystem backend logic from the existing file tools
- keep desktop behavior unchanged
Verify:
- `python -m pytest tests/tools -q`

### [ ] 7.2 Add an Android document-tree workspace bridge
Files:
- `tools/file_backends/android_documents.py`
- `hermes_android/workspace_bridge.py`
- `android/app/src/main/java/com/nousresearch/hermesagent/ui/workspace/WorkspacePicker.kt`
Do:
- use Android's Storage Access Framework instead of raw POSIX assumptions
- support explicit user-granted workspace roots only
Verify:
- manual smoke test with a picked document tree

### [ ] 7.3 Add an opt-in workspace toolset for the app
Files:
- `toolsets.py`
- `hermes_android/mobile_defaults.py`
- `tests/gateway/test_api_server_android_workspace_toolset.py`
Do:
- add a separate opt-in workspace profile instead of changing the safe default
Verify:
- `python -m pytest tests/gateway/test_api_server_android_workspace_toolset.py -q`

### [ ] 7.4 Document the workspace path separately from the chat-first MVP
Files:
- `website/docs/getting-started/android-app.md`
- `website/docs/reference/toolsets-reference.md`
Do:
- keep the docs honest about what is stable vs experimental
Verify:
- `cd website && npm run build`

## Recommended Verification Commands
Use these as the default verification set while implementing the plan.

Python/backend:
- `python -m pytest tests/hermes_android -q`
- `python -m pytest tests/gateway/test_api_server_android_toolset.py -q`
- `python -m pytest tests/gateway/test_api_server.py tests/gateway/test_api_server_toolset.py -q`

Android local:
- `cd android && ./gradlew :app:assembleDebug`
- `cd android && ./gradlew :app:testDebugUnitTest`
- `cd android && ./gradlew :app:connectedDebugAndroidTest`
- `cd android && ./gradlew :app:assembleRelease :app:bundleRelease`

Docs:
- `cd website && npm run build`

Release:
- `python scripts/release.py --bump patch`
- `sha256sum android/app/build/outputs/apk/release/*.apk android/app/build/outputs/bundle/release/*.aab`

## Guardrails For Subagents
- Preserve `gateway/platforms/api_server.py` as the main seam.
- Reuse `run_agent.py`, `agent/*`, `hermes_constants.py`, `hermes_state.py`, `hermes_cli/auth.py`, and `hermes_cli/config.py` before inventing new runtime layers.
- Do not try to render the existing CLI inside Android.
- Prefer a thin native UI over a local Hermes API rather than duplicating agent logic in Kotlin.
- Keep `HERMES_HOME` app-private.
- Keep the loopback API local-only with a generated bearer key.
- Keep the default Android toolset small and honest.
- Get CI debug builds green before working on signing/release polish.
- Stop the first public release at Stage 6. Stage 7 is deliberately post-MVP.
