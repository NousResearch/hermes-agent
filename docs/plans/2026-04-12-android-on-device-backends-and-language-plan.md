# Android On-Device Backends + Language Picker Implementation Plan

> For Hermes: use subagent-driven-development where possible, but continue directly in-controller if delegated tool access is flaky.

Goal: add real in-app on-device inference switching for llama.cpp and LiteRT-LM, plus a one-tap multilingual settings picker that immediately updates the Android UI.

Architecture:
- Preserve the existing local Hermes API server and agent loop.
- Add an on-device backend orchestration layer that can start either a local llama.cpp OpenAI-compatible server or a LiteRT-LM OpenAI-compatible proxy, then point the Hermes runtime config at that local endpoint.
- Preserve separate remote-provider settings so toggling local inference does not destroy the user’s API-based setup.
- Add app-language state to settings and drive Compose text through a shared translation layer so language changes apply immediately without restarting the app.

Tech stack:
- Android Compose + Kotlin view models/stores
- Existing Chaquopy/Python Hermes runtime
- Existing Android Linux subsystem assets
- Termux llama-cpp package inside the embedded Linux suite
- Google LiteRT-LM Android SDK
- Lightweight local HTTP server/proxy on Android for LiteRT-LM OpenAI-compatible bridging

---

### Task 1: Extend persisted app settings for on-device backends and language

Objective: preserve remote provider config while adding explicit local-backend and language preferences.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/data/AppSettingsStore.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt`
- Test: `tests/hermes_android/test_android_model_downloads.py`

Steps:
1. Add persisted fields for selected on-device backend and selected app language.
2. Keep existing provider/base URL/model fields as the user’s remote/default config.
3. Load/save the new fields in `SettingsViewModel`.
4. Add targeted tests that assert the new settings fields are present and wired into the settings state.

Verification:
- `python -m pytest tests/hermes_android/test_android_model_downloads.py -q`

### Task 2: Ship llama.cpp in the embedded Android Linux suite

Objective: make llama.cpp genuinely runnable inside the app-private Linux subsystem.

Files:
- Modify: `hermes_android/linux_assets.py`
- Possibly modify: `scripts/prepare_android_linux_assets.py`
- Test: `tests/hermes_android/test_android_linux_asset_pipeline.py`
- Test: `tests/hermes_android/test_linux_assets.py`

Steps:
1. Add `llama-cpp` to the embedded Linux asset package set.
2. Keep the current asset-manifest generation flow intact.
3. Update tests to assert the embedded Linux suite now includes llama.cpp package coverage.

Verification:
- `python -m pytest tests/hermes_android/test_linux_assets.py tests/hermes_android/test_android_linux_asset_pipeline.py -q`

### Task 3: Add Android-side local backend orchestration for llama.cpp and LiteRT-LM

Objective: start/stop the selected local backend and expose a stable local endpoint/model identity to Hermes.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/backend/OnDeviceBackendManager.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/backend/LlamaCppServerController.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/backend/LiteRtLmOpenAiProxy.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeManager.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/HermesApplication.kt`
- Modify: `android/app/build.gradle.kts`
- Test: `android/app/src/test/java/com/nousresearch/hermesagent/backend/HermesRuntimeManagerTest.kt`
- Test: new backend unit tests under `android/app/src/test/java/com/nousresearch/hermesagent/backend/`

Steps:
1. Add Google Maven repository dependency usage for LiteRT-LM Android plus a lightweight embedded HTTP server dependency for the proxy.
2. Implement llama.cpp startup against the preferred GGUF download using the embedded Linux subsystem and a stable localhost port.
3. Implement a LiteRT-LM-backed localhost OpenAI-compatible proxy with `/health`, `/v1/models`, and `/v1/chat/completions` support.
4. For LiteRT-LM, map OpenAI message history and tool schemas into LiteRT-LM `ConversationConfig` / `OpenApiTool` wrappers with manual tool-calling mode so Hermes still executes tools.
5. Have `HermesRuntimeManager.ensureStarted()` ensure the selected on-device backend is ready before the Python Hermes API server starts.

Verification:
- Android unit tests for controller/proxy logic
- smoke validation that selected backend yields a stable local base URL/model name in runtime state

### Task 4: Wire settings UI for on-device backend switches and preferred local model readiness

Objective: expose obvious, honest toggles for llama.cpp and LiteRT-LM in Settings.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsSection.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/LocalModelDownloadsViewModel.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/models/HermesModelDownloadManager.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/data/LocalModelDownloadStore.kt`
- Test: `tests/hermes_android/test_android_model_downloads.py`

Steps:
1. Add an “On-device inference” card with mutually exclusive switches for `llama.cpp (GGUF)` and `LiteRT-LM`.
2. Surface whether a compatible preferred local model exists for the selected backend.
3. Keep the existing Hugging Face download section but turn runtime target into a clearer GGUF vs LiteRT-LM choice.
4. Extend inspection/download copy so users see which backend each download targets.
5. Save settings so enabling a backend immediately starts the local backend and repoints Hermes to it.

Verification:
- `python -m pytest tests/hermes_android/test_android_model_downloads.py -q`

### Task 5: Add app-wide immediate language switching with flag grid in Settings

Objective: a single tap on a language flag updates visible app names/descriptions immediately and persists the choice.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesLanguage.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/boot/BootScreen.kt`
- Modify other shell/settings helper composables as needed
- Test: add/update Android UI string assertion tests under `tests/hermes_android/`

Steps:
1. Add supported app languages: English, Chinese, Spanish, German, Portuguese, French.
2. Add a settings language card with appropriately sized flag buttons and one-tap save/switch behavior.
3. Provide a shared translation object/composition local and route top-level app labels/descriptions through it.
4. Update key visible text on Hermes, Accounts, Portal, Device, Settings, onboarding/help, and download/backend cards.
5. Keep the language switch immediate and independent of the provider/config save button.

Verification:
- targeted pytest assertions for language selector and translated shell text

### Task 6: Validate, review, ship

Objective: run the relevant validation stack, review the diff, commit, push, monitor CI, and release the next Android alpha.

Files:
- Modify tests as needed
- Possibly update release notes/workflow inputs if release copy needs refreshing

Steps:
1. Run focused pytest coverage for Android shell/chat/device/settings/backend tests.
2. Run Android unit tests and packaging prerequisite task.
3. Review git diff and branch status.
4. Commit with a feature message.
5. Push to `fork/feat/termux-install-path`.
6. Monitor Android CI and Android release workflows.
7. Publish the next alpha release artifact when CI is green.

Verification commands:
- `source .venv/bin/activate && python -m pytest tests/hermes_android tests/gateway/test_api_server_android_toolset.py tests/hermes_android/test_mobile_defaults.py tests/test_toolsets.py tests/test_model_tools.py tests/tools/test_delegate_toolset_scope.py -q`
- `cd android && ./gradlew :app:testDebugUnitTest :app:installDebugPythonRequirements`

Remember:
- keep the existing remote provider configuration intact when local backends are enabled
- do not fake LiteRT-LM or llama.cpp status; only show “ready” when the backend actually starts
- keep the app honest about model-format compatibility: GGUF for llama.cpp, `.litertlm` for LiteRT-LM
- prioritize an actually runnable llama.cpp path first, then keep LiteRT-LM equally real rather than cosmetic
