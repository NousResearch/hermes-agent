# Android Auth + Alpha Release Plan

> For Hermes: use subagent-driven-development skill to implement this plan task-by-task.

Goal: add an Android in-app auth foundation for Corr3xt-backed sign-in methods (email, Google, phone, ChatGPT, Claude, Gemini), persist provider credentials into Hermes runtime config/env, and extend Android release plumbing for a signed v0.0.1-alpha release workflow.

Architecture:
- Add a first-class Android Accounts/Auth section with local session state, deep-link callback handling, and a generic Corr3xt auth launcher.
- Extend hermes_android Python bridges so Android UI can write/read provider-specific auth bundles rather than only a single API key.
- Keep release work separate: support semver alpha tags in Gradle versioning and use GitHub Actions release workflow for signed APK/AAB publishing.

Tech stack:
- Android Compose UI + SharedPreferences/EncryptedSharedPreferences
- CustomTabs/browser deep-link auth callbacks
- Existing hermes_android Python bridges
- GitHub Actions Android release workflow

---

## Shortcomings found

1. Android app currently has no auth/onboarding section; only Hermes, Nous Portal, and Settings tabs exist.
2. Settings only supports provider/base URL/model/API key entry; no provider-specific OAuth/session handling.
3. Android manifest has no deep-link callback intent-filter, so in-app OAuth redirect completion is impossible.
4. There is no Corr3xt client or any email/google/phone sign-in flow in the app.
5. ChatGPT/Claude/Gemini auth exists in Hermes CLI/runtime concepts, but the Android app only exposes generic API-key storage.
6. Signed Android release CI exists, but app versioning is not semver-alpha aware and there is no attempted v0.0.1-alpha release path yet.

## Task 1: Add Android auth data models and session storage
- Create auth models for methods/providers/session state.
- Create SharedPreferences-backed auth session store.
- Include Corr3xt base URL + pending auth request state.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/data/AuthModels.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/data/AuthSessionStore.kt`

## Task 2: Add Corr3xt auth launcher and callback parsing
- Build generic start URLs for methods: email, google, phone, chatgpt, claude, gemini.
- Parse callback URIs delivered back into the app.
- Persist auth result payloads.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/auth/Corr3xtAuthClient.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/auth/AuthCallbackParser.kt`

## Task 3: Add provider-aware Python auth bridge methods
- Extend `hermes_android/auth_bridge.py` beyond one API key.
- Support writing/reading ChatGPT Web session/access tokens, Anthropic token/API key, Gemini/Google API key, and generic provider auth status.

Files:
- Modify: `hermes_android/auth_bridge.py`
- Modify: `tests/hermes_android/test_config_bridge.py`
- Create: `tests/hermes_android/test_auth_bridge_extended.py`

## Task 4: Add Android Accounts screen + ViewModel
- Add first-class Accounts tab.
- Show sign-in cards for email, Google, phone, ChatGPT, Claude, Gemini.
- Launch Corr3xt auth for selected method.
- Apply returned auth bundle into Hermes runtime and provider settings.

Files:
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthViewModel.kt`
- Create: `android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt`

## Task 5: Handle deep-link auth callbacks in Android activity/manifest
- Add callback intent-filter.
- Handle incoming auth redirect in `MainActivity` and hand it to the auth session store.

Files:
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/MainActivity.kt`

## Task 6: Sync auth state into Settings/runtime
- Settings should load provider auth state if available.
- Auth screen should be able to switch active Hermes provider automatically after login.

Files:
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsViewModel.kt`
- Modify: `android/app/src/main/java/com/nousresearch/hermesagent/data/ProviderPresets.kt`

## Task 7: Add semver alpha Android release versioning
- Support tags like `v0.0.1-alpha` and map them to Android `versionName`/`versionCode`.
- Keep existing date-tag support compatible.

Files:
- Modify: `android/app/build.gradle.kts`
- Add/extend tests if present for release artifact naming/version mapping.

## Task 8: Extend Android release workflow ergonomics
- Keep signed release workflow.
- Allow manual or tag-based alpha release attempts if needed.
- Ensure artifact names include tag.

Files:
- Modify: `.github/workflows/android-release.yml`
- Modify if needed: `scripts/android_release_manifest.py`

## Task 9: Validate locally
Commands:
- `source .venv/bin/activate && python -m pytest tests/hermes_android tests/hermes_cli/test_setup.py -q`
- `source .venv/bin/activate && python -m pytest tests/ -q`
- `cd android && export PYTHON_FOR_BUILD="$(command -v python3.11 || command -v python3 || command -v python)" && ./gradlew :app:installDebugPythonRequirements`

## Task 10: Push and attempt signed alpha release CI
- Push branch.
- Verify Android CI green.
- If signing secrets exist, create/publish prerelease tag `v0.0.1-alpha` and watch release workflow.
- If signing secrets are absent, report exact blockers after attempting.

Files/commands:
- `gh run list --repo adybag14-cyber/hermes-agent --branch feat/termux-install-path --limit 5`
- `gh release create v0.0.1-alpha --prerelease --title "v0.0.1-alpha" --notes "Android alpha release"`

## Risks / open questions
- Corr3xt service API contract is not present in the repo; implementation will assume browser/deep-link flow with callback query params unless more detail is discovered.
- Email/google/phone login may require external backend verification beyond what can be fully validated locally.
- Signed release attempt depends on GitHub secrets: `ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`.
