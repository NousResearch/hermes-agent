## Android Auth Fallback And Gemma 4 MTP

This Android release fixes the provider sign-in failure path seen when the
Corr3xt auth host is unreachable, and updates the native LiteRT-LM runtime path
for current Gemma 4 mobile acceleration support.

- Checks the Corr3xt sign-in endpoint before opening the browser, so dead DNS or
  unreachable auth hosts stay inside Hermes with a localized error instead of
  sending the user to a browser failure page.
- Adds a one-tap API-key setup fallback on runtime provider cards. For Qwen,
  the fallback opens Settings with Qwen OAuth, `https://portal.qwen.ai/v1`, and
  `qwen3-coder-plus` already selected.
- Wraps Accounts page action buttons so translated labels remain fully tappable
  on phone-width screens and large font settings.
- Updates LiteRT-LM Android to `0.11.0`, declares the required optional
  `libvndksupport.so` and `libOpenCL.so` libraries, and enables Gemma 4
  speculative decoding for `gemma-4` LiteRT-LM model files.
- Reports the active LiteRT-LM accelerator and speculative-decoding state from
  the local OpenAI-compatible `/health` endpoint.
- Keeps the Android automation capability boundary described in Tasker/Shizuku
  terms: user-granted Shizuku remains required for privileged device actions,
  while normal Settings/API-key setup stays inside the app.

Validation:

- `python -m pytest tests/hermes_android -q`
- `:app:compileDebugKotlin :app:testDebugUnitTest`
- `:app:assembleDebug :app:assembleDebugAndroidTest`
- Direct emulator instrumentation for `BootSmokeTest` and
  `NativeAgentRuntimeSmokeTest`: `OK (4 tests)`.
- Emulator UI validation in Spanish: Qwen Corr3xt sign-in remains in Hermes with
  the localized DNS failure, and the Qwen API-key fallback opens Settings with
  the Qwen OAuth profile and model fields selected.
