# Gemini CLI Remote OAuth Code Flow

Setup note for running Gemini CLI on a headless/remote Hermes host.

## Observed Flow

1. Install and verify:
   ```bash
   npm install -g @google/gemini-cli
   gemini --version
   gemini --help | sed -n '1,80p'
   ```
2. A headless smoke test before auth can fail with JSON error code `41`:
   ```json
   {
     "error": {
       "message": "Please set an Auth method in your .../.gemini/settings.json or specify one of the following environment variables before running: GEMINI_API_KEY, GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_GENAI_USE_GCA",
       "code": 41
     }
   }
   ```
3. Running `gemini` interactively on a remote PTY may first show a workspace trust prompt. Choose the narrow trust scope needed for the task before continuing.
4. Selecting `Sign in with Google` can print a browser URL and then wait at:
   ```text
   Enter the authorization code:
   ```
   Relay the full URL to the user, keep the PTY/process alive, and submit the returned authorization code into the same session.

## Practical Hermes Pattern

```bash
# Verify install and auth state without leaking secrets
printf 'GEMINI_API_KEY set: '; [ -n "${GEMINI_API_KEY:-}" ] && echo yes || echo no
printf 'GOOGLE_GENAI_USE_VERTEXAI: '; echo "${GOOGLE_GENAI_USE_VERTEXAI:-unset}"
printf 'GOOGLE_GENAI_USE_GCA: '; echo "${GOOGLE_GENAI_USE_GCA:-unset}"
gemini -p 'Reply with exactly: GEMINI_CLI_SMOKE_OK' --output-format json
```

If the smoke test returns code `41`, choose one auth method: API key (`GEMINI_API_KEY`), Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=true` plus project/credentials), Gemini Code Assist/Google sign-in (`GOOGLE_GENAI_USE_GCA=true` or interactive `/auth` depending on version), or user settings.

## Pitfalls

- Do not conclude Gemini CLI is broken when the first headless run fails with code `41`; it means auth is not configured.
- Do not restart the interactive process after presenting the OAuth URL; the authorization code must be entered into the same waiting process.
- `org.freedesktop.secrets` warnings can appear on servers without a desktop secret service; they are not necessarily fatal if the CLI still prints an OAuth URL/code prompt.
