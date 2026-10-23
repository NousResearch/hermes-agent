# Gemini CLI remote Google sign-in

Setup reference for a remote Hermes host. Authentication UX depends on the
installed release; consult the current [authentication guide](https://geminicli.com/docs/get-started/authentication/)
and inspect the actual prompt before sending input.

## Flow

1. Use Hermes `terminal` to verify `gemini --version` and `gemini --help`.
   Install only if needed and authorized; reuse existing authentication first.
2. Start `gemini` in an interactive PTY/tmux session in the intended directory.
   Inspect any workspace-trust prompt and obtain the necessary authorization
   before proceeding. Do not automatically trust a repository to reach login.
3. If Google sign-in is chosen, keep the same process alive. Some releases print
   a browser URL and wait for an authorization code; others require a different
   browser/redirect arrangement. Follow the flow actually displayed.
4. For a displayed code flow, give the login URL to the user privately and enter
   their returned one-time code only into that waiting process. Do not log,
   commit, or retain authorization codes, tokens, or credential files. Passwords
   and account security decisions belong in the user's secure sign-in flow.
5. After sign-in completes, verify the task can run and check for reported auth
   errors. A small headless smoke test may incur usage, so keep it within scope.

## Troubleshooting

- A headless auth error means the selected method is not ready; it is not proof
  the CLI installation is broken. Code `41` has been observed for missing auth,
  but read the actual error rather than relying on a fixed code across releases.
- Cached Google credentials, an approved `GEMINI_API_KEY`, or configured Vertex AI
  can support headless tasks. Ask which method to use if none is configured.
- Do not restart a waiting code-flow process after presenting its login URL;
  if it exits or times out, start a new flow and discard the expired code.
- Desktop secret-service warnings on remote hosts are not necessarily fatal.
  Check whether authentication completed before changing credential storage.
- Never dump the environment or auth settings into a captured pane or report.
