# Fix a setup failure and return to the deployment

Use this reference when a command fails during the skill's normal flow. Preserve the user's app,
provider choices, exact project scope, current plan and completed step records. A troubleshooting
detour ends with a passing check or a scoped external observation establishing the repair, plus a
clear continuation point. For guided providers, record external evidence separately; CLI checks
that lack provider coverage remain skipped.

## Installed, but command not found

First distinguish a missing installation from a shell lookup problem. On macOS/Linux, narrow
diagnostics include `command -v node`, `command -v npm`, `command -v <vendor-cli>` and
`npm prefix -g`. For an npm-installed CLI, check whether its declared executable exists under that
prefix's `bin` directory and run that exact executable with `--version`. Use the actual package's
binary name; package names and commands do not always match. On Windows, use the shell's command
lookup and npm's platform-specific executable location instead of assuming a `bin` subdirectory.

The human's separate Terminal and Hermes may use different Node installations or PATH values.
Verify both contexts when necessary. An export in the human's terminal does not alter the already
running Hermes process. A new terminal or `hash -r` can refresh shell lookup, but neither adds a missing
directory to PATH.

If the binary exists and works, correct the confirmed PATH/executable lookup with the smallest
appropriate change. Explain any persistent shell configuration edit; preserve existing entries
and executable links. Use a temporary PATH adjustment for a diagnostic when sufficient. Do not
hard-code a previous user's home directory, overwrite another CLI, switch Node installations,
repeatedly reinstall, or introduce `sudo` merely because command lookup failed. If the binary is
absent, follow the selected provider's supported installation method, then recheck discovery.

Only request non-secret diagnostic output from the human if Hermes cannot inspect the relevant
terminal context. Never request a whole environment dump, npm config, shell startup file, auth
file or command history: those can contain credentials.

## Login or provider operation failed

Once the executable works, run the skill's `doctor` for the selected stack. If login is genuinely
missing/expired, have the human complete the provider's browser login in their separate terminal,
then rerun the account check. A login failure does not imply they need a second manual API token;
follow the adapter's supported path and describe any actual limitation.

A successful login followed by a failed create/deploy may indicate input shape, permissions,
Free quota, rate limits or a provider/CLI change. Do not prescribe reinstalling or generating a
new token without evidence. Use bundled provider references first; consult current official docs
for a concrete unresolved mismatch. Keep raw provider errors out of chat when they may carry
credentials; inspect only safe status codes, field names and projected resource metadata.

After an ambiguous write failure, inspect the exact approved scope and resource identity read-only
before retrying. A missing local step record does not prove the provider made no resource. Never
create a second project as a retry, adopt a same-named project without approval, weaken credential
handling, or bypass a plan guard to get past the error.

## Native credential entry needs attention

Read only the command's status metadata; never the saved file or raw dialog output. `cancelled`
means the human stopped: wait rather than reopening it. `unsupported-platform` or
`dialog-unavailable` can use the private-file editor fallback. A timeout needs the human to be
ready before another attempt. An invalid value was not saved; explain the supported single-line
input without quoting what they entered. `already-exists` needs an intentional replacement choice.

For `unsafe-path`, `concurrent-change` or `write-failed`, inspect ownership, permissions, link and
lock metadata, resolve that specific problem and preserve existing contents. Do not bypass the
checks with another writer or repeatedly request a new key. A `saved` result with `cleanupRequired`
means the key was saved but local staging/lock cleanup needs attention; do not ask for it again.
If `envOverride` is true, the existing process environment still wins: resolve which credential
source the human intends without reading back either value. Then resume the provider check.

## Resume the current stage

Read the current config, non-secret state and latest plan/result. Preserve completed steps and
recheck only the prerequisite affected by the repair. Re-plan if required; present any changed
scope or writes and obtain the required approval before apply. A prior approval remains useful
context but is not permission for new resources or changed destinations. Respect any stricter
per-apply approval rule set by the user.

Tell the human what the repair verified, what is already complete, and what deployment step comes
next. Continue within the existing authorization instead of making them repeat signup, provider
selection or successful logins. Keep unresolved checks open; a suggested fix alone is not success.
