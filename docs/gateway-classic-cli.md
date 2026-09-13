# Classic gateway client (initial migration)

Normal `hermes --cli`, `hermes chat`, `hermes chat -q '…' -Q`, and top-level
`hermes -z '…'` use the gateway authority. Direct `cli.main` chat launches also
hand off before constructing a local agent. Local bootstrap uses the canonical
ensure lifecycle and a single-use private control ticket, not service installation.
An explicit `HERMES_TUI_GATEWAY_URL` remains remote-only: failed connection or
authentication never launches a local replacement.

The client prints the persisted `Session:` ID to stderr. Resume with
`hermes --cli chat --resume ID`. Input is admitted with a fresh `input_id`; events
and final replies come from the authority. `/quit`, `/exit`, `/detach`, EOF and
closing the terminal detach only. `/stop` sends an execution-generation-fenced
interrupt. Pending approvals display `/approve ID once|deny|…`; clarification
uses `/answer ID TEXT`. The client uses the displayed prompt ID and its captured
generation, not a locally reconstructed waiter. Other slash commands explicitly
reject instead of running local mutations.

One-shot stdout contains only the final reply. Exit status is 0 for a completed
admission, 1 for failed execution/connection, 2 for unsupported frontend options,
3 for a pending control that requires interactive reattachment, and 130 for a
keyboard detach. A pending control does not imply cancellation. `-z` no longer
implicitly bypasses approval policy.

## Current parity limits — not full classic CLI parity

The gateway advertises its accepted creation parameters through `runtime.describe`.
Model, provider, reasoning, toolsets and max-turns are sent only when advertised;
otherwise explicitly requested options reject. Caller cwd is sent when advertised;
an old gateway lacking cwd support prints a warning that it uses its configured
execution directory, while explicit `--in` rejects. Resume retains existing
session policy; creation overrides on resume reject.

Images, skills, worktrees, checkpoints, pass-session-id, safe mode, ignored rules
or user config, YOLO, hook acceptance, continue/latest/title selection,
create-if-missing, no-restore-cwd, usage reports, run budgets, direct API credentials,
verbose/compact display and local tool listing are not implemented in this client
and reject explicitly. Full legacy presentation, slash registry parity, interactive
history editing, auto-reconnect, lost-ACK durable client journals, remote login
bootstrap, cold-owner resume and native Windows/macOS QA remain outstanding.
The client preserves an explicit remote URL's existing authentication mechanism;
it does not acquire or mint remote credentials.

`tests/hermes_cli/test_gateway_chat_native.py` exercises native tmux/PTY classic
fresh input, detach, persisted-ID resume and one-shot through the ordinary daemon
with a loopback model, isolated HOME/HERMES_HOME and caller cwd. Constructor
instrumentation records no client AIAgent, SessionDB or GatewayRunner. The test
also denies an invalid approval choice, then reconnects and consents before a real
terminal effect on an owned fixture. Stale-control/active-Stop native coverage and
native clarify interaction remain separate acceptance work.

For two-checkout compatibility testing, load the narrowly scoped pytest option
plugin and specify the disposable gateway peer's checkout (the normal runner's
hermetic environment deliberately removes arbitrary environment overrides):

```sh
scripts/run_tests.sh -j 1 --file-retries 0 tests/hermes_cli/test_gateway_chat_native.py \
  -p tests.hermes_cli.gateway_client_options --client-test-runtime-root=/path/to/runtime-checkout
```

The receipt explicitly reports `caller_cwd_effect_verified`. On a cwd-capable
runtime the native client submits model/toolsets/cwd and the real terminal writes
its actual execution directory into an owned temporary receipt. On an older
runtime, explicit `--in` must reject without that effect. Do not count the latter
as cwd parity. Both client and daemon use isolated test homes; this option does
not install a service or alter the selected checkout.
