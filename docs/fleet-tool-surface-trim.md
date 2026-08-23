# Fleet tool-surface trim

This is an operator configuration profile, not a change to Hermes source
code. It documents the deliberately small tool surface used for the fleet's
interactive CLI and Telegram sessions.

## Scope

Set `platform_toolsets.cli` and `platform_toolsets.telegram` to:

```yaml
- terminal
- file
- web
- memory
- cronjob
- clarify
```

Leave the `cron` platform's configured toolsets unchanged. Verify it still
resolves `terminal` before and after applying the profile.

The selected surface retains `terminal`, `process`, `fast_shell`, `patch`,
`write_file`, `search_files`, `web_search`, and `web_extract`. `fast_shell`
is registered with the terminal toolset in this fork.

## Global disable guard

Set `agent.disabled_toolsets` from the installed checkout's `TOOLSETS` map;
do not maintain a hand-written inventory. Disable every non-kept toolset whose
static resolution has no overlap with the kept toolsets, plus Hermes platform
bundles and posture toolsets (which have special core-preserving subtraction
semantics).

Do **not** globally disable overlapping aliases such as `browser`,
`debugging`, `safe`, or `search`: their definitions contain one or more kept
tools, so global subtraction would remove those tools. They are nevertheless
absent from CLI and Telegram because those platforms use the explicit list
above, and therefore contribute no schemas there.

## Rollout and verification

1. Back up `~/.hermes/config.yaml` with a dated suffix.
2. Apply the profile using the target host's installed `TOOLSETS` map.
3. Restart that host's Hermes gateway using its native service manager.
4. Confirm all eight retained tools above resolve on CLI and Telegram, then
   verify cron resolves `terminal`.
5. Run the host's barometric-pressure monitor and prompt-agent cron jobs when
   present, and confirm their execution records complete.

Rollback is configuration-only: restore the dated backup and restart the
gateway.

## Reference measurement

On the reference Telegram configuration (Qwen 3.6), the profile reduced tool
schemas from 33 to 14 and the system prompt from 4,514 to 2,533 tokens.
