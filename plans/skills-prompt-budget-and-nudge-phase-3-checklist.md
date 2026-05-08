# Phase 3 Wrap-up Checklist

After at least one minor version of dogfooding with `skills.index_v2: true` and
`skills.nudge_signals.enabled: true` set in personal configs:

- [ ] Flip default of `skills.index_v2` to `true` in `cli-config.yaml.example`
- [ ] Flip default of `skills.nudge_signals.enabled` to `true` in `cli-config.yaml.example`
- [ ] Bump default of `skills.creation_nudge_interval` from `15` to `50`
- [ ] Update README / docs / website skills section to reference v2 rendering
- [ ] Add TUI parity: `/skills nudge off|on` in `ui-tui/src/app/slash/commands/ops.ts`
- [ ] Add gateway parity: same intercept in the gateway slash dispatcher
- [ ] After one minor version with v2 default, remove the v1 rendering path
  (`_render_v1` in `agent/prompt_builder.py`) and the `index_v2` flag plumbing
  entirely as Phase 4 cleanup.
