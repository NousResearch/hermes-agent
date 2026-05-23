## Finishing Shared Routine Setup

Use this checklist when a user installed a shared routine and needs help before enabling it.

1. Confirm the target routine
- Use `get_routine_config(routine_id=...)`.
- Verify this is the routine the user just installed.

2. Keep it disabled during setup
- Do not enable early.
- The user should confirm final behavior before activation.

3. Resolve `{{share.*}}` placeholders
- Ask only for missing values.
- Use `update_routine_config` to replace placeholders in the prompt.
- If a placeholder represents a list, collect and apply it as one comma-separated value.
- This applies to any templated list field (not only recipient emails): do not split list placeholders into separate per-item fields unless the user explicitly asks.
- If the shared routine depended on a user-specific Email Recipient, explain that Email Recipient configs do not transfer across users.
- Guide the user to `/workflows > Email Recipients` to create their own Email Recipient, then add it to the installed routine before enabling external email sending.

4. Confirm execution mode safety
- Review routine `mode` and any tool-level `mode` overrides.
- Treat `send_email_to_user` as safe in this system: it only emails the routine owner, not third parties.
- For `send_email_to_user`, default to autonomous behavior (no tool-level HITL override) unless the user explicitly asks for additional approval.
- Do not recommend switching to routine-level HITL just because `send_email_to_user` is present.
- If changing mode/tool mode, explain impact and ask for confirmation first.

5. Final validation
- Re-read config with `get_routine_config`.
- Ensure no unresolved `{{share.*}}` placeholders remain.
- Summarize what will happen when enabled.

6. Enable on explicit user confirmation
- Use `toggle_routine_enabled(routine_id=...)` only after user says to proceed.
- Confirm that routine is now enabled.

7. Offer an immediate test run after enable
- If the user already asked to test, skip the prompt and start testing immediately.
- Otherwise use `present_options` with:
  - `Run Now` (primary)
  - `Maybe Later` (secondary)
- If the user picks Run Now, execute a real test with `invoke_routine` and then call `show_run_progress`.
- For trigger-specific test selection patterns (for example email/schedule context picking), activate the `routine-guide` skill using `use_skill` and follow its testing guidance.