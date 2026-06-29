# Final Proposal: Undo Discord Bot-to-Bot Routing and Restore Normal Gateway Communication

Created: 2026-05-30T03:21:22Z
Finalized after oppositional review: 2026-05-30T03:35Z
Repository: `/Users/johngalt/.hermes/hermes-agent`
Local HEAD: `f8b511c2db09c346a78c00c002f57da328f19562` (`origin/main`)
Upstream HEAD checked: `a7421dc7d2f0659a016092db6fc154526c8734b3` (`upstream/main`)
GitHub reference: `Enragedsaturday/hermes-agent@0be254d567891af1944fa90ca134a006d0cb1705` — `feat: add control-plane approvals and bot decommission guards`

## Objective

Undo Discord-specific bot-to-bot routing procedures and the final-response routing guard. Restore Discord to a normal human communication gateway: receive authorized human messages, ignore other bots, and send final responses exactly as generated, subject only to ordinary delivery/media/retry behavior.

Discord is no longer an authoritative control plane. Any durable PM/worker orchestration, dispatch, approval, route, or lifecycle state must be handled through Hermes-owned local control-plane primitives, not Discord messages, not `BOT_MSG v1`, and not raw bot mentions in Discord.

## Boundary

This is a planning artifact, not an implementation. It is executable by a future agent, but it must not restart the live gateway or push code without explicit approval.

## Non-goals / guardrails

- Do not delete local control-plane DB work merely because it arrived in the same commit. Keep `hermes_cli/control_db.py`, approval persistence, route/dispatch primitives, and their tests unless evidence shows they are broken independently.
- Do not restart the live gateway during implementation without explicit approval.
- Do not remove ordinary Discord gateway controls: allowed users, allowed roles, required mention/free-response behavior, multi-agent mention silence for human messages, media handling, retries, and human approval buttons/text fallback.
- Do not preserve any production environment flag that can silently re-enable Discord bot-to-bot authority. If compatibility shims remain, they must be unregistered from model tools and return decommission errors.
- Do not delete generic helper classes such as `BotThreadMembershipTracker` / `BotLoopFuse` from `gateway/platforms/helpers.py` in this PR unless a repo-wide search proves they are unused and the deletion is separately reviewed.

## Evidence gathered

1. `gateway/platforms/base.py:1552-1559` defines bot-routing regexes for operational payloads, routing targets, and `send_bot_*` tool references.
2. `gateway/platforms/base.py:2988-3117` implements `_should_guard_discord_bot_final_response`, audit writing, and replacement of the real final response with `[ROUTING_GUARD] ...`.
3. `gateway/platforms/base.py:4026-4031` routes final text responses through `_send_text_response_with_routing_guard` instead of direct `_send_with_retry`.
4. `plugins/platforms/discord/adapter.py:719-975` contains bot-to-bot admission helpers: `DISCORD_ALLOWED_BOT_USERS`, bot-control channels, bot thread membership, loop fuse, malformed BOT_MSG reactions, BOT_MSG approval decision handling, and `_should_accept_bot_message`.
5. `plugins/platforms/discord/adapter.py:1139-1190` handles bot-authored messages specially: `DISCORD_ALLOW_BOTS`, BOT_MSG parsing, approval decisions, ACK reactions, body extraction, and bypassing human-user allowlists for admitted bots.
6. Review found more adapter runtime hooks that the first draft missed:
   - `_validate_outbound_bot_msg_v1()` and its call sites in ordinary `send()` / forum send still reject raw allowed-bot mentions.
   - `send_exec_approval()` uses `_discord_approval_notify_mentions()` for human/operator pings as well as prior supervisor-bot behavior.
7. `gateway/run.py:6614-6641` still treats `Platform.DISCORD` as a bot-admission surface via `DISCORD_ALLOW_BOTS`. That must be removed, or replay/test paths can still admit bot-authored Discord events.
8. `gateway/platforms/discord.py` re-exports private BOT_MSG helpers from the adapter and will break if the adapter helpers are removed without wrapper cleanup.
9. `plugins/platforms/discord/bot_msg_protocol.py` is a dedicated `BOT_MSG v1` protocol module.
10. `tools/send_message_tool.py:148-222` defines `send_bot_message` and `send_bot_approval_decision` schemas.
11. `tools/send_message_tool.py:249-570` contains legacy env gates and send handlers for Discord BOT_MSG dispatch.
12. `tools/send_message_tool.py:651-664` blocks ordinary `send_message` if the content raw-mentions an allowed Discord bot, even when legacy bot-to-bot is disabled.
13. `tools/send_message_tool.py:2206-2226` conditionally registers/deregisters the bot-routing tools by `HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT`.
14. Tests that encode removed behavior are broader than the first draft listed: `tests/gateway/test_discord_send.py`, `tests/gateway/test_discord_approval_notify_mentions.py`, `tests/gateway/test_discord_bot_auth_bypass.py`, and `tests/gateway/test_discord_free_response.py` also reference the old bot-control behavior.
15. Public docs/env references also need cleanup: `website/docs/reference/environment-variables.md`, `website/docs/user-guide/messaging/discord.md`, and Chinese i18n copies mention Discord bot-to-bot control env vars/behavior.
16. `docs/control-plane-discord-decommission-plan.md:47-50` already states Slice 6 target: remove default operational routing guard/bot authority behavior from normal Discord gateway path. Current code did not finish that slice.

## Implementation plan

### Phase 0 — isolate work

1. Confirm worktree and avoid unrelated local changes:
   ```bash
   git status --short --branch
   git diff --name-only
   ```
2. Create a branch:
   ```bash
   git switch -c undo-discord-bot-routing
   ```
3. If unrelated local modifications are present in files to edit, stop and ask for owner instruction before overwriting them.

### Phase 1 — remove final-response routing guard from production path

Target: `gateway/platforms/base.py`

1. Replace the final response call at `base.py:4026-4031` with ordinary direct send:
   ```python
   result = await self._send_with_retry(
       chat_id=event.source.chat_id,
       content=text_content,
       reply_to=_reply_anchor,
       metadata=_thread_metadata,
   )
   ```
2. Remove dead final-response guard code after the direct path is in place:
   - `_BOT_ROUTING_OPERATIONAL_RE`
   - `_BOT_ROUTING_TARGET_RE`
   - `_BOT_ROUTING_TOOL_RE`
   - `_should_guard_discord_bot_final_response`
   - `_write_discord_bot_routing_guard_audit`
   - `_send_text_response_with_routing_guard`
3. Remove imports that become unused only because of this guard. Verify with py_compile/lint; do not assume `hashlib`, `datetime`, `Path`, or `contextlib` are otherwise unused.

Acceptance for this phase: a Discord final response containing `ACTION_REQUIRED`, `approval_id:`, `Galt/default`, and `statute PM` is delivered unchanged; no `[ROUTING_GUARD]` text is sent; no `logs/routing_guard` file is created.

### Phase 2 — restore ordinary outbound Discord sending

Target: `tools/send_message_tool.py` and `plugins/platforms/discord/adapter.py`

1. Remove the `send_message_tool.py:651-664` raw-bot-mention rejection. Ordinary `send_message` must attempt normal Discord delivery even if text contains `<@bot_id>`.
2. Remove adapter-level outbound BOT_MSG validation that would still block ordinary sends:
   - remove `_validate_outbound_bot_msg_v1()`;
   - remove call sites in ordinary `send()` and forum/thread send paths;
   - remove now-unused imports/helpers such as `discord_content_mentioned_allowed_bots`, `discord_content_mentions_allowed_bot`, and `bot_msg_required_error` if no other runtime path uses them.
3. Preserve `_standalone_send_discord_message` ordinary-send signature compatibility. If legacy `bot_msg_internal` parameters exist, they must not affect ordinary sends. Prefer returning explicit legacy-disabled errors for bot-control-only branches rather than changing a public-ish signature in the same PR.
4. Keep `BasePlatformAdapter._is_terminal_send_error` intact. Remove only Discord’s override if it becomes equivalent to the base implementation after bot-routing errors are gone.

Acceptance for this phase: `send_message({target:"discord:...", message:"hello <@777>"})` calls the normal send path with the original text and does not return a bot-control error.

### Phase 3 — remove inbound Discord bot-to-bot authority

Target: `plugins/platforms/discord/adapter.py`

1. Reduce bot-authored `on_message` handling back to baseline:
   ```python
   if getattr(message.author, "bot", False):
       return
   ```
   Then keep the existing non-bot allowed-user/role logic untouched.
2. Remove `DISCORD_ALLOW_BOTS` / `DISCORD_ALLOWED_BOT_USERS` admission behavior from the adapter. These env vars must not authorize inbound bot messages.
3. Remove BOT_MSG parsing/ACK/reaction/control behavior from adapter runtime:
   - malformed BOT_MSG reaction path;
   - BOT_MSG body extraction into `message.content`;
   - `reply_expected=false` ACK reaction;
   - `approval_decision` handling;
   - bot thread invitation/membership registration;
   - loop fuse checks for bot-to-bot messages;
   - startup `DISCORD_BOT_MSG_PROTOCOL` version validation.
4. Remove adapter state members only if no longer used:
   - `_bot_threads`
   - `_bot_loop_fuse`
   - `_warned_unrestricted_bot_acceptance`
5. Preserve human/operator approval notifications. `_discord_approval_notify_mentions()` is not necessarily bot-only because it also reads `DISCORD_OPERATOR_MENTIONS`. Keep or rename that helper for human mentions, but remove supervisor-bot/BOT_MSG approval request text and any `approval_decision BOT_MSG` instructions.
6. Do not delete `send_exec_approval` human button/text-fallback behavior.

Acceptance for this phase: bot-authored Discord messages are ignored even if old env vars are set; no BOT_MSG is parsed, ACKed, or dispatched; human approval prompts still work.

### Phase 4 — remove runner-level Discord bot admission

Target: `gateway/run.py`

1. Remove Discord from the platform bot-admission map around `gateway/run.py:6614-6641` so `source.is_bot=True` Discord events cannot be authorized by `DISCORD_ALLOW_BOTS` at runner level.
2. Preserve bot admission for other platforms only if still intentionally supported there (e.g. Feishu), and test the distinction.

Acceptance for this phase: even if a Discord bot-authored event bypasses adapter filtering through replay/test/plugin paths, the runner does not treat `DISCORD_ALLOW_BOTS` as authority for it.

### Phase 5 — remove model-visible bot-control tools; quarantine direct compatibility shims if needed

Target: `tools/send_message_tool.py`

1. Permanently remove model tool registration for:
   - `send_bot_message`
   - `send_bot_approval_decision`
2. Remove `HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT` as a production registration switch. Setting it must not expose those tools.
3. Preferred end state: delete schemas/handlers/helpers for BOT_MSG send if repo-wide import cleanup is straightforward.
4. Safe first-PR compromise if imports/tests/scripts still call direct functions: keep direct compatibility shims for one release, but they must:
   - be unregistered from `tools.registry` always;
   - return explicit decommission/control-plane-DB errors regardless of env;
   - not call Discord;
   - be documented as deprecated quarantine, not active rollback.
5. Do not keep an env flag that re-enables active Discord bot-to-bot authority in production.

Acceptance for this phase: the model registry never exposes `send_bot_message` or `send_bot_approval_decision`, even if legacy env vars are set.

### Phase 6 — update compatibility wrapper and quarantine/remove protocol module

Targets:
- `gateway/platforms/discord.py`
- `plugins/platforms/discord/bot_msg_protocol.py`

1. Update `gateway/platforms/discord.py` so it does not re-export removed BOT_MSG helpers such as `_parse_discord_bot_msg_v1` or `_discord_bot_reply_false_reaction`. Keep only non-BOT_MSG compatibility exports that remain valid.
2. Prefer quarantine over deletion for `plugins/platforms/discord/bot_msg_protocol.py` in the first PR if direct imports remain. Add a top docstring:
   ```python
   """Deprecated Discord BOT_MSG helpers retained only for historical migration tests.
   Do not import in normal gateway/tool runtime.
   """
   ```
3. If repo-wide search proves no runtime/test imports need it, delete the module. Do not leave normal runtime imports either way.

Acceptance for this phase: `gateway.platforms.discord`, `plugins.platforms.discord.adapter`, and `tools.send_message_tool` import successfully; normal runtime files do not import `plugins.platforms.discord.bot_msg_protocol`.

### Phase 7 — tests

Rewrite tests to assert the new normal gateway behavior, not removed private helper behavior.

Affected tests to rewrite/delete:

- `tests/tools/test_discord_bot_to_bot_decommission.py`
- `tests/gateway/test_discord_bot_filter.py`
- `tests/gateway/test_discord_bot_msg_hardening.py`
- `tests/gateway/test_discord_send.py`
- `tests/gateway/test_discord_approval_notify_mentions.py`
- `tests/gateway/test_discord_bot_auth_bypass.py`
- `tests/gateway/test_discord_free_response.py`
- `tests/gateway/test_discord_imports.py` if wrapper imports change

Required regression tests:

1. Final response path:
   - Exercise actual `BasePlatformAdapter.handle_message()` / final delivery path, not a removed private guard helper.
   - Return text containing `ACTION_REQUIRED`, `approval_id:`, `Galt/default`, and `statute PM`.
   - Assert exact original text is sent and no routing guard audit directory/file is created.
2. Static runtime cleanup:
   - Runtime files do not contain `[ROUTING_GUARD]`, `BOT_ROUTING_GUARD`, `_send_text_response_with_routing_guard`, or `_should_guard_discord_bot_final_response`.
3. Discord bot-authored inbound messages:
   - Exercise real registered `on_message` handler path.
   - With `DISCORD_ALLOW_BOTS=all`, `DISCORD_ALLOWED_BOT_USERS=12345`, and legacy env set, a bot-authored message that raw-mentions this bot and contains `BOT_MSG v1` must not call `handle_message` and must not add ACK/error reactions.
   - Human allowed messages still reach `handle_message`.
4. Ordinary outbound `send_message`:
   - Patch delivery to avoid network.
   - Assert a message containing `<@777>` calls normal delivery with original text.
5. Tool registry:
   - `send_bot_message` and `send_bot_approval_decision` are absent even if `HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT=1` is set.
   - If direct compatibility functions remain, direct calls return decommission errors and never send.
6. Approval prompts:
   - Human approval buttons/text fallback still work.
   - Human/operator raw mentions still work if intentionally retained.
   - No approval prompt contains `BOT_MSG`, `approval_decision BOT_MSG`, or `send_bot_approval_decision` instructions.
7. Imports/quarantine:
   - `gateway.platforms.discord`, `plugins.platforms.discord.adapter`, and `tools.send_message_tool` import successfully.
   - Normal runtime files do not import `plugins.platforms.discord.bot_msg_protocol` if quarantined.
8. Control-plane DB:
   - Existing control-plane DB and approval persistence tests continue to pass; this PR must not regress the replacement authority plane.

### Phase 8 — docs and scripts

1. Update `docs/control-plane-discord-decommission-plan.md`: mark Slice 6 implemented by removing Discord routing guard and bot authority runtime paths.
2. Update public docs and i18n docs:
   - `website/docs/reference/environment-variables.md`
   - `website/docs/user-guide/messaging/discord.md`
   - `website/i18n/zh-Hans/.../environment-variables.md`
   - `website/i18n/zh-Hans/.../messaging/discord.md`
3. Remove or clearly mark obsolete env vars as ignored/deprecated:
   - `DISCORD_ALLOW_BOTS`
   - `DISCORD_ALLOWED_BOT_USERS`
   - `DISCORD_BOT_CONTROL_CHANNELS`
   - `DISCORD_BOT_MSG_*`
   - `HERMES_ENABLE_LEGACY_DISCORD_BOT_TO_BOT`
4. Remove or quarantine `scripts/discord_bot_matrix_smoke.py`; it is obsolete if it only tests Discord bot-to-bot routing.

## Verification commands

Syntax/import checks:

```bash
python3 -m py_compile \
  gateway/platforms/base.py \
  gateway/platforms/discord.py \
  gateway/run.py \
  plugins/platforms/discord/adapter.py \
  tools/send_message_tool.py

python3 - <<'PY'
import gateway.platforms.discord
import plugins.platforms.discord.adapter
import tools.send_message_tool
from tools.registry import registry
assert registry.get_entry("send_bot_message") is None
assert registry.get_entry("send_bot_approval_decision") is None
print("runtime import/registry checks ok")
PY
```

Targeted Discord tests:

```bash
pytest -q -o addopts='' \
  tests/tools/test_discord_bot_to_bot_decommission.py \
  tests/gateway/test_discord_bot_filter.py \
  tests/gateway/test_discord_bot_msg_hardening.py \
  tests/gateway/test_discord_send.py \
  tests/gateway/test_discord_approval_notify_mentions.py \
  tests/gateway/test_discord_bot_auth_bypass.py \
  tests/gateway/test_discord_free_response.py \
  tests/gateway/test_discord_imports.py
```

Control-plane replacement authority tests:

```bash
pytest -q -o addopts='' \
  tests/hermes_cli/test_control_db.py \
  tests/tools/test_control_plane_approvals.py
```

Broad smoke:

```bash
pytest -q -o addopts='' --maxfail=1 \
  tests/gateway \
  tests/tools \
  tests/hermes_cli
```

Optional runtime-string check after implementation, adjusted if quarantine shims intentionally remain:

```bash
python3 - <<'PY'
from pathlib import Path
runtime = [
    Path("gateway/platforms/base.py"),
    Path("gateway/platforms/discord.py"),
    Path("plugins/platforms/discord/adapter.py"),
    Path("tools/send_message_tool.py"),
]
needles = [
    "BOT_ROUTING_GUARD",
    "[ROUTING_GUARD]",
    "_send_text_response_with_routing_guard",
    "_should_guard_discord_bot_final_response",
]
for path in runtime:
    text = path.read_text()
    hits = [n for n in needles if n in text]
    if hits:
        raise SystemExit(f"{path}: stale routing-guard strings: {hits}")
print("no stale routing guard runtime strings")
PY
```

## Final acceptance criteria

- No final response path calls a routing guard.
- No final response can emit `[ROUTING_GUARD]`.
- No normal response delivery writes `~/.hermes/logs/routing_guard` audit files.
- Ordinary Discord `send_message` does not reject text solely because it raw-mentions a bot-looking ID.
- Discord adapter ignores bot-authored messages before model dispatch, regardless of old bot-control env vars.
- No BOT_MSG is parsed, ACKed, transformed into message body, or used to resolve approvals in normal Discord runtime.
- `send_bot_message` and `send_bot_approval_decision` are not registered or presented to models, regardless of env.
- Human Discord behavior remains intact: allowed human user/role checks, require-mention/free-response channels, multi-agent silence for human messages, media delivery, retries, approval buttons/text fallback, and intentional human/operator approval mentions.
- `gateway.platforms.discord`, `plugins.platforms.discord.adapter`, and `tools.send_message_tool` import cleanly.
- Existing control-plane DB tests pass; Discord is not needed for PM/worker authority.

## Rollback

Rollback is git revert of the implementation commit/PR. Do not introduce a production env flag that re-enables Discord authority. If compatibility shims are retained, they are rollback scaffolding only: unregistered, non-sending, and decommission-error-only.

Gateway restart remains a separate operational action requiring explicit owner approval.

## Oppositional review adjudication

Three reviewers were run against the draft:

1. Runtime correctness reviewer found missed active code paths: adapter outbound raw-bot-mention validation, `gateway/run.py` bot auth bypass, wrapper exports, additional tests/docs.
2. Minimal-safe-change reviewer objected to broad deletion where compatibility shims and generic hooks reduce risk. I accepted this for first-PR execution: remove active runtime authority, but quarantine direct shims if needed.
3. Test-strategy reviewer found the draft’s tests were not executable because they targeted private helpers slated for deletion. I accepted the amendment: tests must exercise real final delivery, real Discord `on_message`, patched normal send delivery, import/registry checks, and approval prompt behavior.

Net decision: **remove all production/runtime Discord bot-to-bot authority and the routing guard, but do not require destructive deletion of every historical helper in the first PR if that increases migration risk. Runtime behavior, model-visible tools, and tests must prove Discord is no longer a control plane.**
