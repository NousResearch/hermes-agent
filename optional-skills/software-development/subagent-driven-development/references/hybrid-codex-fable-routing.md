# Hybrid Codex + Claude/Fable Routing Pattern

Use when a project should mostly be implemented by Codex, but a scarce higher-capability model is reserved for genuinely hard sections.

## Pattern

1. Verify live auth/model availability before routing tasks.
   - `codex --version && codex login status`
   - Smoke-test Codex models with a no-op prompt, e.g. `gpt-5.3-codex-spark` and `gpt-5.5`.
   - `claude auth status --text && claude --version`
   - Smoke-test Claude/Fable with a no-op prompt: `claude -p "Print exactly OK" --model fable --max-turns 3`.

2. Route by difficulty, not by habit.
   - Codex `gpt-5.3-codex-spark`: mechanical config, docs, scripts, small helpers.
   - Codex `gpt-5.5`: normal hard implementation, multi-file firmware, protocol code, exporters, harnesses.
   - Claude Code `fable`: scarce/high-power lane for sections where reasoning risk dominates implementation volume.

3. Create a routing doc in the repo before dispatching agents.
   - Include live smoke-test evidence.
   - List phase/task -> model route.
   - Explicitly state fallback if Fable is unavailable.
   - State controller verification commands.

4. Controller remains source of truth.
   - Agents write code; controller runs builds/tests/hardware receipts.
   - Hardware flashing/runs should be controller-owned unless the user explicitly delegates the side effect.
   - Agent self-reports are not receipts.

## Fable-worthy examples from ESP32-S3 cluster work

Use Fable for:
- tensor-parallel correctness and synchronization design;
- int4 sharded matmul correctness;
- full H256 cluster inference where numeric drift and token barriers interact;
- transformer runtime primitives and distributed execution;
- TinyStories 33M memory/shard feasibility and exporter design.

Keep Codex for:
- PlatformIO env additions;
- packet framing boilerplate;
- dry-run flash helpers;
- docs/receipt stubs;
- simple Python harnesses.

## Invocation examples

Codex mechanical:

```bash
cat /tmp/task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /path/to/repo \
  -m gpt-5.3-codex-spark \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-task.log | tail -150
```

Codex normal hard:

```bash
cat /tmp/task.md | codex exec \
  --dangerously-bypass-approvals-and-sandbox \
  -C /path/to/repo \
  -m gpt-5.5 \
  -s danger-full-access \
  2>&1 | tee /tmp/codex-task-gpt55.log | tail -150
```

Claude/Fable scarce lane:

```bash
cat /tmp/task.md | claude -p "$(cat /tmp/task.md)" \
  --dangerously-skip-permissions \
  --max-turns 30 \
  --model fable \
  --add-dir /path/to/repo \
  2>&1 | tee /tmp/claude-fable-task.log
```

Do not use `--bare` with Claude Pro/OAuth auth; it can skip OAuth context.
