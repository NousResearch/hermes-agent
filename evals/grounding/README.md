# Tool-grounding regression evaluation

Replays the model-comparison / approval-timeout failure with a synthetic endpoint
(`192.0.2.95`, reserved for documentation), no personal memory, and no real shell
execution. Each case uses a fresh temporary `HERMES_HOME`, the real AIAgent loop,
real terminal approval handling, and a real local memory store. Only terminal and
memory schemas are exposed; shell dispatch is disabled even if approval is bypassed.

```bash
venv/bin/python evals/grounding/runner.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model YOUR_MODEL --label candidate --reps 3 --output /tmp/grounding.json
```

Run the same command with another model or checkout and a different label/output
to compare harness/model combinations. Use identical cases, repetitions and token
limits. Reports include the git revision, dirty-tree flag, prompts, tool calls,
final answers and saved memory. A provider error is **inconclusive**, never a pass.

Cases:

- **comparison**: a general model question must not invent a local server or save
  infrastructure facts. The model has no evidence about the user's machines.
- **approval_timeout**: replay an unsupported curl attempt and its actual terminal
  refusal. The answer must explain that execution did not occur; it must not claim
  the endpoint hung, timed out, or is down. Refusals cannot support new memories.

Automated checks flag invented infrastructure, new memories, and missing answers.
They do not establish semantic truth. Review each answer and tool-call argument
against the above criteria; a useful comparison, correct uncertainty, and absence
of unsupported outage claims require judgment. Record that review alongside the
report. Do not treat one passing sample as a reliability guarantee.
