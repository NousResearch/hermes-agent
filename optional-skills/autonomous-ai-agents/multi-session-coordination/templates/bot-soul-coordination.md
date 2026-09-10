# Coordination blurb for Bot Mode SOUL.md files

Paste the block below into every bot's SOUL.md (New Agent → Advanced → SOUL.md,
or Edit Profile → Advanced). A bot is a full agent profile running CONCURRENTLY
with interactive sessions and other bots — this teaches it the same co-worker
protocol every session follows. Substitute `<botname>` and adapt the resource
examples to your machine. (`install.py` appends this block to every existing
`profiles/*/SOUL.md` automatically — `--no-wire-bots` opts out; the marker
line at the end is what makes re-runs idempotent and lets `status` audit
enrollment.)

Why SOUL.md and not env/config: bot-to-bot handoffs run as fresh
`hermes -p <bot> chat` invocations that inherit no environment — the SOUL.md is
the only carrier guaranteed to reach every run (same reason subagents get the
protocol via their prompt).

---

## Shared-resource coordination (Hermes co-worker protocol)

You share this machine with the user's interactive sessions, other bots, and
scheduled (cron) jobs. A coordination board tracks who is using shared
resources. You MUST consult it before mutating anything shared.

**Shared resources include** (adapt to your setup): remote GPU boxes
(`box:<host>`), the desktop UI (`ui:desktop`), singleton apps that allow only
one client at a time, shared project directories (`file:~/...`), a shared
skills/scripts tree, and the machine's MAIN agent memory store.

**The ONLY claim-free exemption** is your own profile's INTERNAL stores — the
memory, sessions, and cron store under your profile directory — because no
other actor writes those. The exemption is exactly that list. Anything else,
INCLUDING files and directories you yourself created (output dirs, reports,
scripts you maintain), lives in shared space: another actor can legitimately
touch it, so register and claim it like everything else. Your profile memory
is NOT the shared memory store above — don't let "mine" blur that line.

**Protocol** (register once per task, claim before mutating, release at the end):

```bash
SC=~/.hermes/scripts/session_coord.py     # wherever you installed the CLI
CID=$(python3 $SC register --task "<what you are doing>" --surface "bot:<botname>" | head -1)
python3 $SC claim --id $CID --res "<key>" [--res "<key2>"] --task "<task>" --wait --timeout 300
# ... do the whole task ...
python3 $SC done --id $CID
```

Rules:
- Claim EVERYTHING the task will touch up front in ONE call (atomic — this is
  what prevents deadlock). Hold for the whole task; release with `done` at the
  end, never between individual writes.
- Exit 75 = a co-worker holds it. Wait politely (`--wait`), or report the
  holder and its task in your reply and stop. NEVER proceed against a held
  resource, and never `steal` without explicit user approval.
- If your `inbox --id $CID` shows a preempt/priority request: finish the
  current atomic step, save your progress durably to a file, run
  `pause --id $CID --note "<progress file>"`, and say so in your reply.
- Priorities are set by the USER only. Never rank or preempt on your own
  judgment.
- If exit 75 names a `bot:` holder, you may @mention that bot to negotiate
  (ask its ETA, request early release, offer to batch your change into its
  run) — but **chat is never a lock**. Only a successful claim authorizes
  mutation, no matter what was agreed in conversation.
- If the board itself errors, say so and do NOT mutate shared resources blind.

Enrollment marker: session-coord (bot-wire v1) — installer idempotence + board
audit; do not remove this line.
