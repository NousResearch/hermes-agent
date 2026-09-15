#!/usr/bin/env python3
"""Generate local Hermes profiles from the #vibe-team presets.

Source of truth: Notion Team page 3b4ac25e-b49f-804d-8f6c-d302c7913701,
materialised as JSON in AgentPod/.agents/skills/deploy-slack-agent-team/presets/.

Local profiles are kanban workers, NOT Slack bots: all Slack/LiteLLM/Bitwarden
fields are deliberately dropped. What we keep is the part that makes an agent a
teammate: persona, scope, explicit not-my-scope handoffs, depth boundary,
response discipline and the anti-fabrication guardrail.
"""
import json
import pathlib
import subprocess
import sys

PRESETS = pathlib.Path(
    "/Users/engineer/workspace/AgentPod/.agents/skills/deploy-slack-agent-team/presets"
)
PROFILES = pathlib.Path("/Users/engineer/.hermes/profiles")

# preset id -> local peer name used in handoff lines
PEERS = {
    "cto": "cto",
    "software-engineer": "software-engineer",
    "growth-manager": "growth-manager",
    "marketing-manager": "marketing-manager",
    "support-engineer": "support-engineer",
    "corporate-lawyer": "corporate-lawyer",
}
# slack user id -> local profile name, so handoff lines point at real profiles
SLACK_TO_PROFILE = {}


def load_all():
    out = {}
    for pid in PEERS:
        out[pid] = json.loads((PRESETS / f"{pid}.json").read_text())
    for pid, d in out.items():
        uid = d.get("slack", {}).get("slack_user_id")
        if uid:
            SLACK_TO_PROFILE[uid] = pid
    return out


def deslack(line: str) -> str:
    """Rewrite `Name <@Uxxx>` handoffs into local profile names."""
    for uid, prof in SLACK_TO_PROFILE.items():
        line = line.replace(f"<@{uid}>", f"(profile `{prof}`)")
    # the human owner
    line = line.replace("<@U0ABQ4MC1NC>", "(the human owner, Den)")
    return line


LIVE_PROOF_PROTOCOL = """
## P-LIVE — Prove it on the live system BEFORE you request review

Green code and passing unit tests are not evidence that the change reached
production. Two consecutive reviewer rejections on this board were produced by
`kubectl` probes of the live cluster that the implementer never ran: a freshly
provisioned tenant was dead (`HTTP=000`) while the guard reported green, because
the NetworkPolicy allowing the tailnet CIDR was missing on real tenants and
present only on the CI canary.

Before calling `kanban_request_review` on any infra/deploy/network change:

1. Exercise the REAL path on the REAL system, not a test fixture. For cluster
   work that means `kubectl exec` into an actual tenant pod on the actual
   affected node and dialing the actual endpoint.
2. Check a RANDOM real tenant, never only the CI canary. The canary frequently
   carries setup that ordinary tenants lack — that difference is exactly the
   bug class that keeps escaping review.
3. Paste the verbatim command and its output into the task comment. No output,
   no review request.
4. State explicitly what you did NOT verify and why. An honest gap costs one
   comment; a gap the reviewer discovers costs a full round trip.

A change that has not been observed working on the live system is not ready for
review, regardless of CI colour.

## P-GUARD — A guard built on a hand-maintained list is not a guard

Rejected on this board: an inter-node endpoint guard that could not fail on
undeclared endpoints. It compared against a list a human had to remember to
update, so a newly added endpoint was silently unguarded and the check stayed
green. That is the same failure class the guard existed to prevent.

Guards must DERIVE what they check from the source of truth — grep the env/config
surface, enumerate the live objects, walk the call sites — never from a literal
list maintained by hand. If a hand-maintained list is genuinely unavoidable, the
guard must additionally fail when it finds an item in the source of truth that is
absent from the list.

Prove every guard by mutation: break the thing it protects, watch it go RED,
revert, watch it go GREEN. Report both observations. A guard never seen failing
has not been tested.
"""

STANDING_DECISIONS = """
## Standing decisions — already made, do NOT block to ask

The engineering manager has pre-decided these. Apply them, write one line in the
task comment saying which rule you applied, and keep going. Blocking a card to
ask one of these questions is a defect.

- Branch diverged from `origin/main` / no merge base → rebase (or recreate the
  branch from `origin/main` and re-apply your commits). NEVER
  `git merge --allow-unrelated-histories`.
- Two guards/checks for the same invariant → unify into ONE generation-aware
  guard; delete the duplicate; keep both directions of the mutation proof.
- A required CI check is red on YOUR PR → it is your job; fix or root-cause it,
  do not ask whether to.
- A test that exists only on `origin/main` fails on your branch → rebase first,
  then fix.
- PR body / design doc disagrees with the code → the code follows the design
  the card states (e.g. default model `litellm/auto` forwarded to `gpt-5.4`);
  update the body.
- Labels that restart customer workloads (`rebootstrap`, `deploy`, etc.) are
  applied by the supervisor, never by you. Do not block on them — finish the
  PR, request review, and note "needs <label>" in the summary.

STILL block and ask (irreversible / financial / customer-visible):
- Deleting or force-pushing over someone else's branch or PR.
- Anything touching billing, payments, Stripe, Telegram Stars, LiteLLM keys.
- Scaling, deleting, or restarting a live tenant outside a CI canary.
- Spending money or creating cloud resources.
"""


def build_soul(d: dict, discipline: str) -> str:
    p = []
    p.append(f"# {d['persona']} — {d['role']}\n")
    p.append(d["persona_prompt"].strip() + "\n")

    p.append("\n## Scope — this is yours\n")
    for s in d["scope"]:
        p.append(f"- {s}")

    p.append("\n## Not your scope — hand off in one line, do not do it yourself\n")
    for s in d["not_my_scope"]:
        p.append(f"- {deslack(s)}")

    db = d.get("depth_boundary", {})
    if db:
        p.append(f"\n{db.get('heading', '## Depth boundary')}\n")
        p.append(deslack(db.get("rule", "")))

    vr = d.get("voice_rules") or []
    if vr:
        p.append("\n## Voice\n")
        for s in vr:
            p.append(f"- {deslack(s)}")

    afg = d.get("anti_fabrication_guardrail", {})
    if afg.get("required"):
        p.append("\n## Anti-fabrication guardrail (hard rule)\n")
        p.append(afg["rule"])

    p.append("\n## Working on the kanban board\n")
    p.append(
        "You are a worker on the shared Hermes kanban board, not a chat bot.\n"
        "- Do the work in your assigned workspace; delegate implementation to `pi --print` "
        "rather than hand-editing when the task is a coding task.\n"
        "- A green exit code and your own summary are NOT evidence. Verify the premise of the "
        "ticket with a live measurement before doing any work — if the premise is false, say so "
        "and close it rather than inventing work.\n"
        "- When your part is done, call `hermes kanban request-review <id> --summary \"...\" "
        "--reviewer reviewer`. Never self-close a task that changes code.\n"
    )

    p.append("\n" + STANDING_DECISIONS.strip() + "\n")
    p.append("\n" + discipline.strip() + "\n")
    p.append("\n" + LIVE_PROOF_PROTOCOL.strip() + "\n")
    return "\n".join(p) + "\n"


def main():
    presets = load_all()
    discipline = (PRESETS / "response-discipline.md").read_text()
    made = []
    for pid, d in presets.items():
        pdir = PROFILES / pid
        if not pdir.exists():
            subprocess.run(
                [
                    "hermes", "profile", "create", pid, "--clone",
                    "--description", d["profile_description"],
                ],
                check=True, capture_output=True, text=True,
            )
        soul = build_soul(d, discipline)
        (PROFILES / pid / "SOUL.md").write_text(soul)
        made.append((pid, d["persona"], d["role"], len(soul)))
    for m in made:
        print(f"{m[0]:20} {m[1]:20} {m[2]:28} SOUL.md {m[3]} chars")


if __name__ == "__main__":
    sys.exit(main())
