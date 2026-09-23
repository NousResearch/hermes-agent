#!/usr/bin/env python3
"""NOVA deployment journal and state machine — the agent's durable memory.

State file: $NOVA_STATE_FILE or ./.nova/state.json (commit-safe: contains no secrets).

  deploy_state.py init --env test
  deploy_state.py show
  deploy_state.py advance PREFLIGHT --evidence "17/17 checks pass"
  deploy_state.py fail --reason "Bedrock category D" --evidence "req-id ..."
  deploy_state.py note "lesson: preflight must invoke Converse"
  deploy_state.py record image_digest sha256:...      # facts for rollback
  deploy_state.py decision "approve replace aws_instance.runtime" --by human
  deploy_state.py next                                 # what state is legal next

Refuses illegal transitions so the agent can't skip PLAN_REVIEW or VERIFY.
"""
import argparse
import datetime as dt
import json
import os
import sys

HAPPY = [
    "DISCOVER", "PREFLIGHT", "PLAN", "PLAN_REVIEW", "INFRASTRUCTURE_APPLY", "IMAGE_DEPLOY",
    "BUNDLE_DEPLOY", "RUNTIME_BOOT", "PROFILE_APPLY", "MODEL_PREFLIGHT", "WORKER_PREFLIGHT",
    "END_TO_END_TEST", "VERIFY", "READY",
]
# Which failure state each working state falls into.
FAILURE_OF = {
    "DISCOVER": "FAILED_PREFLIGHT", "PREFLIGHT": "FAILED_PREFLIGHT",
    "PLAN": "FAILED_PLAN", "PLAN_REVIEW": "FAILED_PLAN",
    "INFRASTRUCTURE_APPLY": "FAILED_INFRASTRUCTURE", "RUNTIME_BOOT": "FAILED_INFRASTRUCTURE",
    "IMAGE_DEPLOY": "FAILED_IMAGE",
    "BUNDLE_DEPLOY": "FAILED_BUNDLE", "PROFILE_APPLY": "FAILED_BUNDLE",
    "MODEL_PREFLIGHT": "FAILED_MODEL",
    "WORKER_PREFLIGHT": "FAILED_WORKER", "END_TO_END_TEST": "FAILED_WORKER",
    "VERIFY": "FAILED_VERIFY",
}
FAILED = sorted(set(FAILURE_OF.values()))
# Safe retry: from a failure you may re-enter the state that produced the failing step's inputs.
RETRY_FROM = {
    "FAILED_PREFLIGHT": ["DISCOVER", "PREFLIGHT"],
    "FAILED_PLAN": ["PLAN"],
    "FAILED_INFRASTRUCTURE": ["PLAN"],          # always re-plan + re-review after infra failure
    "FAILED_IMAGE": ["IMAGE_DEPLOY"],
    "FAILED_BUNDLE": ["BUNDLE_DEPLOY"],
    "FAILED_MODEL": ["PREFLIGHT", "MODEL_PREFLIGHT"],
    "FAILED_WORKER": ["PROFILE_APPLY", "WORKER_PREFLIGHT"],
    "FAILED_VERIFY": ["IMAGE_DEPLOY", "BUNDLE_DEPLOY", "VERIFY"],
}


def allowed_next(current):
    if current is None:
        return ["DISCOVER"]
    if current in FAILED:
        return list(RETRY_FROM[current])
    if current == "READY":
        # A new change on a READY deployment starts again from discovery/preflight/plan.
        return ["DISCOVER", "PREFLIGHT", "PLAN", "IMAGE_DEPLOY", "BUNDLE_DEPLOY"]
    i = HAPPY.index(current)
    nxt = [HAPPY[i + 1]]
    if current == "PLAN_REVIEW":
        nxt.append("PLAN")  # reviewer asked for a new plan
    return nxt


def path():
    return os.environ.get("NOVA_STATE_FILE", os.path.join(".nova", "state.json"))


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load():
    p = path()
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save(s):
    p = path()
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, p)


def log(s, kind, **kw):
    s["journal"].append({"at": now(), "kind": kind, **kw})


class TransitionError(Exception):
    pass


def transition(s, target, evidence=None):
    cur = s.get("state")
    if target not in allowed_next(cur):
        raise TransitionError(f"illegal transition {cur} -> {target}; allowed: {allowed_next(cur)}")
    if target == "INFRASTRUCTURE_APPLY" and not s["facts"].get("reviewed_plan_sha256"):
        raise TransitionError("INFRASTRUCTURE_APPLY requires facts.reviewed_plan_sha256 (record it at PLAN_REVIEW)")
    if target == "READY" and not s["facts"].get("rollback"):
        raise TransitionError("READY requires facts.rollback (previous digests/bundle/plan)")
    if target == "PLAN":
        s["facts"].pop("reviewed_plan_sha256", None)  # a new plan invalidates the old review
    s["state"] = target
    s["last_failure"] = None if target not in FAILED else s.get("last_failure")
    log(s, "advance", to=target, frm=cur, evidence=evidence)
    return s


def fail(s, reason, evidence=None):
    cur = s.get("state")
    if cur is None or cur in FAILED or cur == "READY":
        raise TransitionError(f"cannot fail from {cur}")
    target = FAILURE_OF[cur]
    s["state"] = target
    s["last_failure"] = {"at": now(), "in": cur, "reason": reason, "evidence": evidence}
    log(s, "fail", frm=cur, to=target, reason=reason, evidence=evidence)
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("--env", required=True); i.add_argument("--force", action="store_true")
    sub.add_parser("show"); sub.add_parser("next")
    a = sub.add_parser("advance"); a.add_argument("state"); a.add_argument("--evidence")
    f = sub.add_parser("fail"); f.add_argument("--reason", required=True); f.add_argument("--evidence")
    n = sub.add_parser("note"); n.add_argument("text")
    r = sub.add_parser("record"); r.add_argument("key"); r.add_argument("value")
    d = sub.add_parser("decision"); d.add_argument("text"); d.add_argument("--by", required=True)
    args = p.parse_args(argv)

    s = load()
    if args.cmd == "init":
        if s and not args.force:
            print(f"state exists at {path()} (use --force to reset)", file=sys.stderr); return 1
        s = {"env": args.env, "created": now(), "state": None, "last_failure": None,
             "facts": {}, "decisions": [], "journal": []}
        log(s, "init", env=args.env); save(s); print(f"initialised {path()} for env={args.env}"); return 0
    if s is None:
        print(f"no state at {path()}; run init --env <name>", file=sys.stderr); return 1
    try:
        if args.cmd == "show":
            view = {k: s[k] for k in ("env", "state", "last_failure", "facts", "decisions")}
            view["allowed_next"] = allowed_next(s["state"])
            view["recent"] = s["journal"][-8:]
            print(json.dumps(view, indent=2)); return 0
        if args.cmd == "next":
            print(" ".join(allowed_next(s["state"]))); return 0
        if args.cmd == "advance":
            transition(s, args.state.upper(), args.evidence)
        elif args.cmd == "fail":
            fail(s, args.reason, args.evidence)
        elif args.cmd == "note":
            log(s, "note", text=args.text)
        elif args.cmd == "record":
            try:
                val = json.loads(args.value)
            except ValueError:
                val = args.value
            s["facts"][args.key] = val; log(s, "record", key=args.key)
        elif args.cmd == "decision":
            s["decisions"].append({"at": now(), "by": args.by, "text": args.text})
            log(s, "decision", by=args.by, text=args.text)
    except TransitionError as e:
        print(f"REFUSED: {e}", file=sys.stderr); return 3
    save(s); print(f"ok: state={s['state']}"); return 0


if __name__ == "__main__":
    sys.exit(main())
