#!/usr/bin/env python3
"""Classify a Terraform JSON plan as SAFE / REVIEW / BLOCKED.

Usage:
  terraform show -json tfplan > tfplan.json
  python3 plan_guard.py tfplan.json [--allow-replace ADDR ...] [--allow-destroy ADDR ...] [--json]

Exit codes: 0 SAFE, 2 REVIEW (human must look), 3 BLOCKED (do not apply), 1 usage/parse error.

Why: the one-line "Plan: X to add, Y to change, Z to destroy" hides replacements inside
add+destroy. This reads resource_changes[].change.actions directly.
"""
import argparse
import json
import sys

PROTECTED_PREFIXES = (
    "aws_instance", "aws_ebs_volume", "aws_volume_attachment", "aws_kms_key",
    "aws_kms_alias", "aws_s3_bucket", "aws_ecr_repository",
    "aws_cloudwatch_log_group", "aws_iam_role",
)
IAM_PREFIXES = ("aws_iam_",)
NETWORK_PREFIXES = (
    "aws_security_group", "aws_vpc_security_group", "aws_route", "aws_nat_gateway",
    "aws_internet_gateway", "aws_subnet", "aws_vpc", "aws_network_acl", "aws_eip",
    "aws_vpc_endpoint", "aws_lb",
)

SAFE, REVIEW, BLOCKED = "SAFE", "REVIEW", "BLOCKED"
EXIT = {SAFE: 0, REVIEW: 2, BLOCKED: 3}
RANK = {SAFE: 0, REVIEW: 1, BLOCKED: 2}


def classify_actions(actions):
    a = list(actions)
    if a in (["no-op"], ["read"]):
        return "noop"
    if a == ["create"]:
        return "add"
    if a == ["update"]:
        return "change"
    if a == ["delete"]:
        return "destroy"
    if sorted(a) == ["create", "delete"]:
        return "replace"
    return "unknown"


def _is(rtype, prefixes):
    return any(rtype.startswith(p) for p in prefixes)


def _walk_policies(obj):
    """Yield parsed IAM policy documents found in string attributes."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("policy", "inline_policy", "assume_role_policy") and isinstance(v, str):
                try:
                    yield json.loads(v)
                except ValueError:
                    pass
            else:
                yield from _walk_policies(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_policies(v)


def wildcard_findings(after):
    """Return human-readable findings for Allow statements with Resource "*" or wildcard actions."""
    out = []
    for doc in _walk_policies(after or {}):
        stmts = doc.get("Statement", [])
        if isinstance(stmts, dict):
            stmts = [stmts]
        for s in stmts:
            if s.get("Effect") != "Allow":
                continue
            res = s.get("Resource", [])
            res = [res] if isinstance(res, str) else res
            act = s.get("Action", [])
            act = [act] if isinstance(act, str) else act
            if "*" in res:
                out.append(f'Allow with Resource "*" for actions {act}')
            for x in act:
                if x == "*" or x.endswith(":*"):
                    out.append(f"wildcard action {x}")
            princ = s.get("Principal")
            if princ == "*" or (isinstance(princ, dict) and "*" in json.dumps(princ)):
                out.append("wildcard Principal")
    return out


def open_ingress_findings(rtype, after):
    out = []
    if not after:
        return out
    blobs = []
    if rtype == "aws_security_group":
        blobs = after.get("ingress") or []
    elif rtype in ("aws_security_group_rule",):
        if after.get("type") == "ingress":
            blobs = [after]
    elif rtype == "aws_vpc_security_group_ingress_rule":
        blobs = [{"cidr_blocks": [after.get("cidr_ipv4")], "ipv6_cidr_blocks": [after.get("cidr_ipv6")],
                  "from_port": after.get("from_port"), "to_port": after.get("to_port")}]
    for r in blobs:
        cidrs = (r.get("cidr_blocks") or []) + (r.get("ipv6_cidr_blocks") or [])
        if "0.0.0.0/0" in cidrs or "::/0" in cidrs:
            out.append(f"public ingress ports {r.get('from_port')}-{r.get('to_port')}")
    if rtype == "aws_instance" and after.get("associate_public_ip_address"):
        out.append("runtime instance gets public IP")
    return out


def evaluate(plan, allow_replace=(), allow_destroy=()):
    counts = {"add": 0, "change": 0, "destroy": 0, "replace": 0}
    findings = []
    verdict = SAFE

    def bump(v, addr, msg):
        nonlocal verdict
        findings.append({"verdict": v, "address": addr, "reason": msg})
        if RANK[v] > RANK[verdict]:
            verdict = v

    for rc in plan.get("resource_changes", []) or []:
        if rc.get("mode") == "data":
            continue
        addr, rtype = rc.get("address", "?"), rc.get("type", "")
        ch = rc.get("change", {})
        kind = classify_actions(ch.get("actions", []))
        if kind == "noop":
            continue
        if kind == "unknown":
            bump(REVIEW, addr, f"unrecognised actions {ch.get('actions')}")
            continue
        counts[kind] += 1
        protected = _is(rtype, PROTECTED_PREFIXES)

        if kind == "replace":
            why = ch.get("replace_paths")
            detail = f" (forced by {why})" if why else ""
            if addr in allow_replace:
                bump(REVIEW, addr, f"approved replacement{detail}")
            else:
                bump(BLOCKED if protected else REVIEW, addr,
                     f"{'PROTECTED ' if protected else ''}replacement not approved{detail}")
        elif kind == "destroy":
            if addr in allow_destroy:
                bump(REVIEW, addr, "approved destroy")
            else:
                bump(BLOCKED if protected else REVIEW, addr,
                     f"{'PROTECTED ' if protected else ''}destroy not approved")
        elif kind == "change" and protected:
            bump(REVIEW, addr, "in-place change to protected resource")

        after = ch.get("after")
        if _is(rtype, IAM_PREFIXES) and kind in ("add", "change", "replace"):
            wf = wildcard_findings(after)
            for w in wf:
                bump(BLOCKED, addr, f"IAM widening: {w}")
            if not wf:
                bump(REVIEW, addr, "IAM change")
        if _is(rtype, NETWORK_PREFIXES) or rtype == "aws_instance":
            for f in open_ingress_findings(rtype, after):
                bump(BLOCKED, addr, f"network exposure: {f}")
            if _is(rtype, NETWORK_PREFIXES) and kind != "noop":
                bump(REVIEW, addr, "networking change")

    return {"verdict": verdict, "counts": counts, "findings": findings}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan_json")
    p.add_argument("--allow-replace", nargs="*", default=[])
    p.add_argument("--allow-destroy", nargs="*", default=[])
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        with open(a.plan_json) as f:
            plan = json.load(f)
    except (OSError, ValueError) as e:
        print(f"plan_guard: cannot read plan: {e}", file=sys.stderr)
        return 1
    r = evaluate(plan, a.allow_replace, a.allow_destroy)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        c = r["counts"]
        print(f"PLAN: {c['add']} add / {c['change']} change / {c['destroy']} destroy / {c['replace']} replace")
        for f in r["findings"]:
            print(f"  [{f['verdict']}] {f['address']}: {f['reason']}")
        print(f"VERDICT: {r['verdict']}")
    return EXIT[r["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
