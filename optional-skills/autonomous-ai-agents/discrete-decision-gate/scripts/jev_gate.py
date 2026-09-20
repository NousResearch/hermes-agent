#!/usr/bin/env python3
"""Jev gate: typed single-label decisions for the code/repo workflow protocol.

Same protocol as the sketch this replaces, with the call path and the parsing fixed:

  Gate 1  triage_intent()                 rules first, Jev only for the ambiguous middle band
  Map     evaluate_blast_radius()         fail DIRECTION: high (closed)
  Gate 3  verify_diff_matches_intent()    noul answer read from the right field
  Gate 2  evaluate_harness_output()       fail DIRECTION: abort (closed), bounded retries

Backends, resolved in this order:
  1. TypeSafe Jev through OpenRouter: POST https://openrouter.ai/api/alpha/decisions with a flat
     body {model, state, questions}, model typesafe/jev-1.13 (OPENROUTER_API_KEY). This is the
     default on this host.
  2. TypeSafe 1P: POST https://api.typesafe.ai/v1/systemone (TYPESAFE_API_KEY). Opt-in only -
     `--backend typesafe` or JEV_BACKEND=typesafe - and not used locally.

  There is no typesafe/jev-latest id on OpenRouter; `typesafe/jev-1.13` is the only Jev id there
  and every response names the snapshot it served (e.g. typesafe/jev-1.13-20260917). The words
  `latest` / `jev-latest` / `typesafe/jev-latest` / `typesafe/jev` normalize to that pin.
  The OpenRouter model catalog does NOT list typesafe/*, so never preflight by looking the model
  up in the catalog - make a live call (`doctor` does exactly that).

Design rules carried over from the sketch and enforced here:
  * A gate NEVER fabricates a label. Every failure returns status != "ok" with an error_kind
    ("no_key" | "transport" | "http" | "schema") so schema drift is distinguishable from an
    outage instead of both looking like a permanent fail-closed verdict.
  * Every result carries provenance: backend, endpoint, model, latency, cost.
  * Keys are read from the process env, then from ~/.hermes/.env (a bare shell does not export
    them - measured on this host).
  * The label vocabulary stays canonical (bash_direct|direct_dsh|crg_first, low|high,
    match|mismatch, complete|retry|abort). Exceptional causes live in "reason", not as new labels.

CLI
  jev_gate.py triage   --intent "refactor auth across 14 modules" [--json]
  jev_gate.py blast    --summary "<code-review-graph impact radius output>" [--json]
  jev_gate.py diff     --intent "..." [--diff-file -|<path>]   # stdin when omitted
  jev_gate.py harness  --exit-code 1 --tail "<output tail>" [--attempt 1] [--max-attempts 3]
                       [--expect-file <path> ...] [--json]
  common: --timeout <s> (default 8), --model <id>

Exit codes: 0 ok verdict, 3 gate unavailable / fail-closed default applied.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
TYPESAFE_MODEL = "jev-latest"
# OpenRouter has NO typesafe/jev-latest alias (measured: "Model typesafe/jev-latest does not
# exist"); typesafe/jev-1.13 is the only Jev id there and it serves the newest snapshot -
# every response names it (e.g. typesafe/jev-1.13-20260917). The aliases below map the
# habitual "latest" spelling onto that id instead of failing.
OPENROUTER_MODEL = "typesafe/jev-1.13"
LATEST_ALIASES = ("latest", "jev-latest", "typesafe/jev-latest", "typesafe/jev")

CONFIDENCE_FLOOR = 0.50
DIFF_MATCH_FLOOR = 0.70
DEFAULT_TIMEOUT = 8.0

# Fail directions: chosen from the stakes of the decision, never from the label list.
FAIL_DIRECTION = {
    "triage": "bash_direct",   # low stakes: fail open, cheapest path
    "blast": "high",           # high stakes: fail closed, ask the operator
    "diff": "mismatch",        # high stakes: fail closed, do not trust the patch
    "harness": "abort",        # high stakes: never auto-declare success
}

# Narrow, harness-level credential/environment failures. Deliberately NOT "permission denied"
# or "no module named" on their own: a test asserting a PermissionError prints those too.
ENV_FAILURE_PATTERNS = (
    "invalid_api_key", "invalid api key", "authentication_error", "no api key",
    "missing_credential", "could not read username for", "permission denied (publickey)",
    "insufficient_quota", "quota exceeded", "unexpected eof while looking for matching",
)
# Masked-failure detection, case-insensitive: pytest prints "1 failed", not "FAIL".
FAILURE_PATTERNS = (
    r"\b\d+\s+failed\b", r"\b\d+\s+errors?\b", r"\bfailed\b", r"traceback \(most recent call last\)",
    r"\bassertionerror\b", r"^\s*\[fail\]", r"\bnpm err!", r"\berror:", r"\bfatal:",
)


# --------------------------------------------------------------------------- plumbing

def load_env_file(path: str | None = None) -> list[str]:
    """Fill missing keys from ~/.hermes/.env (existing environment wins)."""
    loaded: list[str] = []
    p = Path(path) if path else Path.home() / ".hermes" / ".env"
    if not p.is_file():
        return loaded
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _resolution(prefer: str | None = None) -> tuple[str, str, str, str] | None:
    """Return (backend, url, model, api_key).

    OpenRouter's Decisions route is the DEFAULT: this host runs OpenRouter Jev only, and the
    TypeSafe 1P path is opt-in (`--backend typesafe` / `JEV_BACKEND=typesafe`) so a stray
    TYPESAFE_API_KEY cannot take over the gate. With no explicit pin the order is
    openrouter -> typesafe (the 1P key is only reached when the OpenRouter key is absent).
    """
    prefer = (prefer or os.getenv("JEV_BACKEND") or "auto").strip().lower()
    if prefer in ("openrouter", "typesafe"):
        order = (prefer,)
    else:
        order = ("openrouter", "typesafe")
    for name in order:
        if name == "typesafe" and os.getenv("TYPESAFE_API_KEY"):
            return ("typesafe", TYPESAFE_URL, os.getenv("JEV_MODEL", TYPESAFE_MODEL),
                    os.environ["TYPESAFE_API_KEY"])
        if name == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            return ("openrouter", OPENROUTER_DECISIONS_URL, os.getenv("JEV_MODEL", OPENROUTER_MODEL),
                    os.environ["OPENROUTER_API_KEY"])
    return None


def _post(url: str, payload: dict, key: str, timeout: float) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"error": {"message": e.reason}}


def query_jev(state, questions: dict, *, timeout: float = DEFAULT_TIMEOUT,
              model: str | None = None, backend: str | None = None,
              max_retries: int = 1) -> dict:
    """One decisions request. Returns {status, answers?, backend, endpoint, model, latency_ms,
    cost, reason?, error_kind?}. status is "ok" only when a well-formed answers map came back."""
    resolved = _resolution(backend)
    if resolved is None:
        pinned = (backend or os.getenv("JEV_BACKEND") or "").strip().lower()
        reason = (f"{pinned.upper()}_API_KEY not set (backend pinned to {pinned})"
                  if pinned in ("openrouter", "typesafe") else
                  "no OPENROUTER_API_KEY and no TYPESAFE_API_KEY (checked env and ~/.hermes/.env)")
        return {"status": "unavailable", "error_kind": "no_key", "reason": reason}
    backend, url, default_model, key = resolved
    chosen = model or default_model
    if chosen.strip().lower() in LATEST_ALIASES:
        chosen = default_model  # "latest" on OpenRouter == the pinned id + newest snapshot
    payload = {"model": chosen, "state": state, "questions": questions}

    last: dict = {}
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        try:
            code, data = _post(url, payload, key, timeout)
        except Exception as exc:  # socket timeouts, DNS, TLS, connection resets
            last = {"status": "unavailable", "error_kind": "transport",
                    "reason": f"{type(exc).__name__}: {exc}"}
        else:
            latency = round((time.monotonic() - started) * 1000)
            if code != 200:
                message = str(data.get("error", {}).get("message", data))[:300]
                last = {"status": "unavailable", "error_kind": "http",
                        "reason": f"HTTP {code}: {message}"}
            elif not isinstance(data.get("answers"), dict) or not data["answers"]:
                last = {"status": "unavailable", "error_kind": "schema",
                        "reason": f"no answers map in response: {json.dumps(data)[:200]}"}
            else:
                return {"status": "ok", "answers": data["answers"], "backend": backend,
                        "endpoint": url, "model": data.get("model", chosen),
                        "latency_ms": latency, "cost": (data.get("usage") or {}).get("cost"),
                        "attempts": attempt + 1}
        if attempt < max_retries and last.get("error_kind") == "transport":
            continue
        break
    return {**last, "backend": backend, "endpoint": url, "model": chosen}


def _answer(res: dict, key: str) -> dict:
    return (res.get("answers") or {}).get(key) or {}


def _result(label: str, fail_direction: str, gate: str, res: dict, reason: str = "",
            **extra) -> dict:
    """Assemble the caller-facing result. Never invents a label for a failed gate."""
    out = {"gate": gate, "label": label, "status": res.get("status", "unavailable"),
           "fail_direction": fail_direction,
           "backend": res.get("backend"), "model": res.get("model"),
           "latency_ms": res.get("latency_ms"), "cost": res.get("cost")}
    if out["status"] != "ok":
        out["reason"] = res.get("reason", "gate unavailable")
        out["error_kind"] = res.get("error_kind")
    if reason:
        out["why"] = reason
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


# ------------------------------------------------------------------- Gate 1: triage

READ_ONLY = re.compile(r"^(git (status|diff|log|show|branch)|ls|pwd|cat |find |rg |grep |show |view |head |tail )")
SCRIPT_ONLY = ("script only", "one-off", "one off", "standalone", "new helper script",
               "single new file", "throwaway")
STRUCTURAL = ("refactor", "rename symbol", "callers", "dependents", "broken test", "failing test",
              "bug", "regression", "wire up", "trace", "across ", "modules", "schema",
              "migration", "why does", "investigate", "since last deploy", "used to work")
# Any-order failure language: "the cart test has been failing", "ci is red", "build broke".
FAILURE_INVESTIGATION = re.compile(
    r"\b(tests?|ci|build|suite|pipeline)\b.{0,40}\b(fail\w*|broke\w*|broken|red|flak\w*)\b"
    r"|\b(fail\w*|broke\w*|broken)\b.{0,40}\b(tests?|ci|build|suite|pipeline)\b")


def triage_intent(intent: str, *, timeout: float = DEFAULT_TIMEOUT, model: str | None = None,
                  backend: str | None = None, json_out: bool = False) -> dict:
    """Rules first; a Jev call only for the ambiguous middle band. Fail open to bash_direct."""
    text = (intent or "").strip()
    low = text.lower()
    if READ_ONLY.match(low):
        return _result("bash_direct", FAIL_DIRECTION["triage"], "triage",
                       {"status": "ok", "backend": "rules"}, "read-only shell inspection")
    if any(k in low for k in SCRIPT_ONLY):
        return _result("direct_dsh", FAIL_DIRECTION["triage"], "triage",
                       {"status": "ok", "backend": "rules"}, "isolated single-file work")
    if any(k in low for k in STRUCTURAL) or FAILURE_INVESTIGATION.search(low):
        return _result("crg_first", FAIL_DIRECTION["triage"], "triage",
                       {"status": "ok", "backend": "rules"}, "structural / multi-module signal")

    res = query_jev(text, {"tier": {
        "type": "choice",
        "instructions": "Determine the execution tier for this repository request",
        "criteria": {
            "bash_direct": "Inspection, reading, status or log checks; nothing is mutated.",
            "direct_dsh": "Isolated single-file creation or edit with no project-wide dependencies.",
            "crg_first": "Refactor, dependency tracing, or bug fix spanning multiple modules.",
        }}}, timeout=timeout, model=model, backend=backend)
    if res["status"] != "ok":
        return _result(FAIL_DIRECTION["triage"], FAIL_DIRECTION["triage"], "triage", res,
                       "ambiguous middle band + gate unavailable")
    ans = _answer(res, "tier")
    choice = ans.get("choice")
    if choice not in ("bash_direct", "direct_dsh", "crg_first"):
        return _result(FAIL_DIRECTION["triage"], FAIL_DIRECTION["triage"], "triage",
                       {**res, "status": "unavailable", "error_kind": "schema",
                        "reason": f"off-schema tier: {choice!r}"})
    return _result(choice, FAIL_DIRECTION["triage"], "triage", res,
                   f"Jev choice (confidence {ans.get('confidence')})",
                   confidence=ans.get("confidence"), probabilities=ans.get("probabilities"))


# ------------------------------------------------------- Map: blast radius (fail closed)

def evaluate_blast_radius(crg_summary: str, *, timeout: float = DEFAULT_TIMEOUT,
                          model: str | None = None, backend: str | None = None) -> dict:
    """Jev verdict on a code-review-graph impact radius. Fail closed to "high"."""
    if not (crg_summary or "").strip():
        return _result("high", FAIL_DIRECTION["blast"], "blast", {
            "status": "unavailable", "error_kind": "empty_input",
            "reason": "no CRG summary supplied - refresh the graph, do not assume low risk"})
    res = query_jev(crg_summary, {"risk": {
        "type": "choice",
        "instructions": "Is the blast radius isolated or high risk?",
        "criteria": {
            "low": "Three or fewer files, leaf modules or tests only, no shared state.",
            "high": "Core config, auth or schema models, shared state, or four or more dependents.",
        }}}, timeout=timeout, model=model, backend=backend)
    if res["status"] != "ok":
        return _result(FAIL_DIRECTION["blast"], FAIL_DIRECTION["blast"], "blast", res)
    ans = _answer(res, "risk")
    choice = ans.get("choice")
    confidence = ans.get("confidence")
    if choice not in ("low", "high"):
        return _result("high", FAIL_DIRECTION["blast"], "blast",
                       {**res, "status": "unavailable", "error_kind": "schema",
                        "reason": f"off-schema risk: {choice!r}"})
    if confidence is None:
        # A missing confidence on a choice answer is schema drift, not a low score.
        return _result("high", FAIL_DIRECTION["blast"], "blast",
                       {**res, "status": "unavailable", "error_kind": "schema",
                        "reason": f"choice answer without confidence: {json.dumps(ans)[:200]}"})
    if confidence < CONFIDENCE_FLOOR:
        return _result("high", FAIL_DIRECTION["blast"], "blast", res,
                       f"confidence {confidence} below floor {CONFIDENCE_FLOOR} - treated as high",
                       confidence=confidence, low_confidence=True,
                       probabilities=ans.get("probabilities"))
    return _result(choice, FAIL_DIRECTION["blast"], "blast", res,
                   f"Jev confidence {confidence}", confidence=confidence,
                   probabilities=ans.get("probabilities"))


# ------------------------------------------- Gate 3: does the diff match the intent (noul)

def verify_diff_matches_intent(intent: str, git_diff: str, *, timeout: float = DEFAULT_TIMEOUT,
                               model: str | None = None, backend: str | None = None) -> dict:
    """noul question: is this patch the patch that was asked for? Fail closed to "mismatch"."""
    diff = git_diff or ""
    if not diff.strip():
        return _result("mismatch", FAIL_DIRECTION["diff"], "diff", {
            "status": "unavailable", "error_kind": "empty_input",
            "reason": "empty diff"})
    files = re.findall(r"^diff --git a/(\S+)", diff, flags=re.MULTILINE)
    state = {
        "task_intent": intent,
        "files_touched": files[:50],
        "files_touched_count": len(files),
        "patch_excerpt": diff[-4000:],
    }
    res = query_jev(state, {"matches": {
        "type": "noul",
        "instructions": "Does the patch accurately reflect the user's intent, without unrelated "
                        "or destructive edits?",
        "criteria": {
            "true": "The patch fulfills the stated intent and touches only related files.",
            "false": "The patch is unrelated, destructive, or fails to address the intent.",
        }}}, timeout=timeout, model=model, backend=backend)
    if res["status"] != "ok":
        return _result(FAIL_DIRECTION["diff"], FAIL_DIRECTION["diff"], "diff", res)
    ans = _answer(res, "matches")
    # The noul answer is {"type": "noul", "noul": <prob>} - there is no probabilities.true.
    prob = ans.get("noul")
    if not isinstance(prob, (int, float)):
        return _result("mismatch", FAIL_DIRECTION["diff"], "diff",
                       {**res, "status": "unavailable", "error_kind": "schema",
                        "reason": f"noul answer malformed: {json.dumps(ans)[:200]}"})
    label = "match" if prob >= DIFF_MATCH_FLOOR else "mismatch"
    why = (f"noul probability {prob} vs floor {DIFF_MATCH_FLOOR}")
    if label == "mismatch":
        why += " - do not run tests on this patch; send it back with the intent"
    return _result(label, FAIL_DIRECTION["diff"], "diff", res, why, probability=prob,
                   files_touched=files[:50],
                   low_confidence=(prob < DIFF_MATCH_FLOOR))


# ------------------------------------------------ Gate 2: harness verdict (fail closed)

def _looks_failed(tail: str) -> str | None:
    for pattern in FAILURE_PATTERNS:
        m = re.search(pattern, tail, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(0).strip()
    return None


def evaluate_harness_output(exit_code: int, stdout_tail: str, attempt_count: int = 1,
                            max_attempts: int = 3, expect_files: list[str] | None = None,
                            *, timeout: float = DEFAULT_TIMEOUT, model: str | None = None,
                            backend: str | None = None, cwd: str | None = None) -> dict:
    """Verdict for one harness attempt. Fail closed to "abort"; retries are bounded here."""
    tail = stdout_tail or ""
    low = tail.lower()

    hit = next((p for p in ENV_FAILURE_PATTERNS if p in low), None)
    if hit:
        return _result("abort", FAIL_DIRECTION["harness"], "harness", {
            "status": "unavailable", "error_kind": "environment",
            "reason": f"environment/credential failure in output: {hit!r}"})

    if attempt_count >= max_attempts:
        return _result("abort", FAIL_DIRECTION["harness"], "harness", {
            "status": "unavailable", "error_kind": "max_attempts",
            "reason": f"attempt {attempt_count} of {max_attempts} - retries exhausted"})

    missing = [f for f in (expect_files or [])
               if not (Path(cwd or ".") / f).is_file() or (Path(cwd or ".") / f).stat().st_size == 0]
    clean = exit_code == 0 and not _looks_failed(tail)

    if clean and missing:
        return _result("retry", FAIL_DIRECTION["harness"], "harness", {
            "status": "unavailable", "error_kind": "artifact_missing",
            "reason": f"exit 0 but expected artifact(s) absent or empty: {missing}"})
    if missing:
        return _result("retry", FAIL_DIRECTION["harness"], "harness", {
            "status": "unavailable", "error_kind": "artifact_missing",
            "reason": f"artifact(s) absent or empty: {missing}"})

    questions = {"verdict": clean and {
        "type": "choice",
        "instructions": "Exit code is 0 and no failure text matched. Did the run genuinely "
                        "succeed, or are failures masked?",
        "criteria": {
            "complete": "Exit code 0 and the checks the task required actually ran and passed.",
            "retry": "Exit 0 but the required checks were skipped, partially run, or masked.",
            "abort": "The run is incoherent or unrecoverable.",
        }} or {
        "type": "choice",
        "instructions": "Evaluate this failed harness run.",
        "criteria": {
            "complete": "The failure text is unrelated to the task and the work is done.",
            "retry": "An actionable test, type, or syntax failure the harness can fix next pass.",
            "abort": "Workspace corruption, repeating crash, or an unrecoverable error.",
        }}}
    res = query_jev({"exit_code": exit_code, "output_tail": tail[-2000:]}, questions,
                    timeout=timeout, model=model, backend=backend)
    if res["status"] != "ok":
        return _result(FAIL_DIRECTION["harness"], FAIL_DIRECTION["harness"], "harness", res)
    ans = _answer(res, "verdict")
    choice = ans.get("choice")
    confidence = ans.get("confidence")
    if choice not in ("complete", "retry", "abort"):
        return _result("abort", FAIL_DIRECTION["harness"], "harness",
                       {**res, "status": "unavailable", "error_kind": "schema",
                        "reason": f"off-schema verdict: {choice!r}"})
    if confidence is None:
        return _result("abort", FAIL_DIRECTION["harness"], "harness",
                       {**res, "status": "unavailable", "error_kind": "schema",
                        "reason": f"choice answer without confidence: {json.dumps(ans)[:200]}"})
    if confidence < CONFIDENCE_FLOOR:
        return _result("abort", FAIL_DIRECTION["harness"], "harness", res,
                       f"confidence {confidence} below floor {CONFIDENCE_FLOOR}",
                       confidence=confidence, low_confidence=True,
                       probabilities=ans.get("probabilities"))
    if choice == "complete" and exit_code != 0:
        return _result("abort", FAIL_DIRECTION["harness"], "harness", res,
                       "model said complete on a non-zero exit code - refused",
                       confidence=confidence)
    return _result(choice, FAIL_DIRECTION["harness"], "harness", res,
                   f"Jev confidence {confidence}", confidence=confidence,
                   probabilities=ans.get("probabilities"))


# ------------------------------------------------------------------------- doctor

def doctor(model: str | None = None, timeout: float = DEFAULT_TIMEOUT,
           backend: str | None = None) -> dict:
    """Preflight: which backend answers, which Jev snapshot is served, cost and latency.

    OpenRouter has no `-latest` alias, so "latest" is read two ways: the pinned model id, and the
    provider snapshot the endpoints API currently advertises for it.
    """
    resolved = _resolution(backend)
    report: dict = {"keys": {"TYPESAFE_API_KEY": bool(os.getenv("TYPESAFE_API_KEY")),
                             "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY"))}}
    if resolved is None:
        report["status"] = "no_keys"
        return report
    backend, url, default_model, _ = resolved
    chosen = model if model and model.strip().lower() not in LATEST_ALIASES else default_model
    report.update({"backend": backend, "endpoint": url, "requested_model": chosen})

    if backend == "openrouter":
        req = urllib.request.Request(f"{OPENROUTER_MODELS_URL}/{chosen}/endpoints")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode()).get("data", {})
            endpoints = data.get("endpoints") or []
            first = endpoints[0] if endpoints else {}
            report["advertised"] = {
                "model_id": data.get("id"),
                "name": data.get("name"),
                "created": data.get("created"),
                "provider_endpoint": first.get("name"),
                "prompt_price_per_token": (first.get("pricing") or {}).get("prompt"),
                "context_length": first.get("context_length"),
                "uptime_last_1d": first.get("uptime_last_1d"),
                "modality": (data.get("architecture") or {}).get("modality"),
            }
        except Exception as exc:
            report["advertised"] = {"error": f"{type(exc).__name__}: {exc}"}

    probe = query_jev("Probe: the build finished and every check passed.",
                      {"ok": {"type": "choice", "instructions": "Did the checks pass?",
                              "criteria": {"yes": "Checks passed.", "no": "Checks failed."}}},
                      timeout=timeout, model=chosen)
    report["probe"] = {"status": probe["status"], "served_model": probe.get("model"),
                       "choice": (probe.get("answers", {}).get("ok") or {}).get("choice"),
                       "latency_ms": probe.get("latency_ms"), "cost": probe.get("cost")}
    if probe["status"] != "ok":
        report["probe"]["reason"] = probe.get("reason")
        report["probe"]["error_kind"] = probe.get("error_kind")
    report["status"] = probe["status"]
    return report


# --------------------------------------------------------------------------------- CLI

def _emit(res: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(res, indent=2))
    else:
        line = res["label"]
        if res["status"] != "ok":
            line = (f"{res['label']}  (gate unavailable: {res.get('error_kind')} - "
                    f"{res.get('reason')})")
        elif res.get("confidence") is not None:
            line = f"{res['label']}  (confidence {res['confidence']})"
        print(line)
        if res.get("why"):
            print(f"  why: {res['why']}")
        if res.get("backend"):
            print(f"  via {res['backend']} {res.get('model') or ''} "
                  f"{res.get('latency_ms') or '-'}ms cost={res.get('cost')}")
    return 0 if res["status"] == "ok" else 3


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage")
    t.add_argument("--intent", required=True)

    b = sub.add_parser("blast")
    b.add_argument("--summary", required=True)

    d = sub.add_parser("diff")
    d.add_argument("--intent", required=True)
    d.add_argument("--diff-file", default=None, help="'-' or a path; stdin when omitted")

    h = sub.add_parser("harness")
    h.add_argument("--exit-code", type=int, required=True)
    h.add_argument("--tail", default="")
    h.add_argument("--attempt", type=int, default=1)
    h.add_argument("--max-attempts", type=int, default=3)
    h.add_argument("--expect-file", action="append", default=[])

    dr = sub.add_parser("doctor", help="preflight: backend, served Jev snapshot, cost, latency")

    for sp in (t, b, d, h, dr):
        sp.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
        sp.add_argument("--model", default=None)
        sp.add_argument("--backend", default="auto", choices=["auto", "openrouter", "typesafe"],
                        help="auto (default) = OpenRouter Jev, falling back to the 1P key only if "
                             "OPENROUTER_API_KEY is absent; JEV_BACKEND env is honoured when "
                             "this is left as 'auto'")
        sp.add_argument("--json", action="store_true")
        sp.add_argument("--env-file", default=None, help="default: ~/.hermes/.env")

    a = p.parse_args(argv)
    load_env_file(a.env_file)
    prefer = None if a.backend == "auto" else a.backend

    if a.cmd == "doctor":
        rep = doctor(model=a.model, timeout=a.timeout, backend=prefer)
        print(json.dumps(rep, indent=2))
        return 0 if rep.get("status") == "ok" else 3

    if a.cmd == "triage":
        res = triage_intent(a.intent, timeout=a.timeout, model=a.model, backend=prefer)
    elif a.cmd == "blast":
        res = evaluate_blast_radius(a.summary, timeout=a.timeout, model=a.model, backend=prefer)
    elif a.cmd == "diff":
        if a.diff_file in (None, "-"):
            git_diff = sys.stdin.read() if not sys.stdin.isatty() else ""
        else:
            git_diff = Path(a.diff_file).read_text(encoding="utf-8", errors="replace")
        res = verify_diff_matches_intent(a.intent, git_diff, timeout=a.timeout, model=a.model,
                                         backend=prefer)
    else:
        res = evaluate_harness_output(a.exit_code, a.tail, a.attempt, a.max_attempts,
                                      a.expect_file, timeout=a.timeout, model=a.model,
                                      backend=prefer)
    return _emit(res, a.json)


if __name__ == "__main__":
    sys.exit(main())
