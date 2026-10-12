#!/usr/bin/env python3
"""Opt-in synthetic evaluation runner; dry runs make no model calls."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORBIDDEN = ("anthropic", "claude")
SECRET_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret)(\s*[:=]\s*|\s+)[^\s,;]+")
URL_RE = re.compile(r"https?://[^\s)\]>]+")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s?(?:%|°C|mg|mL|min|horas?|días?))?")

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def redact(text: str) -> str:
    return SECRET_RE.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)

def iso_now():
    return datetime.now(UTC).isoformat()

def validate_provider(p):
    joined = " ".join(p["command"]).lower()
    if any(x in joined or x in p["id"].lower() for x in FORBIDDEN):
        raise ValueError(f"forbidden provider in {p['id']}")
    if "{prompt_file}" not in p["command"]:
        raise ValueError(f"missing prompt placeholder in {p['id']}")

def response_metrics(text: str):
    urls = URL_RE.findall(text)
    numbers = NUMBER_RE.findall(text)
    return {"response_chars": len(text), "url_count": len(urls), "urls": urls,
            "clinical_number_mentions_proxy": numbers,
            "has_any_source_for_numbers_proxy": bool(urls) if numbers else None}

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset.jsonl")
    ap.add_argument("--models", type=Path, default=ROOT / "models.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--case", action="append", dest="cases")
    ap.add_argument("--provider", action="append", dest="providers")
    ap.add_argument("--timeout", type=int, default=210)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Execute explicitly enabled providers in isolated scratch homes.")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.execute:
        ap.error("choose --dry-run or explicitly opt in with --execute")
    if args.dry_run and args.execute:
        ap.error("--dry-run and --execute are mutually exclusive")
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    cases = load_jsonl(args.dataset)
    cfg = json.loads(args.models.read_text(encoding="utf-8"))
    providers = [p for p in cfg["providers"] if args.dry_run or p.get("enabled") is True]
    if args.cases: cases = [c for c in cases if c["id"] in args.cases]
    if args.providers: providers = [p for p in providers if p["id"] in args.providers]
    if not cases or not providers: raise SystemExit("no selected cases/providers; providers are disabled by default")
    if len({p["id"] for p in providers}) != len(providers):
        raise SystemExit("duplicate provider id")
    for p in providers: validate_provider(p)
    questions = {c["id"]: c["question"] for c in cases}
    if len(questions) != len(cases): raise SystemExit("duplicate case id")
    if args.dry_run:
        print(json.dumps({"status":"valid","cases":len(cases),"providers":[p["id"] for p in providers],"enabled_providers":[p["id"] for p in providers if p.get("enabled") is True],"model_calls":0}, ensure_ascii=False))
        return
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(c,p) for c in cases for p in providers]
    random.Random(args.seed).shuffle(jobs)
    records = []
    for case, provider in jobs:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False, dir=run_dir) as f:
            f.write(case["question"]); prompt_path = Path(f.name)
        cmd = [x.replace("{prompt_file}", str(prompt_path)).replace("{python}", sys.executable)
               for x in provider["command"]]
        started_at = iso_now(); started = time.perf_counter(); timed_out = False
        try:
            from tools.environments.local import build_subprocess_env
            from hermes_constants import set_hermes_home_override, reset_hermes_home_override
            with tempfile.TemporaryDirectory(prefix="clinical-eval-home-") as home:
                token = set_hermes_home_override(home)
                try:
                    env = build_subprocess_env(base={"PATH": os.environ.get("PATH", "")},
                                               extra={"HERMES_HOME": home, "HERMES_RUNTIME_DIR": home, "HOME": home})
                finally:
                    reset_hermes_home_override(token)
                cp = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout,
                                    cwd=ROOT.parents[1], env=env, check=False)
            stdout, stderr, code = redact(cp.stdout.strip()), redact(cp.stderr.strip()), cp.returncode
        except OSError as e:
            stdout, stderr, code = "", type(e).__name__, None
        except subprocess.TimeoutExpired as e:
            def timeout_text(value):
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return value or ""
            stdout = redact(timeout_text(e.stdout).strip())
            stderr = redact(timeout_text(e.stderr).strip())
            code, timed_out = None, True
        finally:
            prompt_path.unlink(missing_ok=True)
        latency = round(time.perf_counter() - started, 3)
        rec = {"run_id":run_id,"case_id":case["id"],"provider_id":provider["id"],"model":provider["model"],
               "question_sha256":hashlib.sha256(case["question"].encode()).hexdigest(),"started_at":started_at,
               "wall_latency_seconds":latency,"exit_code":code,"timeout":timed_out,"response":stdout,
               "stderr":stderr,"reported_input_tokens":None,"reported_output_tokens":None,"reported_cost_usd":None}
        rec.update(response_metrics(stdout)); records.append(rec)
        with (run_dir/"responses.jsonl").open("a",encoding="utf-8") as out: out.write(json.dumps(rec,ensure_ascii=False)+"\n")
    aliases = list(range(len(records))); random.Random(args.seed ^ 0xA11F).shuffle(aliases)
    blind, key = [], {}
    for alias_num, rec in zip(aliases, records):
        alias=f"R{alias_num+1:04d}"; key[alias]={"case_id":rec["case_id"],"provider_id":rec["provider_id"],"model":rec["model"]}
        blind.append({"response_id":alias,"case_id":rec["case_id"],"question":questions[rec["case_id"]],"response":rec["response"]})
    (run_dir/"blind_responses.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in blind),encoding="utf-8")
    (run_dir/"identity_key.json").write_text(json.dumps(key,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    worksheet=[]
    for x in blind:
        worksheet.append({"response_id":x["response_id"],"rater_id":"","clinical":{"safety":None,"accuracy":None,"evidence_retrieved":None,"currency":None,"traceability":None,"calibration":None,"utility":None},"gates":{"critical_safety_failure":None,"unsupported_clinical_number":None,"fabricated_source":None},"notes":""})
    (run_dir/"clinical_scores.template.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in worksheet),encoding="utf-8")
    style=[{"response_id":x["response_id"],"rater_id":"","style":{"clarity":None,"conciseness":None,"empathy":None,"organization":None,"language_fit":None},"notes":""} for x in blind]
    (run_dir/"style_scores.template.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in style),encoding="utf-8")
    manifest={"run_id":run_id,"created_at":iso_now(),"seed":args.seed,"dataset_sha256":hashlib.sha256(args.dataset.read_bytes()).hexdigest(),"models_sha256":hashlib.sha256(args.models.read_bytes()).hexdigest(),"case_count":len(cases),"provider_count":len(providers),"response_count":len(records),"completed_ok":sum(r["exit_code"]==0 and not r["timeout"] for r in records),"comparison_complete":False,"note":"Requires blinded clinical ratings and adjudication before comparison."}
    (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest))
if __name__ == "__main__":
    # Direct script invocation needs the checkout for the shared subprocess environment factory.
    sys.path.insert(0, str(ROOT.parents[1]))
    main()
