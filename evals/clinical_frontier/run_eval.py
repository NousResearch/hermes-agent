#!/usr/bin/env python3
"""Paired clinical-response runner. Stdlib only; never invokes Anthropic."""
from __future__ import annotations
import argparse, hashlib, json, random, re, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORBIDDEN = ("anthropic", "claude")
SECRET_RE = re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret)(\s*[:=]\s*|\s+)[^\s,;]+")
URL_RE = re.compile(r"https?://[^\s)\]>]+")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:\s?(?:%|°C|mg|mL|min|horas?|días?))?")

def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def redact(text: str) -> str:
    return SECRET_RE.sub(lambda m: m.group(1) + m.group(2) + "[REDACTED]", text)

def iso_now():
    return datetime.now(timezone.utc).isoformat()

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset.jsonl")
    ap.add_argument("--models", type=Path, default=ROOT / "models.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs")
    ap.add_argument("--case", action="append", dest="cases")
    ap.add_argument("--provider", action="append", dest="providers")
    ap.add_argument("--timeout", type=int, default=210)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cases = load_jsonl(args.dataset)
    cfg = json.loads(args.models.read_text(encoding="utf-8"))
    providers = [p for p in cfg["providers"] if p.get("enabled")]
    if args.cases: cases = [c for c in cases if c["id"] in args.cases]
    if args.providers: providers = [p for p in providers if p["id"] in args.providers]
    if not cases or not providers: raise SystemExit("no selected cases/providers")
    for p in providers: validate_provider(p)
    questions = {c["id"]: c["question"] for c in cases}
    if len(questions) != len(cases): raise SystemExit("duplicate case id")
    if args.dry_run:
        print(json.dumps({"status":"valid","cases":len(cases),"providers":[p["id"] for p in providers]}, ensure_ascii=False))
        return
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    jobs = [(c,p) for c in cases for p in providers]
    random.Random(args.seed).shuffle(jobs)
    records = []
    for case, provider in jobs:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False, dir=run_dir) as f:
            f.write(case["question"]); prompt_path = Path(f.name)
        cmd = [x.replace("{prompt_file}", str(prompt_path)) for x in provider["command"]]
        started_at = iso_now(); started = time.perf_counter(); timed_out = False
        try:
            cp = subprocess.run(cmd, text=True, capture_output=True, timeout=args.timeout, cwd=ROOT)
            stdout, stderr, code = redact(cp.stdout.strip()), redact(cp.stderr.strip()), cp.returncode
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
    manifest={"run_id":run_id,"created_at":iso_now(),"seed":args.seed,"case_count":len(cases),"provider_count":len(providers),"response_count":len(records),"completed_ok":sum(r["exit_code"]==0 and not r["timeout"] for r in records),"comparison_complete":False,"note":"Requires blinded clinical ratings and adjudication before comparison."}
    (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifest))
if __name__ == "__main__": main()
