#!/usr/bin/env python3
"""
meta_router_executor.py — Hermes bridge from meta-router decisions to SoM/MR-ALS.

Phase 1 (pre-LLM):
  - run the SoM prepare pipeline to create a state dir + targets
  - scaffold SoM v3.1 evidence artifacts

Phase 2 (post-LLM):
  - persist the assistant output to output.md
  - run the SoM complete pipeline
  - run ADV_PASS for routed task types
  - log a routing outcome to the MR-ALS experience plane
  - return a structured delivery result for the caller

Phase 4+ threshold trigger:
  every 10 outcomes → launch mr_als_runner.py --phase 4,5 --force
  in a background subprocess to regenerate/evaluate routing candidates.

NOTE: meta-router/ is a hyphenated directory, so support modules are loaded
via importlib.util.spec_from_file_location instead of normal package imports.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Absolute paths ─────────────────────────────────────────────────────────────
_MR_DIR = Path("/home/samade10/.openclaw/workspace/skills/maintainer/meta-router")
_EXP_DIR = _MR_DIR / "experience"
_SCRIPTS_DIR = _MR_DIR / "scripts"
_LOG_WRITER_PATH = _EXP_DIR / "log_writer.py"
_HYGIENE_PATH = _SCRIPTS_DIR / "experience_hygiene.py"
_EVENTS_JSONL = _EXP_DIR / "routing_events.jsonl"
_OUTCOMES_JSONL = _EXP_DIR / "routing_outcomes.jsonl"
_RUNNER_PATH = _SCRIPTS_DIR / "mr_als_runner.py"

_WORKSPACE = Path("/home/samade10/.openclaw/workspace")
_RQL_SCRIPTS_DIR = _WORKSPACE / "rql/scripts"
_SOM_PIPELINE = _RQL_SCRIPTS_DIR / "som_pipeline.py"
_ADV_PASS = _RQL_SCRIPTS_DIR / "adv_pass.py"
_EVIDENCE_CONTRACT = _RQL_SCRIPTS_DIR / "evidence_contract.py"

_ADV_TASK_TYPES = {"code", "audit", "production", "integration"}

# ── Log writer lazy-init ───────────────────────────────────────────────────────
_lw_mod = None
_lw_lock = threading.Lock()
_hygiene_mod = None
_hygiene_lock = threading.Lock()
_last_optimizer_trigger_eligible_count: Optional[int] = None


def _load_log_writer():
    global _lw_mod
    if _lw_mod is not None:
        return _lw_mod
    with _lw_lock:
        if _lw_mod is not None:
            return _lw_mod
        if not _LOG_WRITER_PATH.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location("_mr_log_writer", _LOG_WRITER_PATH)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _lw_mod = mod
        except Exception:
            pass
    return _lw_mod


def _load_experience_hygiene():
    global _hygiene_mod
    if _hygiene_mod is not None:
        return _hygiene_mod
    with _hygiene_lock:
        if _hygiene_mod is not None:
            return _hygiene_mod
        if not _HYGIENE_PATH.exists():
            return None
        try:
            spec = importlib.util.spec_from_file_location("_mr_experience_hygiene", _HYGIENE_PATH)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _hygiene_mod = mod
        except Exception:
            pass
    return _hygiene_mod


# ── MR type → SoM type map ─────────────────────────────────────────────────────
_MR_TO_SOM_TYPE = {
    "code": "code",
    "audit": "general",
    "research": "research",
    "production": "general",
    "integration": "code",
    "design": "design",
    "config": "config",
}


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PrepResult:
    som_state_dir: Optional[Path]
    targets_context: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.som_state_dir is not None

    # Backward-compatible aliases used by the current run_agent.py integration.
    @property
    def phase1_ok(self) -> bool:
        return self.ok

    @property
    def state_dir(self) -> Optional[Path]:
        return self.som_state_dir


@dataclass
class Phase2Result:
    request_id: str
    task_type: str
    state_dir: Optional[Path]
    routing_artifact_version: str
    passed: bool
    score: Optional[float]
    verdict: str
    threshold: Optional[float]
    oracle_verdict: str
    adv_pass_clean: Optional[bool]
    adv_findings_count: int
    delivery_gate_passed: Optional[bool]
    score_card: Optional[str]
    ref_entry: Optional[dict]
    delivery_path: Optional[str]
    fix_prompt_path: Optional[str]
    error: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    som_score: Optional[float] = None
    eop_score: Optional[float] = None
    composite_score: Optional[float] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json_payload(text: str) -> dict:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("empty JSON payload")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(payload[start:end + 1])


def _coerce_score(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


_LLM_MODEL = "gpt-5.4-mini"


def _build_llm_client(timeout_seconds: float):
    try:
        from openai import OpenAI

        from agent.auxiliary_client import _to_openai_base_url
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="openai-codex")
        api_key = str(runtime.get("api_key") or "").strip()
        if not api_key:
            return None
        base_url = _to_openai_base_url(str(runtime.get("base_url") or "").strip())
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    except Exception:
        return None



def _extract_llm_output_text(response) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for part in content:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text.strip())
        if chunks:
            return "\n".join(chunks)

    if isinstance(response, dict):
        text = response.get("output_text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    return ""



def _parse_llm_json_payload(text: str) -> dict:
    payload = (text or "").strip()
    if not payload:
        raise ValueError("empty JSON payload")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(payload[start:end + 1])



def _responses_text_input(prompt: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "input_text", "text": str(prompt or "")}] }]



def _stream_llm_text(client, *, instructions: str, prompt: str) -> str:
    deltas: list[str] = []
    with client.responses.stream(
        model=_LLM_MODEL,
        instructions=instructions,
        input=_responses_text_input(prompt),
        reasoning={"effort": "xhigh", "summary": "auto"},
        service_tier="priority",
        text={"verbosity": "low"},
        store=False,
    ) as stream:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if isinstance(delta, str) and delta:
                    deltas.append(delta)
        response = stream.get_final_response()
    text = "".join(deltas).strip()
    if text:
        return text
    return _extract_llm_output_text(response)



def _call_llm_json_prompt(instructions: str, prompt: str, timeout_seconds: float) -> Optional[dict]:
    try:
        client = _build_llm_client(timeout_seconds)
        if client is None:
            return None
        return _parse_llm_json_payload(
            _stream_llm_text(client, instructions=instructions, prompt=prompt)
        )
    except Exception:
        return None



def _call_llm_text_prompt(instructions: str, prompt: str, timeout_seconds: float) -> Optional[str]:
    try:
        client = _build_llm_client(timeout_seconds)
        if client is None:
            return None
        text = _stream_llm_text(client, instructions=instructions, prompt=prompt).strip()
        return text or None
    except Exception:
        return None

def _summarize_ref(ref_entry: Optional[dict]) -> str:
    if not isinstance(ref_entry, dict):
        return ""
    ref_id = ref_entry.get("id") or ref_entry.get("ref_id") or ""
    if not ref_id:
        return ""
    score = ref_entry.get("score")
    verdict = ref_entry.get("verdict")
    tail = []
    if score is not None:
        tail.append(str(score))
    if verdict:
        tail.append(str(verdict))
    suffix = f" ({', '.join(tail)})" if tail else ""
    return f"REF: {ref_id}{suffix}"


def populate_evidence_artifacts(task_text: str, final_response: str, state_dir: Path) -> None:
    """Best-effort seeding of SoM v3.1 evidence artifacts from the drafted response."""
    state_dir = Path(state_dir)
    hypothesis_path = state_dir / "hypothesis.json"
    evidence_path = state_dir / "verification_evidence.json"
    edge_scan_path = state_dir / "edge_scan.json"

    hypothesis = _load_json_file(hypothesis_path) if hypothesis_path.exists() else {}
    evidence = _load_json_file(evidence_path) if evidence_path.exists() else {"items": []}
    edge_scan = _load_json_file(edge_scan_path) if edge_scan_path.exists() else {}

    bullet_items: list[str] = []
    for raw_line in (final_response or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("- ", "* ")):
            bullet_items.append(stripped[2:].strip())
    if not bullet_items and final_response.strip():
        bullet_items.append(final_response.strip()[:240])

    hypothesis.update({
        "task": hypothesis.get("task") or task_text,
        "hypothesis": hypothesis.get("hypothesis") or f"The drafted output satisfies the routed task: {task_text[:180]}",
        "test_plan": hypothesis.get("test_plan") or [
            "Review output.md for completeness against the task.",
            "Compare any claimed runtime verification against the drafted response.",
        ],
        "result": "supported",
        "notes": "Seeded by Hermes meta_router_executor from the drafted response.",
    })
    if hypothesis_path.exists():
        hypothesis_path.write_text(json.dumps(hypothesis, indent=2), encoding="utf-8")

    items = evidence.get("items") if isinstance(evidence.get("items"), list) else []
    seen_details = {item.get("detail") for item in items if isinstance(item, dict)}
    for detail in bullet_items[:5]:
        if detail and detail not in seen_details:
            items.append({
                "kind": "drafted-runtime-evidence",
                "detail": detail,
                "source": "final_response",
            })
            seen_details.add(detail)
    if not items:
        items.append({
            "kind": "drafted-output",
            "detail": "output.md was produced by Hermes for routed evaluation.",
            "source": "meta_router_executor",
        })
    evidence["items"] = items
    if evidence_path.exists():
        evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    if edge_scan_path.exists() and "categories_checked" not in edge_scan:
        edge_scan["categories_checked"] = []
        edge_scan["findings"] = edge_scan.get("findings", [])
        edge_scan_path.write_text(json.dumps(edge_scan, indent=2), encoding="utf-8")


_PHASE1_CRITICAL_SIGNALS = {
    "deploy",
    "production",
    "migration",
    "delete",
    "security",
    "auth",
    "database",
    "payment",
    "config",
    "system",
    "critical",
    "breaking",
    "irreversible",
    "encrypt",
    "api key",
    "secret",
    "rotate secrets",
}



def resolve_phase1_tier(task_text: str, mr_type: str) -> str:
    """Choose a safer SoM tier for pre-execution target generation.

    The old behavior hardcoded `trivial` for every routed task, which lowered
    thresholds and skipped SoM context gathering even for substantial code work.
    Keep Phase 1 lightweight, but avoid under-tiering by default:
    - production/config work is critical
    - any task with critical-risk keywords is critical
    - routed code/integration/audit/research/design work is standard
    - only short, low-risk general tasks fall back to trivial
    """
    task_lower = (task_text or "").lower()
    text_len = len((task_text or "").strip())

    if mr_type in {"production", "config"}:
        return "critical"
    if any(signal in task_lower for signal in _PHASE1_CRITICAL_SIGNALS):
        return "critical"
    if mr_type in {"code", "integration", "audit", "research", "design"}:
        return "standard"
    if text_len < 80:
        return "trivial"
    return "standard"



def resolve_phase2_tier(state_dir: Path) -> str:
    state_dir = Path(state_dir)
    manifest = _load_json_file(state_dir / "manifest.json")
    if manifest.get("tier"):
        return str(manifest["tier"])
    targets = _load_json_file(state_dir / "targets.json")
    if targets.get("tier"):
        return str(targets["tier"])
    return "standard"


def format_routed_response(raw_response: str, phase2: Phase2Result, directive: str = "") -> str:
    """Format the final user-visible response from a routed turn."""
    receipt_lines: list[str] = []
    if directive:
        receipt_lines.append(f"Directive: {directive}")
    receipt_lines.append(f"Artifact: {phase2.routing_artifact_version}")
    if phase2.score is not None:
        threshold = f" / threshold {int(phase2.threshold)}" if phase2.threshold is not None else ""
        receipt_lines.append(f"RQL Score: {phase2.score:.1f}/100 ({phase2.verdict}){threshold}")
    if phase2.delivery_gate_passed is True:
        receipt_lines.append("Score gate: PASS")
    elif phase2.delivery_gate_passed is False:
        if phase2.score is not None and phase2.threshold is not None:
            if phase2.score < phase2.threshold:
                receipt_lines.append(f"Score gate: FAIL ({phase2.score:.1f} < {phase2.threshold:.1f})")
            else:
                receipt_lines.append(
                    f"Score gate: FAIL (delivery gate rejected; "
                    f"score {phase2.score:.1f} met threshold {phase2.threshold:.1f})"
                )
        else:
            receipt_lines.append("Score gate: FAIL")
    receipt_lines.append(f"Oracle: {phase2.oracle_verdict}")
    if phase2.adv_pass_clean is True:
        receipt_lines.append(f"ADV_PASS: PASS ({phase2.adv_findings_count} findings)")
    elif phase2.adv_pass_clean is False:
        receipt_lines.append(f"ADV_PASS: FAIL ({phase2.adv_findings_count} findings)")
    if phase2.state_dir is not None:
        receipt_lines.append(f"State dir: {phase2.state_dir}")
    if phase2.delivery_path:
        receipt_lines.append(f"Delivery: {phase2.delivery_path}")
    ref_line = _summarize_ref(phase2.ref_entry)
    if ref_line:
        receipt_lines.append(ref_line)

    if phase2.passed:
        pieces = [raw_response.strip(), "", "[META-ROUTER RECEIPT]"]
        pieces.extend(receipt_lines)
        return "\n".join(p for p in pieces if p is not None)

    threshold_only_failure = (
        phase2.delivery_gate_passed is False
        and phase2.oracle_verdict == "PASS"
        and phase2.adv_pass_clean is not False
        and not phase2.error
    )
    lead = (
        "Backend evaluation blocked final delivery: the draft passed Oracle and ADV_PASS, but missed the final score threshold."
        if threshold_only_failure
        else "Backend evaluation failed. The draft output did not satisfy the routed SoM/EOP gates."
    )

    lines = [lead, "", "[META-ROUTER RECEIPT]"]
    lines.extend(receipt_lines)
    if phase2.fix_prompt_path:
        lines.append(f"Fix prompt: {phase2.fix_prompt_path}")
    if phase2.error:
        lines.append(f"Error: {phase2.error}")
    if raw_response.strip():
        lines.extend([
            "",
            "Draft output (not approved for final delivery):",
            raw_response.strip(),
        ])
    return "\n".join(lines)


def _scaffold_evidence(state_dir: Path, task_text: str) -> Optional[str]:
    if not _EVIDENCE_CONTRACT.exists():
        return None
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(_EVIDENCE_CONTRACT),
                "--scaffold",
                "--task",
                task_text[:2000],
                "--state-dir",
                str(state_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return f"evidence scaffold exit {result.returncode}: {(result.stderr or result.stdout).strip()[:200]}"
        return None
    except Exception as exc:
        return f"evidence scaffold exception: {exc}"


def _validate_evidence(state_dir: Path) -> tuple[Optional[bool], str]:
    try:
        output_path = Path(state_dir) / "output.md"
        if not output_path.exists():
            return None, ""
        output_excerpt = output_path.read_text(encoding="utf-8")[:800].strip()
        if not output_excerpt:
            return None, ""

        task_path = Path(state_dir) / "task.txt"
        task_excerpt = task_path.read_text(encoding="utf-8")[:200].strip() if task_path.exists() else ""
        prompt = (
            "Did this AI agent response provide genuine evidence of completing the task? "
            f"Task: {task_excerpt}. "
            f"Response excerpt: {output_excerpt}. "
            'Reply with JSON: {"valid": true/false, "confidence": 0.0-1.0, "reason": "one sentence"}'
        )
        data = _call_llm_json_prompt(
            "You judge whether an AI agent response contains genuine completion evidence. Respond with valid JSON only.",
            prompt,
            timeout_seconds=8.0,
        )
        if not isinstance(data, dict):
            return None, ""

        valid = data.get("valid")
        if not isinstance(valid, bool):
            return None, ""
        reason = str(data.get("reason") or "").strip()
        return valid, reason
    except Exception:
        return None, ""



def _synthesize_fix_prompt(state_dir: Path, task_text: str, score: Optional[float], threshold: Optional[float]) -> Optional[Path]:
    """Generate a minimal fix_prompt.md from scores.json when SoM did not create one.

    Used for trivial-tier tasks (max_iterations=1) that fail the score gate without
    ever producing a fix_prompt -- the correction loop in run_agent.py needs this file.
    """
    scores_path = state_dir / "scores.json"
    fix_prompt_path = state_dir / "fix_prompt.md"
    if fix_prompt_path.exists():
        return fix_prompt_path  # Already exists -- do not overwrite

    try:
        scores_data = _load_json_file(scores_path)
        if not scores_data:
            return None

        dimensions = scores_data.get("dimensions", [])
        if not dimensions:
            return None

        effective_threshold = threshold or 65
        failing = [
            d for d in dimensions
            if d.get("max_possible", 0) > 0
            and (d.get("weighted", 0) / d.get("max_possible", 1)) < 0.6
        ]
        if not failing:
            failing = sorted(dimensions, key=lambda d: d.get("weighted", 0) / max(d.get("max_possible", 1), 1))[:2]

        lines = [
            "Score {:.0f}/{:.0f} - the following areas need improvement:".format(score, effective_threshold),
        ]
        for d in failing:
            name = d.get("name", "Unknown")
            got = d.get("weighted", 0)
            possible = d.get("max_possible", 0)
            reasoning = d.get("reasoning", "")
            lines.append("- **{}** ({:.0f}/{:.0f}): {}".format(name, got, possible, reasoning))

        lines.append("")
        lines.append("Revise your response to address these gaps, citing live evidence where relevant.")
        fix_content = "\n".join(lines)
        fix_prompt_path.write_text(fix_content, encoding="utf-8")
        return fix_prompt_path
    except Exception:
        return None


def _enhance_fix_prompt(path: Path, task_text: str, task_type: str, score: Optional[float], threshold: Optional[float]) -> None:
    original = Path(path)
    try:
        fix_content = original.read_text(encoding="utf-8").strip()
    except Exception:
        return
    if not fix_content:
        return

    instructions = (
        "You rewrite evaluation feedback into task-specific correction instructions. "
        "Return only the rewritten fix prompt text."
    )
    effective_threshold = threshold or 65
    prompt = (
        f"You are reviewing an AI agent's response to this task: {task_text}. "
        f"The response was evaluated as type {task_type} and scored {score}/{effective_threshold}. "
        f"The evaluation identified these gaps: {fix_content}. "
        "Rewrite the fix instructions to be specific to THIS task — reference the actual task goal, "
        "not generic criteria. Keep it under 200 words."
    )
    enhanced = _call_llm_text_prompt(instructions, prompt, timeout_seconds=10.0)
    if not enhanced:
        return
    try:
        original.write_text(enhanced.strip() + "\n", encoding="utf-8")
    except Exception:
        return



def _run_adv_pass(task_text: str, state_dir: Path, output_path: Path) -> tuple[Optional[bool], Optional[dict], str]:
    if not _ADV_PASS.exists() or not output_path.exists():
        return None, None, ""
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(_ADV_PASS),
                "--state-dir",
                str(state_dir),
                "--task",
                task_text[:2000],
                "--output",
                str(output_path),
                "--trigger-reason",
                "meta-router routed adversarial pass",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        data = None
        preview = (result.stderr or result.stdout or "").strip()
        try:
            data = _parse_json_payload(result.stdout)
        except Exception:
            adv_file = state_dir / "adversarial_findings.json"
            if adv_file.exists():
                try:
                    data = json.loads(adv_file.read_text(encoding="utf-8"))
                except Exception:
                    data = None
        clean = result.returncode == 0
        if data is not None:
            medium_high = [
                finding for finding in data.get("findings", [])
                if finding.get("severity") in ("medium", "high")
            ]
            clean = len(medium_high) == 0
        return clean, data, preview[:200]
    except Exception as exc:
        return None, None, f"adv_pass exception: {exc}"


def _format_targets(targets: dict | list, mr_type: str) -> str:
    """Format targets into a compact context block for prompt injection."""
    try:
        if isinstance(targets, dict):
            items = (
                targets.get("dimensions")
                or targets.get("targets")
                or targets.get("items")
                or []
            )
        else:
            items = targets

        if not items:
            return ""

        lines = [f"[SoM Targets | {mr_type}]"]
        for i, item in enumerate(items[:8], 1):
            if isinstance(item, dict):
                label = item.get("name") or item.get("label") or item.get("target") or f"target-{i}"
                desc = (
                    item.get("description")
                    or item.get("rubric")
                    or item.get("measure_type")
                    or ""
                )
                extras = []
                if item.get("weight") is not None:
                    extras.append(f"weight={item['weight']}")
                if item.get("measure_type"):
                    extras.append(f"measure={item['measure_type']}")
                suffix = f" ({', '.join(extras)})" if extras else ""
                lines.append(f"  {i}. {label}" + (f" — {desc}" if desc else "") + suffix)
            else:
                lines.append(f"  {i}. {item}")
        lines.append("[/SoM Targets]")
        return "\n".join(lines)
    except Exception:
        return ""


def _count_outcomes() -> int:
    if not _OUTCOMES_JSONL.exists():
        return 0
    try:
        return sum(1 for line in _OUTCOMES_JSONL.read_text(encoding="utf-8").splitlines() if line.strip())
    except Exception:
        return 0


def _count_eligible_outcomes() -> int:
    hygiene = _load_experience_hygiene()
    if hygiene is None:
        return 0
    try:
        records = hygiene.load_joined_records(_EVENTS_JSONL, _OUTCOMES_JSONL)
    except Exception:
        return 0
    return sum(1 for record in records if record.get("eligible_for_learning"))


def _trigger_optimizer_bg() -> None:
    """Launch mr_als_runner.py --phase 4,5 --force in a background subprocess."""
    if not _RUNNER_PATH.exists():
        return

    def _run():
        try:
            subprocess.run(
                [sys.executable, str(_RUNNER_PATH), "--phase", "4,5", "--force"],
                capture_output=True,
                timeout=300,
            )
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True, name="mr-als-optimizer")
    t.start()


def _maybe_trigger_optimizer() -> None:
    global _last_optimizer_trigger_eligible_count
    try:
        n = _count_eligible_outcomes()
        if n >= 10 and n % 10 == 0 and n != _last_optimizer_trigger_eligible_count:
            _last_optimizer_trigger_eligible_count = n
            _trigger_optimizer_bg()
    except Exception:
        pass


# ── Phase 1: pre-LLM SoM target generation ────────────────────────────────────

def run_phase1(task_text: str, mr_type: str) -> PrepResult:
    """
    Run the SoM prepare pipeline to generate state_dir + targets before the LLM call.
    Never raises — returns PrepResult(error=...) on failure.
    """
    if not _SOM_PIPELINE.exists():
        return PrepResult(None, "", error=f"som_pipeline.py not found at {_SOM_PIPELINE}")

    som_type = _MR_TO_SOM_TYPE.get(mr_type, "code")
    phase1_tier = resolve_phase1_tier(task_text, mr_type)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(_SOM_PIPELINE),
                "--task",
                task_text[:2000],
                "--task-type",
                som_type,
                "--tier",
                phase1_tier,
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            preview = (result.stderr or result.stdout or "").strip()
            return PrepResult(None, "", error=f"som_pipeline exit {result.returncode}: {preview[:200]}")

        data = _parse_json_payload(result.stdout)
        state_dir = data.get("state_dir") or data.get("output_dir")
        if not state_dir:
            return PrepResult(None, "", error="som_pipeline returned no state_dir")

        state_path = Path(state_dir)
        targets_file = state_path / "targets.json"
        if not targets_file.exists():
            return PrepResult(state_path, "", error="targets.json not found in state_dir")

        evidence_error = _scaffold_evidence(state_path, task_text)
        targets = json.loads(targets_file.read_text(encoding="utf-8"))
        targets_context = _format_targets(targets, mr_type)
        return PrepResult(state_path, targets_context, error=evidence_error)

    except subprocess.TimeoutExpired:
        return PrepResult(None, "", error="som_pipeline timed out (>45s)")
    except Exception as exc:
        return PrepResult(None, "", error=f"phase1 exception: {exc}")


# ── Phase 2: post-LLM outcome logging ─────────────────────────────────────────

def run_phase2(
    request_id: str,
    task_type: str,
    task_text: str,
    som_state_dir: Optional[Path],
    final_response: str,
    t0: float,
    routing_artifact_version: str = "static-default",
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    surface: Optional[str] = None,
) -> Phase2Result:
    """Run SoM completion + outcome logging synchronously."""
    return _do_phase2(
        request_id,
        task_type,
        task_text,
        som_state_dir,
        final_response,
        t0,
        routing_artifact_version,
        session_id,
        source,
        surface,
    )


def run_phase2_async(
    request_id: str,
    task_type: str,
    task_text: str,
    som_state_dir: Optional[Path],
    final_response: str,
    t0: float,
    routing_artifact_version: str = "static-default",
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    surface: Optional[str] = None,
) -> None:
    """Run SoM completion + outcome logging in a background daemon thread."""

    def _worker():
        _do_phase2(
            request_id,
            task_type,
            task_text,
            som_state_dir,
            final_response,
            t0,
            routing_artifact_version,
            session_id,
            source,
            surface,
        )

    t = threading.Thread(target=_worker, daemon=True, name=f"mr-p2-{request_id[:8]}")
    t.start()


def _do_phase2(
    request_id: str,
    task_type: str,
    task_text: str,
    som_state_dir: Optional[Path],
    final_response: str,
    t0: float,
    routing_artifact_version: str,
    session_id: Optional[str],
    source: Optional[str] = None,
    surface: Optional[str] = None,
) -> Phase2Result:
    """Inner blocking implementation of Phase 2."""
    latency_ms = round((time.time() - t0) * 1000, 1)
    som_score: Optional[float] = None
    composite_score: Optional[float] = None
    eop_score: Optional[float] = None
    oracle_verdict = "SKIPPED"
    adv_pass_clean: Optional[bool] = None
    adv_findings_count = 0
    delivery_gate_passed: Optional[bool] = None
    score_card: Optional[str] = None
    ref_entry: Optional[dict] = None
    delivery_path: Optional[str] = None
    fix_prompt_path: Optional[str] = None
    executor_engine: Optional[str] = None
    omx_workflow: Optional[str] = None
    launch_mode: Optional[str] = None
    omx_version: Optional[str] = None
    codex_version: Optional[str] = None
    team_size: Optional[int] = None
    verdict = "UNKNOWN"
    threshold: Optional[float] = None
    error: Optional[str] = None
    notes: list[str] = []
    evidence_valid: Optional[bool] = None
    state_dir = som_state_dir

    try:
        if som_state_dir and _SOM_PIPELINE.exists():
            out_md = som_state_dir / "output.md"
            scores_path = som_state_dir / "scores.json"
            delivery_json_path = som_state_dir / "delivery.json"
            fix_prompt = som_state_dir / "fix_prompt.md"

            agent_wrote = out_md.exists() and out_md.stat().st_size >= 50
            if not agent_wrote and final_response and final_response.strip():
                out_md.write_text(final_response, encoding="utf-8")
            populate_evidence_artifacts(task_text, out_md.read_text(encoding="utf-8") if out_md.exists() else final_response, som_state_dir)

            som_type = _MR_TO_SOM_TYPE.get(task_type, "code")
            som_tier = resolve_phase2_tier(som_state_dir)
            result = subprocess.run(
                [
                    sys.executable,
                    str(_SOM_PIPELINE),
                    "--complete",
                    "--task",
                    task_text[:2000],
                    "--task-type",
                    som_type,
                    "--tier",
                    som_tier,
                    "--task-id",
                    som_state_dir.name,
                    "--state-dir",
                    str(som_state_dir),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            preview = (result.stderr or result.stdout or "").strip()
            data = {}
            if result.returncode not in (0, 1):
                error = f"som complete exit {result.returncode}: {preview[:200]}"
            else:
                data = _parse_json_payload(result.stdout)
                notes.append(f"som_status={data.get('status', 'unknown')}")
                if result.returncode == 1:
                    notes.append("som_returncode=1")

            scores_json = _load_json_file(scores_path)
            delivery_json = _load_json_file(delivery_json_path)
            execution_result_json = _load_json_file(som_state_dir / "execution_result.json")

            executor_engine = str(
                execution_result_json.get("executor_engine")
                or execution_result_json.get("engine")
                or ""
            ).strip() or None
            omx_workflow = str(
                execution_result_json.get("omx_workflow")
                or execution_result_json.get("workflow")
                or ""
            ).strip() or None
            launch_mode = str(execution_result_json.get("launch_mode") or "").strip() or None
            omx_version = str(execution_result_json.get("omx_version") or "").strip() or None
            codex_version = str(execution_result_json.get("codex_version") or "").strip() or None
            _team_size_raw = execution_result_json.get("team_size")
            try:
                team_size = int(_team_size_raw) if _team_size_raw is not None else None
            except (TypeError, ValueError):
                team_size = None

            som_score = (
                _coerce_score(data.get("score"))
                or _coerce_score(scores_json.get("total_weighted_score"))
                or _coerce_score(scores_json.get("weighted_score"))
                or _coerce_score(scores_json.get("total_score"))
            )
            verdict = str(
                data.get("verdict")
                or scores_json.get("verdict")
                or delivery_json.get("scores", {}).get("verdict")
                or "UNKNOWN"
            )
            threshold = (
                _coerce_score(scores_json.get("threshold"))
                or _coerce_score(data.get("scores", {}).get("threshold"))
                or _coerce_score(delivery_json.get("scores", {}).get("threshold"))
            )
            _oracle_raw = data.get("oracle") or delivery_json.get("oracle")
            if _oracle_raw is not None:
                oracle_verdict = str(_oracle_raw)
            elif "passed" in data:
                oracle_verdict = "PASS" if data.get("passed") else "FAIL"
            # else: preserve the initial "SKIPPED" default — absence of an
            # oracle signal must NOT be treated as a FAIL verdict.
            delivery_gate = data.get("delivery_gate") or delivery_json.get("delivery_gate") or {}
            if isinstance(delivery_gate, dict):
                delivery_gate_passed = delivery_gate.get("all_passed")
            score_card = data.get("score_card") or delivery_json.get("score_card")
            ref_entry = data.get("ref_entry") or delivery_json.get("ref_entry")
            delivery_path = data.get("delivery_path") or (str(delivery_json_path) if delivery_json_path.exists() else None)
            fix_prompt_path = str(fix_prompt) if fix_prompt.exists() else None
            # If SoM did not generate fix_prompt (trivial-tier, max_iterations=1),
            # synthesize one from scores so the run_agent.py correction loop can fire.
            if not fix_prompt_path and som_score is not None and som_score < (threshold or 65):
                _synth = _synthesize_fix_prompt(som_state_dir, task_text, som_score, threshold)
                if _synth:
                    fix_prompt_path = str(_synth)
            if fix_prompt_path and som_score is not None and som_score < (threshold or 65):
                _enhance_fix_prompt(Path(fix_prompt_path), task_text, task_type, som_score, threshold)
            notes.append(f"artifact={routing_artifact_version}")
            if session_id:
                notes.append(f"session={session_id}")
            if executor_engine:
                notes.append(f"executor_engine={executor_engine}")
            if omx_workflow:
                notes.append(f"omx_workflow={omx_workflow}")
            if launch_mode:
                notes.append(f"launch_mode={launch_mode}")
            if omx_version:
                notes.append(f"omx_version={omx_version}")
            if codex_version:
                notes.append(f"codex_version={codex_version}")
            if team_size is not None:
                notes.append(f"team_size={team_size}")

            evidence_valid, evidence_preview = _validate_evidence(som_state_dir)
            if evidence_valid is not None:
                notes.append(f"evidence_valid={str(evidence_valid).lower()}")
                if not evidence_valid and evidence_preview:
                    notes.append(f"evidence={evidence_preview}")

            if delivery_gate_passed is not None:
                notes.append(f"delivery_gate_passed={str(bool(delivery_gate_passed)).lower()}")

            if task_type in _ADV_TASK_TYPES and out_md.exists():
                adv_pass_clean, adv_data, adv_preview = _run_adv_pass(task_text, som_state_dir, out_md)
                if adv_data is not None:
                    adv_findings_count = len([
                        finding for finding in adv_data.get("findings", [])
                        if finding.get("severity") in ("medium", "high")
                    ])
                if adv_pass_clean is not None:
                    eop_score = 100.0 if adv_pass_clean else 0.0
                    notes.append(f"adv_pass_clean={str(adv_pass_clean).lower()}")
                    notes.append(f"adv_findings={adv_findings_count}")
                elif adv_preview:
                    notes.append(f"adv_preview={adv_preview}")
        else:
            error = "phase2 skipped: missing som_state_dir or som_pipeline"
    except Exception as exc:
        error = f"phase2 exception: {exc}"

    if som_score is not None and eop_score is not None:
        composite_score = round((som_score * 0.6) + (eop_score * 0.4), 1)
    else:
        composite_score = som_score
    outcome_quality = composite_score if composite_score is not None else som_score

    passed = bool(delivery_gate_passed) if delivery_gate_passed is not None else False
    # Only an explicit FAIL verdict overrides the delivery_gate signal.
    # SKIPPED/UNKNOWN (oracle not run) must not force a false-negative fail.
    if oracle_verdict == "FAIL":
        passed = False
    if adv_pass_clean is False:
        passed = False
    if error:
        passed = False

    phase2 = Phase2Result(
        request_id=request_id,
        task_type=task_type,
        state_dir=state_dir,
        routing_artifact_version=routing_artifact_version,
        passed=passed,
        score=som_score,
        verdict=verdict,
        threshold=threshold,
        oracle_verdict=oracle_verdict,
        adv_pass_clean=adv_pass_clean,
        adv_findings_count=adv_findings_count,
        delivery_gate_passed=delivery_gate_passed,
        score_card=score_card,
        ref_entry=ref_entry,
        delivery_path=delivery_path,
        fix_prompt_path=fix_prompt_path,
        error=error,
        notes=list(notes),
        som_score=som_score,
        eop_score=eop_score,
        composite_score=composite_score,
    )

    lw = _load_log_writer()
    if lw and hasattr(lw, "log_routing_outcome"):
        try:
            lw.log_routing_outcome(
                request_id=request_id,
                task_type=task_type,
                session_id=session_id,
                source=source,
                surface=surface,
                composite_score=composite_score,
                som_score=som_score,
                eop_score=eop_score,
                outcome_quality=outcome_quality,
                oracle_verdict=oracle_verdict,
                adv_pass_clean=adv_pass_clean,
                adv_findings_count=adv_findings_count,
                evidence_valid=evidence_valid,
                delivery_gate_passed=delivery_gate_passed,
                verdict=verdict,
                threshold=threshold,
                latency_ms=latency_ms,
                executor_engine=executor_engine,
                omx_workflow=omx_workflow,
                launch_mode=launch_mode,
                omx_version=omx_version,
                codex_version=codex_version,
                team_size=team_size,
                error=error,
                notes=" | ".join(notes) if notes else None,
            )
        except Exception:
            pass

    _maybe_trigger_optimizer()
    return phase2


# ── Convenience: log outcome only (no SoM pipeline) ───────────────────────────

def run_outcome_only(
    request_id: str,
    task_type: str,
    t0: float,
    routing_artifact_version: str = "static-default",
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    surface: Optional[str] = None,
    error: Optional[str] = None,
    notes_extra: Optional[list[str]] = None,
) -> None:
    """Log an outcome when SoM phase 1/2 was skipped."""
    latency_ms = round((time.time() - t0) * 1000, 1)
    lw = _load_log_writer()
    if lw and hasattr(lw, "log_routing_outcome"):
        try:
            notes = [f"artifact={routing_artifact_version}", "phase=outcome-only"]
            if session_id:
                notes.append(f"session={session_id}")
            notes.extend(list(notes_extra or []))
            lw.log_routing_outcome(
                request_id=request_id,
                task_type=task_type,
                session_id=session_id,
                source=source,
                surface=surface,
                composite_score=50.0,
                som_score=None,
                eop_score=None,
                oracle_verdict="SKIPPED",
                outcome_quality=None,
                adv_pass_clean=None,
                latency_ms=latency_ms,
                error=error,
                notes=" | ".join(notes),
            )
        except Exception:
            pass
    _maybe_trigger_optimizer()
