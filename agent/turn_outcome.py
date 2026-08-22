"""End-of-turn outcome evaluation (Layer 0 of failure tracing).

Runs after a turn finishes and declares whether the WORK held up — the third
failure category Hermes has never had. ``finalize_turn`` already distinguishes
"the machinery broke" (``failed``) from "the loop ended with text"
(``completed``); this module adds "the work didn't hold up", attributed to the
skills that ran.

Mechanism:
  1. Mechanical layer first (Layer 1, ``tools/skill_verify`` + the existing
     per-turn file-mutation state). A verifier FAIL is foreclosed: it is
     recorded for that skill and no aux judgment can overturn it (down-only).
  2. Signal-gated aux call (the "residue" judge): only when a verifier FAILed,
     when a used skill had no verifier (unverified residue — the common case),
     or when configured ``run: always``. The aux prompt is seeded with the
     verdict report so it can't ignore a mechanical fail line.
  3. Attribution is dumb-recorder: mechanical FAILs always land on their skill;
     the eval's extra failure points merge in (union). Down-only
     governs attribution too: once a mechanical FAIL is on the table, the
     eval's ``failure_points`` are ignored so a low-context judge can't pin
     extra blame on an unrelated skill that merely ran unverified, and a
     mechanically-confirmed PASS protects that skill from the judge's contrary
     blame. Symmetrically, a per-skill PASS requires per-skill evidence (a
     mechanical verifier PASS); on a confident eval success a skill that ran
     unverified records a NEUTRAL outcome — a sample that keeps the recovery
     window sliding but never claims success, so incidentally-loaded skills
     can't bank fake passes and wash out their failure history. Environmental
     reads live in the reason string, never in the verdict — the curator
     review is the arbiter, not this recorder.
  4. Best-effort everywhere: any failure here must never break the turn.

Recorder-side rejection (the enumerated-evidence guard): the aux judge blames
via CITATION, not bare names. ``failure_points`` is now
``[{"skill": ..., "evidence": [IDs]}]`` pointing at a numbered evidence
catalog — verifier verdicts, tool errors, and file mutations that happened
THIS turn — and the recorder refuses to write a hard False a citation can't
back. A point citing a mechanical verifier FAIL of the exact skill is hard
(the citation is the evidence, no confidence gate); a point citing only
tool-error/file-mutation evidence, or carrying no citation at all (legacy
bare-name blame), is gated (the skill tie is judge-attributed, so the
confidence floor still applies); a point whose citations are ALL invalid
(fabricated ID, cited PASS, another skill's FAIL) is a soft suspect —
recorded NEUTRAL with the judge's reason preserved, never a hard False. This
closes the "conclusion stored as if it were a fact" gap: nothing downstream
can tell a verifier's True/False from the judge's guess, and a bare-name
guess previously landed with the same sidecar status.

Known limitation (future work, not fixed here): the aux judge sees only the
summarized final response + verdict report — no tool outputs, diffs, or file
state — so its verdict over unverified residue is a thin signal. The
down-only attribution rule above keeps that weakness from leaking into blame;
feeding the judge real evidence (a distilled trace from the turn's tool
results) is the lever to raise the signal's quality.

The seam core exposes is the returned ``TurnOutcome`` (attached to the session
result dict by ``finalize_turn``). The ACSS Hypothesize consumer reads that
seam from the edge — it is not built into this module or the turn loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Union

logger = logging.getLogger(__name__)

_AUX_TASK = "outcome"

# The judge must return ``{"task_succeeded", "confidence", "failure_points",
# "reason"}`` as JSON. Models routinely wrap answers in ```json fences, lead
# with a sentence of prose, or truncate mid-JSON at max_tokens — a bare
# ``json.loads(content)`` would return None for all of these and silently
# record nothing (the eval's verdict is the whole feature). We therefore scan
# for the first BALANCED ``{...}`` object in the response so prose/fences
# around it don't kill the parse. The scan is balance-aware and skips braces
# inside string literals, so trailing prose that re-opens a brace after the
# object — or a second JSON fragment — cannot swallow the verdict the way a
# greedy ``{.*}`` match would. A truncated object (unbalanced braces) is
# rejected, but a valid object embedded anywhere in the text survives.


def _first_balanced_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` span in *text*, or None.

    Walks *text* tracking ``{``/``}`` depth and string literals so the scan
    stops at the object's true closing brace — not the first or last ``}`` in
    the whole response. A span that balances but isn't valid JSON is still
    returned; ``json.loads`` in the caller rejects it and the parse ends.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return text[start : i + 1]
    return None

# The eval-success gate for NEUTRAL outcomes. A neutral (a sample that claims
# neither success nor failure) is only recorded when the eval declares success
# at or above this confidence — a weak eval must not add samples that could
# dilute a genuinely-flagged skill's failure rate into recovery. Real passes
# are never gated here: they require a mechanical verifier PASS, which is
# strong per-skill evidence regardless of the eval.
_PASS_CONFIDENCE_THRESHOLD = 0.6

# Judge-only blame lands a hard False only at or above this confidence. The
# judge attributes from a summarized account (no diffs / tool state), so below
# the floor a blamed skill is recorded NEUTRAL — a suspicion for curator
# review, never a failure that could corrupt the outcome window.
_BLAME_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class TurnOutcome:
    task_succeeded: bool  # did the WORK hold up (not "did the loop end")
    confidence: float  # 0..1
    failure_points: List[str]  # skill names to blame, [] when none attributable
    reason: str  # merged verifier+eval text; feeds curator review / ACSS


def _default_outcome_config() -> Dict[str, Any]:
    """Read ``auxiliary.outcome`` from config. {} when unavailable."""
    try:
        from agent.auxiliary_client import _get_auxiliary_task_config

        cfg = _get_auxiliary_task_config(_AUX_TASK)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _resolve_skill_dirs(
    skills_used_this_turn: Union[Iterable[str], Mapping[str, Path]]
) -> List[tuple[str, Path]]:
    """Normalize used skills to ``(name, skill_dir)`` pairs.

    Accepts either a mapping (name → dir, from the turn accumulator) or a
    plain iterable of names (resolved via the skill_usage index).
    """
    if isinstance(skills_used_this_turn, Mapping):
        return [(str(name), Path(d)) for name, d in skills_used_this_turn.items()]
    from tools.skill_usage import _find_skill_dir

    pairs: List[tuple[str, Path]] = []
    for name in skills_used_this_turn:
        d = _find_skill_dir(str(name))
        if d is not None:
            pairs.append((str(name), d))
    return pairs


def _run_skill_verifier(
    skill_name: str, skill_dir: Path, task_cwd: Path
) -> tuple[str, str]:
    """Run one skill's verifier. Returns (verdict, reason); verdict is one of
    ``pass`` | ``fail`` | ``skip``. ``skip`` covers not-opted-in, no verify
    block, not curation-eligible, applicability-gated-out, or a broken check.
    """
    try:
        from tools.skill_verify import run_verification

        outcome = run_verification(skill_name, skill_dir, task_cwd)
    except Exception as e:
        logger.debug("turn_outcome verifier error for %s: %s", skill_name, e, exc_info=True)
        return ("skip", "")
    if outcome is None:
        return ("skip", "")
    return ("pass" if outcome.success else "fail", outcome.reason or "")


def _build_prompt(
    user_message: Optional[str],
    final_response: Optional[str],
    verdict_report: str,
    file_previews: str,
    tool_error_count: int,
    evidence_catalog: str = "",
) -> str:
    lines = [
        "Evaluate whether the task Hermes just completed actually held up.",
        "The work can look finished while being semantically wrong. Judge the WORK,",
        "not whether the loop ended, and only blame skills you can justify.",
        "",
        f"Task: {user_message or '(none)'}",
        f"Final response: {(final_response or '').strip()[:2000]}",
        f"Tool-call errors this turn: {tool_error_count}",
        "Per-skill mechanical verifier verdicts (pass/fail/skip):",
        verdict_report or "  (none)",
    ]
    if file_previews:
        lines.append(f"Failed file mutations: {file_previews}")
    if evidence_catalog:
        lines.append(
            "Evidence catalog (cite by ID — the ONLY support your blame may use):"
        )
        lines.append(evidence_catalog)
    lines.append(
        'Reply with strict JSON: {"task_succeeded": bool, "confidence": 0-1, '
        '"failure_points": [{"skill": "<name>", "evidence": [<IDs>]}], '
        '"reason": "short explanation"}. '
        "For every failure point, cite at least one ID from the evidence catalog "
        "that supports it. If a skill failed but you cannot cite catalog evidence "
        'for it, omit it from "failure_points" and explain in "reason" instead.'
    )
    return "\n".join(lines)


def _parse_judge_json(content: str) -> Optional[Dict[str, Any]]:
    """Tolerantly parse the judge's JSON verdict from raw model output.

    Tries strict parsing first, then falls back to grabbing the first
    ``{...}`` object in the text. Handles the three real-world shapes that
    break a bare ``json.loads``:

      - fenced: `````json\n{"task_succeeded": ...}\n`````
      - prose-wrapped: ``"The task failed. {"task_succeeded": false, ...}"``
      - exact: ``{"task_succeeded": ...}``

    Returns None when nothing parseable survives (truncated/unbalanced object,
    prose with no JSON). Never raises.
    """
    if not content or not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    # Fast path: the whole response is the JSON object.
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Tolerant path: strip fences/prose and parse the first balanced {…} object.
    span = _first_balanced_json_object(text)
    if span is None:
        return None
    try:
        parsed = json.loads(span)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


# ── Enumerated evidence (Layer 0 "conclusion vs. fact" guard) ───────────────
#
# The judge previously returned ``failure_points`` as bare skill names, which
# the recorder wrote with the same sidecar status as a mechanical verifier's
# True/False — a conclusion stored as if it were a fact, with nothing
# downstream able to tell them apart. This section turns blame into a
# CITATION: ``failure_points`` becomes ``[{"skill": ..., "evidence": [IDs]}]``
# pointing at a numbered catalog of THIS turn's evidence (verifier verdicts,
# tool errors, file mutations), and the recorder refuses to write a False that
# can't be backed by a real, matching item.
#
# Validation strength per evidence kind:
#   - verifier: the strongest check — the cited item must be a FAIL of the
#     EXACT blamed skill. Citing a PASS (or another skill's FAIL) invalidates
#     the whole point.
#   - tool_error / file_mutation: existence is the check. The recorder cannot
#     mechanically tie an error to a skill, so a citation here is judge-
#     attributed evidence and still passes through the confidence gate below —
#     the gate is the remaining guarantee for these kinds.
#   - no citation at all (legacy bare-name blame): gated — stays a candidate
#     for a hard False but must clear the confidence gate below (the skill tie
#     is judge-attributed and the gate is the remaining guarantee).
#   - only invalid citations (fabricated ID, cited PASS, another skill's FAIL):
#     the point is a suspicion, recorded NEUTRAL with the judge's reason
#     preserved, never a hard False.

# Cap on catalog size — bounds the judge's prompt cost (mirrors the PR's
# ``total_verify_budget_seconds`` / ``max_tokens`` budget philosophy).
_MAX_EVIDENCE_ITEMS = 20


def _build_evidence_catalog(
    verdicts: Mapping[str, tuple[str, str]],
    tool_errors: Iterable[Mapping[str, Any]],
    file_mutations: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Assemble the numbered evidence catalog for this turn.

    Each entry is ``{"eid": int, "kind": str, "skill": str, "verdict":
    Optional[bool], "text": str}``. Verifier verdicts appear only for
    ``pass``/``fail`` (a ``skip`` is "no judgment" and can't be cited as
    evidence of anything). Tool errors and file mutations carry no skill — the
    judge attributes them. Best-effort: any malformed input is dropped.
    """
    catalog: list[Dict[str, Any]] = []

    def _add(
        kind: str, skill: str, subject: str, verdict: Optional[bool], text: str
    ) -> None:
        if len(catalog) >= _MAX_EVIDENCE_ITEMS:
            return
        catalog.append(
            {
                "eid": len(catalog) + 1,
                "kind": kind,
                "skill": skill,
                "subject": subject,
                "verdict": verdict,
                "text": text[:400],
            }
        )

    for name, (verdict, reason) in (verdicts or {}).items():
        if verdict not in ("pass", "fail"):
            continue
        verdict_bool = verdict == "pass"
        tail = f" — {reason}" if reason else ""
        _add("verifier", str(name), str(name), verdict_bool, f"{verdict.upper()}{tail}")

    for err in tool_errors or ():
        if not isinstance(err, dict):
            continue
        tool = str(err.get("tool") or "?")
        error = str(err.get("error") or "").strip()
        if not error:
            continue
        _add("tool_error", "", tool, None, error)

    for path, meta in (file_mutations or {}).items():
        preview = ""
        if isinstance(meta, dict):
            preview = str(meta.get("error_preview") or "")
        _add("file_mutation", "", str(path), None, preview)

    return catalog


def _render_evidence_catalog(catalog: Sequence[Mapping[str, Any]]) -> str:
    """Render the catalog for the judge prompt.

    One line per item, prefixed with its ID so the judge can cite it. Verifier
    items show the skill + verdict; tool errors / file mutations show their
    subject (tool name / path) + the error text. Never duplicates the subject
    (it lives in its own field, not embedded in the text).
    """
    if not catalog:
        return "  (none)"
    lines = []
    for e in catalog:
        if e["kind"] == "verifier":
            subject = e["skill"]
        else:
            subject = e["subject"]
        lines.append(f"  [{e['eid']}] {e['kind']}({subject}) {e['text']}")
    return "\n".join(lines)


def _normalize_failure_points(fp: Any) -> List[Dict[str, Any]]:
    """Coerce the judge's ``failure_points`` into a uniform list of dicts.

    Accepts both the new citation schema (``[{"skill": ..., "evidence":
    [...]}]``) and the legacy bare-name schema (``["skill-a", ...]``), plus
    ragged variants (dicts missing keys, names wrapped in lists). Never raises.
    """
    if not isinstance(fp, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for p in fp:
        if isinstance(p, dict):
            skill = p.get("skill")
            evidence = p.get("evidence")
            if isinstance(evidence, (list, tuple)):
                cleaned: List[int] = []
                for e in evidence:
                    if isinstance(e, int):
                        cleaned.append(e)
                    elif isinstance(e, str) and e.strip().isdigit():
                        # Judges frequently stringify IDs ("1", " 2 ").
                        cleaned.append(int(e.strip()))
                evidence = cleaned
            else:
                evidence = []
            if not isinstance(skill, str) or not skill:
                continue
            out.append({"skill": skill, "evidence": evidence})
        elif isinstance(p, str) and p.strip():
            # Legacy schema: a bare name with no citation. Records as a
            # suspicion (NEUTRAL) unless the confidence gate lets it through —
            # see _validate_eval_blame.
            out.append({"skill": p.strip(), "evidence": []})
    return out


def _validate_eval_blame(
    points: List[Dict[str, Any]],
    catalog: Sequence[Mapping[str, Any]],
    used_names: Set[str],
) -> tuple[List[str], List[str], List[str]]:
    """Split eval failure points into (hard, gated, soft) skill names.

    A point hard-blames its skill only when it carries a valid citation to a
    mechanical verifier FAIL of that EXACT skill — the strongest evidence, so
    it lands as a hard False regardless of the judge's confidence. A point
    citing only tool-error / file-mutation evidence, or carrying NO citation
    (legacy bare-name blame), is ``gated``: it stays a candidate for a hard
    False but must clear the confidence gate below — the citation is
    existence-only (or absent), so the skill tie is judge-attributed and the
    gate is the remaining guarantee. A point whose citations are ALL invalid
    (fabricated ID, cited PASS, another skill's FAIL) is a soft suspect —
    recorded NEUTRAL, never a hard False.
    """
    by_id = {int(e["eid"]): e for e in catalog}
    hard: List[str] = []
    gated: List[str] = []
    soft: List[str] = []
    for p in points:
        skill = p.get("skill")
        if not skill or skill not in used_names:
            continue
        evidence = p.get("evidence") or []
        saw_toolish = False
        saw_verifier_fail = False
        for eid in evidence:
            item = by_id.get(int(eid)) if isinstance(eid, int) else None
            if item is None:
                continue  # fabricated/unknown ID — never evidence
            if item["kind"] == "verifier":
                # The citation must be a mechanical FAIL of THIS skill. Citing
                # a PASS (or another skill's FAIL) is not evidence of failure.
                if item["skill"] == skill and item["verdict"] is False:
                    saw_verifier_fail = True
            else:
                # tool_error / file_mutation: existence is the check.
                saw_toolish = True
        if saw_verifier_fail:
            hard.append(skill)
        elif saw_toolish or not evidence:
            # tool/file-cited OR uncited (legacy) blame → confidence-gated.
            gated.append(skill)
        else:
            soft.append(skill)
    return (
        list(dict.fromkeys(hard)),
        list(dict.fromkeys(gated)),
        list(dict.fromkeys(soft)),
    )


def _default_aux_eval(prompt: str) -> Optional[Dict[str, Any]]:
    """Real aux-client path. Best-effort: no client / any error → None.

    Routed through ``call_llm`` (not a raw ``client.chat.completions.create``)
    so the task's configured ``auxiliary.outcome.timeout``, ``extra_body``,
    and ``reasoning_effort`` actually reach the wire, and so ``max_tokens`` is
    translated to ``max_completion_tokens`` where the provider requires it.
    The token budget is config-driven (``auxiliary.outcome.max_tokens``,
    default 1000): a fixed 200 was far too small for reasoning models, which
    burn their whole budget on ``reasoning`` and return empty ``content`` —
    silently recording nothing (the eval's verdict is the whole feature).
    """
    try:
        from agent.auxiliary_client import call_llm

        config = _default_outcome_config()
        max_tokens = int(config.get("max_tokens") or 1000)
        resp = call_llm(
            task=_AUX_TASK,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        content = ""
        for choice in getattr(resp, "choices", None) or ():
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", "") or ""
            if content:
                break
        return _parse_judge_json(content)
    except Exception as e:
        logger.debug("outcome aux eval failed: %s", e, exc_info=True)
        return None


def _record(
    skill_name: str,
    success: Optional[bool],
    reason: str = "",
    skill_dir: Optional[Path] = None,
) -> None:
    """Best-effort write to the usage sidecar. Never raises into the turn.

    ``success`` is three-state: True (mechanical PASS), False (mechanical FAIL
    or eval-attributed blame), None (neutral — the turn succeeded but this
    skill ran unverified). See ``tools.skill_usage.bump_outcome``.

    Gated on curation eligibility, mirroring the mechanical verifier path
    (``tools.skill_verify.run_verification`` refuses non-eligible skills before
    it can produce a FAIL). Without this gate, an eval-attributed failure point
    — the judge may name ANY skill, including a bundled/hub-installed one that
    also ran — would flip ``needs_review`` on a skill the curator never
    surfaces, and its reason would sit in the sidecar permanently. Eligible
    skills are the only ones whose outcomes feed curator review.
    """
    try:
        from tools.skill_usage import bump_outcome, is_curation_eligible

        if not is_curation_eligible(skill_name, skill_dir):
            logger.debug(
                "turn_outcome: skipping outcome record for %s — not "
                "curation-eligible",
                skill_name,
            )
            return
        bump_outcome(skill_name, success, reason=reason or None)
    except Exception as e:
        logger.debug("turn_outcome: failed to record %s: %s", skill_name, e, exc_info=True)


def _resolve_task_cwd(task_cwd: Optional[Union[str, Path]]) -> Path:
    """Resolve the directory verifiers run against.

    ``task_cwd`` wins when the caller supplies it. Otherwise use the agent's
    canonical working directory — ``agent.runtime_cwd.resolve_agent_cwd``, the
    same resolver the system prompt and file tools use (honors the per-session
    pin set by gateway/ACP sessions and the ``TERMINAL_CWD`` bridge). Falling
    back to ``Path.cwd()`` would run gateway/server verifiers against the
    backend process's cwd, not the session's worktree. Process cwd is used only
    as a last resort.
    """
    if task_cwd is not None:
        return Path(task_cwd)
    try:
        from agent.runtime_cwd import resolve_agent_cwd

        return resolve_agent_cwd()
    except Exception:
        return Path.cwd()


def evaluate_turn_outcome(
    *,
    skills_used_this_turn: Union[Iterable[str], Mapping[str, Path]] = (),
    task_cwd: Optional[Union[str, Path]] = None,
    final_response: Optional[str] = None,
    user_message: Optional[str] = None,
    failed: bool = False,
    interrupted: bool = False,
    exit_reason: Optional[str] = None,
    file_mutation_state: Optional[Mapping[str, Any]] = None,
    tool_error_count: int = 0,
    tool_error_evidence: Optional[Iterable[Mapping[str, Any]]] = None,
    outcome_config: Optional[Mapping[str, Any]] = None,
    _aux_eval: Optional[Callable[[str], Optional[Mapping[str, Any]]]] = None,
) -> Optional[TurnOutcome]:
    """Evaluate whether the finished turn's work held up.

    Returns None when there is nothing to record: feature disabled, the turn
    was interrupted, no signal triggered the eval, or no judgment could be
    produced. Never raises — this runs at end-of-turn and must not break it.
    """
    try:
        cfg = (
            dict(outcome_config)
            if outcome_config is not None
            else _default_outcome_config()
        )
        if not cfg.get("enabled"):
            return None

        if interrupted:
            return None  # user-stopped turns are not work failures
        if failed:
            # Infra-failed turn: an outcome, but no skill is to blame.
            return TurnOutcome(
                task_succeeded=False,
                confidence=1.0,
                failure_points=[],
                reason=f"infra failure: {exit_reason or 'unknown'}",
            )

        cwd = _resolve_task_cwd(task_cwd)
        run_mode = str(cfg.get("run") or "auto")

        # ── Mechanical layer first ──────────────────────────────────────────
        # Aggregate ceiling across ALL verifier subprocesses this turn: each
        # verifier has its own timeout_seconds (default 30, capped at 300) and
        # an applicability probe adds up to 10 more, so without a turn-level
        # budget a skill-heavy turn could hold finalization for minutes. Once
        # the budget is exhausted, remaining skills record as ``skip`` so the
        # downstream has_residue handling stays correct.
        import time as _time

        total_budget = float(cfg.get("total_verify_budget_seconds") or 60)
        _started = _time.monotonic()
        skill_dirs = _resolve_skill_dirs(skills_used_this_turn)
        verdicts = {}
        for name, d in skill_dirs:
            if _time.monotonic() - _started >= total_budget:
                logger.debug(
                    "turn_outcome: verify budget exhausted — skipping %s", name
                )
                verdicts[name] = ("skip", "")
                continue
            verdicts[name] = _run_skill_verifier(name, d, cwd)
        fail_verdicts = [(n, r) for n, (v, r) in verdicts.items() if v == "fail"]
        pass_names = [n for n, (v, r) in verdicts.items() if v == "pass"]
        skip_names = {n for n, (v, r) in verdicts.items() if v == "skip"}

        dir_by_name = dict(skill_dirs)

        fm = file_mutation_state or {}
        has_mechanical_fail = bool(fail_verdicts) or bool(fm)
        has_residue = bool(skip_names)

        should_eval = run_mode == "always" or has_mechanical_fail or has_residue
        if not should_eval:
            # All used skills verified clean and nothing failed — no residue to
            # judge, so no global verdict is produced. Still record the clean
            # verifier passes: a mechanical PASS is per-skill evidence and must
            # not be lost just because nothing else triggered the judge
            # (previously these passes were discarded entirely).
            for n in pass_names:
                _record(n, True, skill_dir=dir_by_name.get(n))
            return None

        # ── Signal-gated aux judgment ───────────────────────────────────────
        # Numbered evidence catalog: verifier verdicts + tool errors + file
        # mutations. The judge may only blame via citations into this catalog;
        # the recorder validates every citation mechanically before writing a
        # False (see _validate_eval_blame).
        evidence_catalog = _build_evidence_catalog(
            verdicts, tool_error_evidence, fm
        )
        verdict_report = "\n".join(
            f"  - {n}: {v}{f' ({r})' if r else ''}" for n, (v, r) in verdicts.items()
        )
        file_previews = "; ".join(
            f"{k}: {str(v.get('error_preview') if isinstance(v, dict) else '')[:200]}"
            for k, v in fm.items()
        )
        prompt = _build_prompt(
            user_message,
            final_response,
            verdict_report,
            file_previews,
            tool_error_count,
            evidence_catalog=_render_evidence_catalog(evidence_catalog),
        )
        aux_data: Optional[Mapping[str, Any]] = None
        if _aux_eval is not None:
            try:
                aux_data = _aux_eval(prompt)
            except Exception as e:
                logger.debug("turn_outcome: injected aux eval raised: %s", e, exc_info=True)
                aux_data = None
        else:
            aux_data = _default_aux_eval(prompt)

        eval_succeeded: Optional[bool] = None
        eval_confidence: Optional[float] = None
        eval_points: List[Dict[str, Any]] = []
        eval_reason = ""
        if isinstance(aux_data, dict):
            if "task_succeeded" in aux_data:
                v = aux_data.get("task_succeeded")
                if isinstance(v, bool):
                    eval_succeeded = v
                elif v in ("true", "false"):
                    # The judge frequently stringifies booleans ("false") —
                    # bool("false") is True, so coerce strings explicitly.
                    eval_succeeded = v == "true"
            conf = aux_data.get("confidence")
            if isinstance(conf, (int, float)):
                eval_confidence = float(min(max(conf, 0.0), 1.0))
            eval_points = _normalize_failure_points(aux_data.get("failure_points"))
            eval_reason = str(aux_data.get("reason") or "")

        # ── Verdict resolution ──────────────────────────────────────────────
        if has_mechanical_fail:
            # Down-only: a mechanical FAIL is foreclosed regardless of the eval.
            task_succeeded = False
            confidence = 1.0
            eval_points = []  # down-only covers attribution too: a mechanical FAIL
                              # already explains the turn on its own evidence — don't
                              # let a low-context judge pin extra blame on some other
                              # skill this turn just because it also ran and nothing
                              # mechanically checked it.
        elif eval_succeeded is not None:
            task_succeeded = eval_succeeded
            confidence = eval_confidence if eval_confidence is not None else 0.5
        else:
            # Only unverified residue and no aux judgment available — nothing
            # to record.
            return None

        # ── Attribution (dumb recorder) + persistence ───────────────────────
        mechanical_points = [n for n, _ in fail_verdicts]
        fail_reasons = {n: r for n, r in fail_verdicts}
        # Eval-attributed points are intersected with the skills actually used
        # this turn. The judge has only the prompt's summary of the turn and can
        # hallucinate a skill name ("summarization", ...) that never ran — the
        # recorder must not pin blame (or flip needs_review) on a skill the turn
        # never touched. Mechanical points are already used-skills-only.
        used_names = {name for name, _d in skill_dirs}
        # Recorder-side rejection (enumerated-evidence guard): the judge may
        # only blame via citations into this turn's evidence catalog. A point
        # citing a mechanical verifier FAIL of the same skill is hard —
        # the citation IS the evidence, so it lands regardless of the judge's
        # confidence. A point citing only tool-error / file-mutation evidence,
        # or carrying no citation (legacy bare name), is gated: the tie to a
        # skill is judge-attributed, so the confidence gate below is the
        # remaining guarantee. A point whose citations are ALL invalid
        # (fabricated ID, cited PASS, another skill's FAIL) is a soft suspect —
        # recorded NEUTRAL, never a hard False.
        hard_blamed, gated_blamed, soft_blamed = _validate_eval_blame(
            eval_points, evidence_catalog, used_names
        )

        # Confidence gate on judge-only blame. The judge attributes from a
        # summarized account (no diffs / tool state), so a low-confidence blame
        # on a skill with no mechanical result is a suspicion, not evidence:
        # recording it as a hard False would let a coincidental name corrupt
        # that skill's outcome history. Below the floor, judge blame is
        # downgraded — an unverified suspect is recorded NEUTRAL, a
        # mechanically-passed suspect keeps its PASS — with the judge's reason
        # preserved either way so curator review can still weigh it. Mechanical
        # FAILs stay down-only: this gate never softens a verifier FAIL (and
        # never runs when one exists — eval_points is cleared above).
        if not has_mechanical_fail and confidence < _BLAME_CONFIDENCE_THRESHOLD:
            soft_blamed = list(dict.fromkeys(soft_blamed + gated_blamed))
            gated_blamed = []
        eval_blamed = list(dict.fromkeys(hard_blamed + gated_blamed))

        failure_points = list(dict.fromkeys(mechanical_points + eval_blamed))
        _blamed_set = set(failure_points) | set(soft_blamed)

        # An eval-blamed skill that ALSO mechanically PASSed gets the fail, not
        # the pass: a verifier checks one narrow thing (e.g. a commit-message
        # prefix), and the judge's semantic read (e.g. "the wrong change was
        # committed") can be right even when the narrow check passed. Never
        # double-record one skill twice in the same turn. Downgraded (soft)
        # judge blame suppresses the pass the same way — the suspect gets its
        # own record in the soft-blame loop below.
        _suppressed_pass = set(eval_blamed) | set(soft_blamed)
        effective_pass = [n for n in pass_names if n not in _suppressed_pass]
        _pass_set = set(effective_pass)

        for s in failure_points:
            # Mechanical fails carry their verifier's reason; eval-attributed
            # points carry the eval's merged reason text. Pass the skill dir so
            # the eligibility gate resolves external/bundled skills correctly.
            _record(
                s,
                False,
                fail_reasons.get(s) or eval_reason,
                skill_dir=dir_by_name.get(s),
            )
        # A per-skill PASS requires per-skill evidence: only mechanical verifier
        # PASSes get one (recorded here so an eval blame on the same skill can
        # suppress them). Previously the eval's global success minted a pass for
        # every used skill — fake success that let incidentally-loaded skills
        # wash out their failure history.
        for n in effective_pass:
            _record(n, True, skill_dir=dir_by_name.get(n))

        # Downgraded judge blame: an unverified suspect is NEUTRAL — a sample
        # that never claims pass or fail — while a mechanically-passed suspect
        # keeps its PASS (a verifier is per-skill evidence). The judge's reason
        # rides along in both cases so curator review can weigh the suspicion
        # without the sidecar claiming the skill failed.
        for s in soft_blamed:
            _record(
                s,
                True if s in pass_names else None,
                eval_reason,
                skill_dir=dir_by_name.get(s),
            )

        # On a confident eval success, skills that merely ran unverified get a
        # NEUTRAL outcome: a sample that keeps the recovery window sliding but
        # never claims success (see tools.skill_usage.bump_outcome). Skills that
        # mechanically passed or were blamed already have an outcome this turn.
        if task_succeeded and eval_succeeded is True:
            if confidence >= _PASS_CONFIDENCE_THRESHOLD:
                for name in used_names:
                    if name not in _pass_set and name not in _blamed_set:
                        _record(name, None, skill_dir=dir_by_name.get(name))

        # ── Reason corpus ───────────────────────────────────────────────────
        parts = []
        for n, r in fail_verdicts:
            parts.append(f"verifier ({n}): {r or 'failed'}")
        if fm:
            parts.append(f"file-mutation: {file_previews}")
        if eval_reason:
            parts.append(eval_reason)
        reason = "; ".join(parts)
        if not reason:
            reason = "task did not hold up" if not task_succeeded else "ok"

        return TurnOutcome(
            task_succeeded=task_succeeded,
            confidence=confidence,
            failure_points=failure_points,
            reason=reason,
        )
    except Exception as e:
        logger.debug("evaluate_turn_outcome failed: %s", e, exc_info=True)
        return None
