"""tiny-router integration: per-turn classification for Hermes routing and policy.

Uses the same label schema as https://github.com/UdaraJay/tiny-router when a model
is configured; otherwise falls back to deterministic heuristics.

Inference backends (in order):
1. Subprocess: ``python -m scripts.predict`` from ``repo_root`` with ``model_dir``
   (matches upstream training/inference CLI).
2. Heuristic: keyword- and length-based labels (always available).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# No canonical GitHub "release" tags are currently published for tiny-router.
# Hermes therefore pins to an immutable upstream commit SHA for reproducibility.
DEFAULT_PINNED_COMMIT = "9d6b2a718a205d90ebe85e9a28f9b8a1f20801e4"

# Active router output for the current turn (tool handlers can read policy).
_active_router: ContextVar[Optional["RouterOutput"]] = ContextVar("hermes_tiny_router_active", default=None)


def set_active_router_output(output: Optional["RouterOutput"]):
    return _active_router.set(output)


def get_active_router_output() -> Optional["RouterOutput"]:
    return _active_router.get()


def reset_active_router_output(token) -> None:
    if token is None:
        return
    _active_router.reset(token)


@dataclass
class HeadPrediction:
    label: str
    confidence: float = 0.0


@dataclass
class RouterInput:
    current_text: str
    interaction: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "current_text": self.current_text,
            "interaction": dict(self.interaction or {}),
        }


@dataclass
class RouterOutput:
    relation_to_previous: HeadPrediction
    actionability: HeadPrediction
    retention: HeadPrediction
    urgency: HeadPrediction
    overall_confidence: float = 0.0
    source: str = "heuristic"  # subprocess | heuristic | disabled | error

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "tiny_router": {
                "relation_to_previous": {
                    "label": self.relation_to_previous.label,
                    "confidence": self.relation_to_previous.confidence,
                },
                "actionability": {
                    "label": self.actionability.label,
                    "confidence": self.actionability.confidence,
                },
                "retention": {"label": self.retention.label, "confidence": self.retention.confidence},
                "urgency": {"label": self.urgency.label, "confidence": self.urgency.confidence},
                "overall_confidence": self.overall_confidence,
                "source": self.source,
            }
        }

    @classmethod
    def disabled(cls) -> "RouterOutput":
        return cls(
            relation_to_previous=HeadPrediction("new", 0.0),
            actionability=HeadPrediction("none", 0.0),
            retention=HeadPrediction("ephemeral", 0.0),
            urgency=HeadPrediction("low", 0.0),
            overall_confidence=0.0,
            source="disabled",
        )

    def head_confidence(self, name: str) -> float:
        m = {
            "relation_to_previous": self.relation_to_previous.confidence,
            "actionability": self.actionability.confidence,
            "retention": self.retention.confidence,
            "urgency": self.urgency.confidence,
        }
        return float(m.get(name, 0.0))

    def meets_overall_threshold(self, cfg: Dict[str, Any]) -> bool:
        th = (cfg or {}).get("confidence_thresholds") or {}
        try:
            need = float(th.get("overall", 0.45))
        except (TypeError, ValueError):
            need = 0.45
        return self.overall_confidence >= need

    def should_use_cheap_model_route(self, cfg: Dict[str, Any]) -> bool:
        """When behavior_mode is active, prefer cheap model for low-stakes turns."""
        if self.source == "disabled":
            return False
        th = (cfg or {}).get("confidence_thresholds") or {}
        try:
            o_th = float(th.get("overall", 0.45))
            act_th = float(th.get("actionability", 0.5))
            urg_th = float(th.get("urgency", 0.5))
            ret_th = float(th.get("retention", 0.5))
        except (TypeError, ValueError):
            o_th, act_th, urg_th, ret_th = 0.45, 0.5, 0.5, 0.5

        if self.overall_confidence < o_th:
            return False

        # Strong signals → primary model
        if self.actionability.label == "act" and self.actionability.confidence >= act_th:
            return False
        if self.urgency.label == "high" and self.urgency.confidence >= urg_th:
            return False
        if self.retention.label == "remember" and self.retention.confidence >= ret_th:
            return False
        if self.relation_to_previous.label in ("correction", "cancellation") and self.relation_to_previous.confidence >= 0.5:
            return False

        # Cheap when the model thinks the turn is light-touch
        if self.actionability.label in ("none", "review") and self.urgency.label == "low":
            return True
        return False

    def should_boost_memory_nudge(self, cfg: Dict[str, Any]) -> bool:
        th = (cfg or {}).get("confidence_thresholds") or {}
        try:
            ret_th = float(th.get("retention", 0.5))
        except (TypeError, ValueError):
            ret_th = 0.5
        return self.retention.label == "remember" and self.retention.confidence >= ret_th

    def should_aggressive_memory_flush(self, cfg: Dict[str, Any]) -> bool:
        th = (cfg or {}).get("confidence_thresholds") or {}
        try:
            act_th = float(th.get("actionability", 0.5))
            urg_th = float(th.get("urgency", 0.5))
        except (TypeError, ValueError):
            act_th, urg_th = 0.5, 0.5
        return (
            self.actionability.label == "act"
            and self.actionability.confidence >= act_th
            and self.urgency.label == "high"
            and self.urgency.confidence >= urg_th
        )

    def needs_terminal_review_escalation(self, cfg: Dict[str, Any]) -> bool:
        if not (cfg or {}).get("apply_approval_posture", True):
            return False
        th = (cfg or {}).get("confidence_thresholds") or {}
        try:
            act_th = float(th.get("actionability", 0.5))
        except (TypeError, ValueError):
            act_th = 0.5
        return self.actionability.label == "review" and self.actionability.confidence >= act_th


def _coerce_float(val: Any, default: float) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_head(obj: Any, default_label: str) -> HeadPrediction:
    if isinstance(obj, dict):
        lab = str(obj.get("label") or default_label).strip() or default_label
        conf = _coerce_float(obj.get("confidence"), 0.0)
        return HeadPrediction(lab, max(0.0, min(1.0, conf)))
    return HeadPrediction(default_label, 0.0)


def _normalize_text_input(value: Any) -> str:
    """Best-effort conversion of multimodal/user content into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    chunks.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
                continue
            for key in ("content", "caption", "alt_text", "title"):
                cand = item.get(key)
                if isinstance(cand, str) and cand.strip():
                    chunks.append(cand.strip())
                    break
        return "\n".join(chunks).strip()
    return str(value)


def parse_router_json(data: Dict[str, Any], source: str) -> RouterOutput:
    return RouterOutput(
        relation_to_previous=_parse_head(data.get("relation_to_previous"), "new"),
        actionability=_parse_head(data.get("actionability"), "none"),
        retention=_parse_head(data.get("retention"), "ephemeral"),
        urgency=_parse_head(data.get("urgency"), "low"),
        overall_confidence=_coerce_float(data.get("overall_confidence"), 0.0),
        source=source,
    )


def build_interaction_from_history(
    prior_messages: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build tiny-router interaction object from OpenAI-style history (no current turn)."""
    previous_text = ""
    previous_action = "none"
    previous_outcome = "unknown"
    recency_seconds = -1.0

    if not prior_messages:
        return {
            "previous_text": "",
            "previous_action": previous_action,
            "previous_outcome": previous_outcome,
            "recency_seconds": recency_seconds,
        }

    last_assistant_content = ""
    last_user_content = ""
    for msg in reversed(prior_messages):
        role = msg.get("role")
        if role == "assistant" and not last_assistant_content:
            c = msg.get("content")
            c_text = _normalize_text_input(c)
            if c_text.strip():
                last_assistant_content = c_text.strip()[:2000]
        elif role == "user" and not last_user_content:
            c = msg.get("content")
            c_text = _normalize_text_input(c)
            if c_text.strip():
                last_user_content = c_text.strip()[:2000]
        if last_assistant_content and last_user_content:
            break

    previous_text = last_user_content or last_assistant_content
    # Heuristic: last tool name hints prior "action"
    for msg in reversed(prior_messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tcs = msg.get("tool_calls") or []
            if isinstance(tcs, list) and tcs:
                first = tcs[0]
                if isinstance(first, dict):
                    fn = first.get("function") or {}
                    name = (fn.get("name") or "").strip()
                    if name:
                        previous_action = name.replace("_", " ")[:80]
                        previous_outcome = "success"
                break

    return {
        "previous_text": previous_text,
        "previous_action": previous_action,
        "previous_outcome": previous_outcome,
        "recency_seconds": recency_seconds,
    }


def _heuristic_classify(current_text: str, interaction: Dict[str, Any]) -> RouterOutput:
    text = (current_text or "").strip()
    low = text.lower()
    prev = (interaction.get("previous_text") or "").strip()

    # Relation
    if prev:
        if re.search(r"\b(no|wrong|actually|instead|cancel|undo|stop)\b", low):
            rel = "correction"
            rel_conf = 0.82
        elif re.search(r"\b(yes|ok|thanks|done|perfect|great)\b", low) and len(text) < 80:
            rel = "confirmation"
            rel_conf = 0.75
        elif len(text) < 40 and not re.search(r"[.?]", text):
            rel = "follow_up"
            rel_conf = 0.6
        else:
            rel = "follow_up"
            rel_conf = 0.55
    else:
        rel = "new"
        rel_conf = 0.9

    # Actionability / urgency
    if re.search(r"\b(run|execute|implement|fix|debug|deploy|write|patch|test|build)\b", low):
        act, act_conf = "act", 0.88
    elif "?" in text or re.search(r"\b(what|how|why|explain)\b", low):
        act, act_conf = "review", 0.78
    else:
        act, act_conf = "none", 0.72

    if re.search(r"\b(asap|urgent|immediately|now\b|critical|production down)\b", low):
        urg, urg_conf = "high", 0.9
    elif len(text) > 400 or text.count("\n") > 4:
        urg, urg_conf = "medium", 0.65
    else:
        urg, urg_conf = "low", 0.78

    # Retention
    if re.search(r"\b(remember|always|never forget|preference|my name is|i prefer)\b", low):
        ret, ret_conf = "remember", 0.85
    elif act == "act" or len(text) > 120:
        ret, ret_conf = "useful", 0.7
    else:
        ret, ret_conf = "ephemeral", 0.68

    overall = (rel_conf + act_conf + urg_conf + ret_conf) / 4.0
    return RouterOutput(
        relation_to_previous=HeadPrediction(rel, rel_conf),
        actionability=HeadPrediction(act, act_conf),
        retention=HeadPrediction(ret, ret_conf),
        urgency=HeadPrediction(urg, urg_conf),
        overall_confidence=overall,
        source="heuristic",
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    expected_heads = {"relation_to_previous", "actionability", "retention", "urgency"}

    # Fast path: stdout is a single clean JSON object.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try line-by-line JSON payloads (common when upstream prints logs).
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and expected_heads & set(obj.keys()):
            return obj

    # Fallback: scan for decodable JSON objects and prefer the one that
    # contains expected tiny-router heads; otherwise choose the largest object.
    dec = json.JSONDecoder()
    best_obj: Optional[Dict[str, Any]] = None
    best_len = -1
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if expected_heads & set(obj.keys()):
            return obj
        if end > best_len:
            best_len = end
            best_obj = obj
    return best_obj


def _run_subprocess_predict(cfg: Dict[str, Any], payload: Dict[str, Any]) -> Optional[RouterOutput]:
    repo_root = str(cfg.get("repo_root") or "").strip()
    model_dir = str(cfg.get("model_dir") or "").strip()
    if not repo_root or not model_dir:
        return None
    repo_path = Path(repo_root).expanduser()
    model_path = Path(model_dir).expanduser()
    if not repo_path.is_dir() or not model_path.is_dir():
        return None
    predict_py = repo_path / "scripts" / "predict.py"
    if not predict_py.is_file():
        logger.debug("tiny-router predict script not found at %s", predict_py)
        return None

    timeout = _coerce_float(cfg.get("predict_timeout_seconds"), 30.0)
    timeout = max(5.0, min(120.0, timeout))
    cmd = [
        sys.executable,
        "-m",
        "scripts.predict",
        "--model-dir",
        str(model_path),
        "--input-json",
        json.dumps(payload, ensure_ascii=False),
        "--pretty",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("tiny-router subprocess failed: %s", exc)
        return None

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        logger.warning("tiny-router predict exit %s: %s", proc.returncode, out[:500])
        return None
    data = _extract_json_object(proc.stdout or "")
    if not data:
        data = _extract_json_object(out)
    if not data:
        logger.warning("tiny-router predict returned no JSON")
        return None
    return parse_router_json(data, "subprocess")


def classify_turn(
    tiny_router_cfg: Optional[Dict[str, Any]],
    current_text: Any,
    prior_messages: Optional[List[Dict[str, Any]]],
) -> RouterOutput:
    """Classify the current user turn. Always returns a RouterOutput."""
    cfg = tiny_router_cfg or {}
    if not cfg.get("enabled"):
        return RouterOutput.disabled()
    current_text_norm = _normalize_text_input(current_text)

    interaction = build_interaction_from_history(prior_messages)
    router_input = RouterInput(
        current_text=current_text_norm or "",
        interaction={
            "previous_text": interaction["previous_text"],
            "previous_action": interaction["previous_action"],
            "previous_outcome": interaction["previous_outcome"],
            "recency_seconds": interaction["recency_seconds"],
        },
    )
    payload = router_input.to_payload()

    fb = str(cfg.get("fallback_mode") or "heuristic").strip().lower()
    backend = str(cfg.get("backend") or "subprocess").strip().lower()
    out: Optional[RouterOutput] = None
    if backend == "subprocess":
        try:
            out = _run_subprocess_predict(cfg, payload)
        except Exception as exc:
            logger.debug("tiny-router subprocess error: %s", exc)
            out = None

    if out is None:
        if fb == "none":
            return RouterOutput.disabled()
        return _heuristic_classify(current_text_norm, interaction)
    return out


def router_dict_from_output(output: RouterOutput) -> Dict[str, Any]:
    """Serializable dict for run_conversation / session metadata."""
    return output.to_metadata_dict()


def validate_tiny_router_config(cfg: Optional[Dict[str, Any]]) -> tuple[bool, str]:
    """Validate runtime tiny-router config and return (ok, error_message)."""
    conf = cfg or {}
    if not conf.get("enabled"):
        return True, ""

    mode = str(conf.get("backend") or "subprocess").strip().lower()
    if mode not in {"subprocess", "heuristic"}:
        return (
            False,
            "smart_model_routing.tiny_router.backend must be 'subprocess' or 'heuristic' "
            f"(got: {mode!r})",
        )

    if mode == "heuristic":
        return True, ""

    repo_root = str(conf.get("repo_root") or "").strip()
    model_dir = str(conf.get("model_dir") or "").strip()
    if not repo_root:
        return False, "smart_model_routing.tiny_router.repo_root is required when backend=subprocess"
    if not model_dir:
        return False, "smart_model_routing.tiny_router.model_dir is required when backend=subprocess"

    repo_path = Path(repo_root).expanduser()
    model_path = Path(model_dir).expanduser()
    if not repo_path.is_dir():
        return False, f"smart_model_routing.tiny_router.repo_root does not exist: {repo_root}"
    if not model_path.is_dir():
        return False, f"smart_model_routing.tiny_router.model_dir does not exist: {model_dir}"

    predict_py = repo_path / "scripts" / "predict.py"
    if not predict_py.is_file():
        return False, f"tiny-router predict entrypoint missing: {predict_py}"

    enforce_pin = bool(conf.get("enforce_pinned_commit", True))
    if enforce_pin:
        pinned_commit = str(conf.get("pinned_commit") or DEFAULT_PINNED_COMMIT).strip()
        if not pinned_commit:
            return (
                False,
                "smart_model_routing.tiny_router.pinned_commit must be set when enforce_pinned_commit=true",
            )
        head = ""
        git_err = ""
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode == 0:
                head = (proc.stdout or "").strip().lower()
            else:
                git_err = (proc.stderr or proc.stdout or "").strip()
        except (subprocess.SubprocessError, OSError) as exc:
            git_err = str(exc)

        if not head:
            revision_file = str(conf.get("source_revision_file") or "REVISION").strip() or "REVISION"
            revision_path = Path(revision_file).expanduser()
            if not revision_path.is_absolute():
                revision_path = repo_path / revision_path
            if revision_path.is_file():
                try:
                    head = (revision_path.read_text(encoding="utf-8").strip().splitlines()[0]).lower()
                except Exception:
                    head = ""
        if not head:
            return (
                False,
                "unable to determine tiny-router revision via git or source_revision_file. "
                f"git_error={git_err or 'none'}",
            )
        pin = pinned_commit.lower()
        if not head.startswith(pin):
            return (
                False,
                "tiny-router checkout is not pinned: "
                f"expected {pinned_commit}, got {head}. "
                "Update repo_root checkout or set enforce_pinned_commit=false.",
            )

    return True, ""


def apply_router_thread_local(output: Optional[RouterOutput]) -> None:
    """Set thread-local active router (for tools that cannot use contextvars)."""
    if not hasattr(apply_router_thread_local, "_local"):
        apply_router_thread_local._local = threading.local()  # type: ignore[attr-defined]
    apply_router_thread_local._local.output = output  # type: ignore[attr-defined]


def get_active_router_thread_local() -> Optional[RouterOutput]:
    if not hasattr(apply_router_thread_local, "_local"):
        return None
    return getattr(apply_router_thread_local._local, "output", None)  # type: ignore[attr-defined]


def get_active_router_for_tools() -> Optional[RouterOutput]:
    """Prefer ContextVar; fall back to thread-local (delegate/threads)."""
    ctx = get_active_router_output()
    if ctx is not None:
        return ctx
    return get_active_router_thread_local()
