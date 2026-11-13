"""Intent detectors for acknowledgements and announced next actions."""

from __future__ import annotations

import re
from typing import Any

_ACK_FUTURE_RE = re.compile(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b")
# Anchor Spanish intentions to a clause, excluding negations, questions and lists.
_SPANISH_PLAN_PREFIX = (
    r"(?:^|[.!…—–]\s+|\n\s*)(?:ahora\s+|primero[, :]\s*|luego[, :]\s*)?"
    r"(?:voy a|vamos a|d[eé]jame|d[eé]janos|necesito|tengo que)\s+"
)
_SPANISH_ACTION = (
    r"(?:lanzar|relanzar|iniciar|reiniciar|crear|ejecutar|correr|revisar|comprobar|"
    r"verificar|leer|buscar|inspeccionar|abrir|probar|analizar|arreglar|depurar|"
    r"explorar|escribir|resumir)\b"
)
_SPANISH_ACK_RE = re.compile(_SPANISH_PLAN_PREFIX + _SPANISH_ACTION, re.IGNORECASE)
_SPANISH_PLAN_TAIL_RE = re.compile(
    _SPANISH_PLAN_PREFIX + _SPANISH_ACTION + r"[^.!?¿\n]{0,100}[.:…]?\s*$",
    re.IGNORECASE,
)
_ACK_ACTION_MARKERS = (
    "look into", "look at", "inspect", "scan", "check", "analyz", "review", "explore", "read", "open",
    "run", "test", "fix", "debug", "search", "find", "walkthrough", "report back", "summarize",
)
_ACK_WORKSPACE_MARKERS = (
    "directory", "current directory", "current dir", "cwd", "repo", "repository", "codebase",
    "project", "folder", "filesystem", "file tree", "files", "path",
    "directorio", "carpeta", "repositorio", "proyecto", "archivo", "archivos",
)


def looks_like_codex_intermediate_ack(
    agent, user_message: Any, assistant_content: str, messages: list[dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.
    ``require_workspace=False`` (opt-in for all api_modes) drops the filesystem/repo reference
    requirement; short-content + no-tool-results-this-turn guardrails always apply (every row after this
    turn's own user row: ``messages`` is the whole session, #69778, and a nudge row starts no new turn).
    A response then needs a future ack + action marker, or a bounded pronounless clause + transition cue."""
    from agent.turn_context import reanchor_current_turn_user_idx as turn_user_idx
    if any(isinstance(m, dict) and m.get("role") == "tool" for m in messages[turn_user_idx(messages, user_message) + 1:]):
        return False
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    # Some tool-using providers narrate the next action in terse log style: "Brief written. Creating the
    # session now." Bare gerunds are too ambiguous ("Reading the traceback explains the failure" is a final
    # answer), so this path also requires an explicit transition cue such as "now", "on <ACP agent> via acpx",
    # "actual log", or "then launching". Completed reports and questions stay final.
    action_clause_pattern = re.compile(
        r"(?:^|[.!…—–]\s+|\n+\s*|,\s+then\s+)"
        r"(?:(?P<transition>then)\s+)?"
        r"(?P<action>(?:re)?launching|(?:re)?starting|creating|"
        r"checking|running|writing|opening|reading|inspecting|reviewing|"
        r"testing|debugging|searching|fixing|(?:re)?lanzando|(?:re)?iniciando|creando|"
        r"ejecutando|corriendo|revisando|comprobando|verificando|leyendo|abriendo|"
        r"inspeccionando)\b"
    )
    list_item_prefix_pattern = re.compile(r"(?:^|\n)\s*(?:[-*+]|\d+[.)])\s*$")
    numbered_item_pattern = re.compile(r"(?:^|\s)(\d+)[.)](?=\s)")
    status_number_pattern = re.compile(
        r"\b(?P<label>exit\s+code|attempt|step|retry)\s+"
        r"(?P<number>\d+)[.)](?=\s)"
    )
    question_pattern = re.compile(r"\?(?=\s|$|[\"'’”)\\]])")
    sentence_boundary_pattern = re.compile(
        r"(?:[.!…](?=\s|$)|\?(?=\s|$|[\"'’”)\\]])|\n+)"
    )
    allowed_trailing_action_pattern = re.compile(
        r"\s*(?:will\s+report\s+back|"
        r"then\s+(?:launching|relaunching|starting|restarting|creating)\s+"
        r"(?:it|(?:the\s+)?(?:session|worker|agent|job|process))|"
        r"(?:checking|reading|opening|inspecting)\s+(?:the\s+)?(?:log|output))"
        r"\s*[.!…]?\s*$"
    )
    launch_now_tail_pattern = re.compile(
        r"\s+(?:(?:it|(?:the\s+)?(?:session|worker|agent|job|process)|"
        r"(?:the\s+)?migration\s+file\s+in\s+the\s+repo)\s+)?"
        r"now(?:\s*\([^)]*\))?\s*$"
    )
    run_now_tail_pattern = re.compile(
        r"\s+(?:(?:it|(?:the\s+)?(?:(?:repo\s+)?suite|tests?|job|process|"
        r"server|service))\s+)?now(?:\s*\([^)]*\))?\s*$"
    )
    check_now_tail_pattern = re.compile(
        r"\s+(?:the\s+)?(?:log|output|status|service|session|job)\s+"
        r"now(?:\s*\([^)]*\))?\s*$"
    )
    launch_url_tail_pattern = re.compile(
        r"\s+now\s*[—–-]\s*see\s+https?://\S+"
        r"(?:\s+for\s+progress)?\s*$"
    )
    from providers import list_providers  # "relaunching it on <an ACP agent: its name, alias or binary>"
    provider_names = sorted({n.lower() for p in list_providers() if p.auth_type == "external_process"
                             for n in (p.name, *p.aliases, p.process_command) if n})
    provider_tail_pattern = re.compile(
        r"\s+(?:it|(?:the\s+)?(?:session|worker|agent|job|process))\s+"
        rf"on\s+(?:{'|'.join(map(re.escape, provider_names)) or '(?!)'})(?:\s+via\s+acpx)?\s*$"
    )
    actual_log_tail_pattern = re.compile(r"\s+(?:the\s+)?actual\s+(?:log|output)\s*$")
    health_check_tail_pattern = re.compile(
        r"\s+whether\b.*\b(?:healthy|ready|running|available|reachable|working)\s*$"
    )
    launch_with_pattern = re.compile(
        r"\s+with\s+(?:(?:corrected|updated|new)\s+"
        r"(?:arguments?|args?|options?|flags?|parameters?)|"
        r"globals?\s+before\s+(?:the\s+)?agent\s+name)\s*$"
    )
    spanish_now_tail_pattern = re.compile(
        r"\s+(?:(?:el|la|los|las|un|una)\s+)?"
        r"(?:agente|proceso|trabajo|sesi[oó]n|servicio|servidor|pruebas|tests|"
        r"registro|salida|estado|archivo|repositorio)\s+ahora"
        r"(?:\s*\([^)]*\))?\s*$"
    )
    spanish_actions = {
        "lanzando", "relanzando", "iniciando", "reiniciando", "creando",
        "ejecutando", "corriendo", "revisando", "comprobando", "verificando",
        "leyendo", "abriendo", "inspeccionando",
    }
    has_pronounless_action = False
    step_numbers = {
        status_match.group("number")
        for status_match in status_number_pattern.finditer(assistant_text)
        if status_match.group("label") == "step"
    }
    has_step_sequence = len(step_numbers) >= 2
    numbering_text = status_number_pattern.sub("", assistant_text)
    numbered_markers = {
        match.group(1) for match in numbered_item_pattern.finditer(numbering_text)
    }
    has_numbered_list = has_step_sequence or bool(numbered_markers)
    if not question_pattern.search(assistant_text):
        for action_match in action_clause_pattern.finditer(assistant_text):
            action_prefix = assistant_text[: action_match.start("action")]
            if has_numbered_list or list_item_prefix_pattern.search(action_prefix):
                continue
            raw_clause_tail = assistant_text[action_match.end() :]
            boundary_match = sentence_boundary_pattern.search(raw_clause_tail)
            if boundary_match:
                clause_tail = raw_clause_tail[: boundary_match.start()]
                remaining_text = raw_clause_tail[boundary_match.end() :]
            else:
                clause_tail = raw_clause_tail
                remaining_text = ""
            if remaining_text.strip() and not allowed_trailing_action_pattern.fullmatch(
                remaining_text
            ):
                continue
            action_word = action_match.group("action")
            is_launch_action = action_word in {
                "launching",
                "relaunching",
                "starting",
                "restarting",
                "creating",
            }
            has_transition_cue = bool(
                (action_word in spanish_actions and spanish_now_tail_pattern.fullmatch(clause_tail))
                or (is_launch_action and launch_now_tail_pattern.fullmatch(clause_tail))
                or (action_word == "running" and run_now_tail_pattern.fullmatch(clause_tail))
                or (
                    action_word == "checking"
                    and check_now_tail_pattern.fullmatch(clause_tail)
                )
                or (is_launch_action and launch_url_tail_pattern.fullmatch(clause_tail))
                or (is_launch_action and provider_tail_pattern.fullmatch(clause_tail))
                or (
                    action_word in {"checking", "reading", "opening", "inspecting"}
                    and actual_log_tail_pattern.fullmatch(clause_tail)
                )
                or (
                    action_word == "checking"
                    and health_check_tail_pattern.fullmatch(clause_tail)
                )
                or (is_launch_action and launch_with_pattern.fullmatch(clause_tail))
            )
            if not has_transition_cue:
                continue
            has_pronounless_action = True
            break
    spanish_future = bool(
        _SPANISH_ACK_RE.search(assistant_text)
        and not question_pattern.search(assistant_text)
        and not has_numbered_list
    )
    if not (_ACK_FUTURE_RE.search(assistant_text) or spanish_future or has_pronounless_action):
        return False
    if not has_pronounless_action and not spanish_future and not any(
        marker in assistant_text for marker in _ACK_ACTION_MARKERS
    ):
        return False
    # Opted-in (all-api_mode) path: future-ack + action verb + no prior tool call suffices.
    if not require_workspace:
        return True
    # ``user_message`` may be a multi-part content list (vision via the OpenAI-compat server); a
    # list survives ``or ""`` and ``.strip()`` raises, so flatten first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    return (
        any(marker in user_text for marker in _ACK_WORKSPACE_MARKERS)
        or "~/" in user_text
        or "/" in user_text
        or any(marker in assistant_text for marker in _ACK_WORKSPACE_MARKERS)
    )


# Narrow "trailing continue-intent" detector for the stall guard (agent.stall_guards): only the
# message TAIL announcing a next action, so mid-sentence "I will" never trips it.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$", re.IGNORECASE,
)

# Content longer than this is a substantive reply, not a dangling ack.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 400


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action (stall-guard re-prompt trigger)."""
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    return bool(_TRAILING_CONTINUE_INTENT_RE.search(t[-160:]) or _SPANISH_PLAN_TAIL_RE.search(t))


# Broader tail detector for PROMOTED REASONING only (reasoning-only clean stop with tools offered
# and no tool call). Visible content keeps the narrow ``let me now`` shape above because a real
# reply legitimately says "I'll" mid-text; chain-of-thought that ENDS on a first-person plan
# ("Let me batch the terminal calls and run them in parallel.", "I need to check the log.") is a
# stalled model whose turn would otherwise report "complete" with zero tool calls (#111761).
# Tail-only and anchored on the last sentence, so reasoning that merely mentions a plan before
# stating its answer ("...Let me check. The answer is 42.") still promotes.
# Thai (unsegmented script, so no \b after the trigger, unlike the English group) shares the same
# tail shape: a first-person future-action marker immediately followed by more Thai text, often
# preceded by an em/en dash rather than sentence punctuation (#116495). Trigger glosses, in
# pattern order: "I will give you" / "I will", "next I('ll)" + one of {start,try,check,fix,send,
# do,look}, "please let me" + one of {start,try,check,fix,send,do,look}, "I('ll)" + one of
# {start,try,check,fix,send,do,look,run,fire}.
_PROMOTED_REASONING_PLAN_TAIL_RE = re.compile(
    r"(?:^|[.!?:\u3002\uff01\uff1f\u2014\u2013\n]\s*|\u2026\s*)"
    r"(?:let(?:['\u2019]s| me)\b|i(?:['\u2019]ll| will| need to| should| am going to|['\u2019]m going to)\b"
    r"|next[,:]? i\b|now i(?:['\u2019]ll| will| need to)\b|first[,:]? i(?:['\u2019]ll| will| need to)\b"
    r"|\u0e08\u0e30\u0e43\u0e2b\u0e49\u0e1c\u0e21|\u0e1c\u0e21\u0e08\u0e30"
    r"|\u0e15\u0e48\u0e2d\u0e44\u0e1b(?:\u0e08\u0e30|\u0e1c\u0e21\u0e08\u0e30)"
    r"|\u0e02\u0e2d(?:\u0e40\u0e23\u0e34\u0e48\u0e21|\u0e25\u0e2d\u0e07|\u0e15\u0e23\u0e27\u0e08|\u0e41\u0e01\u0e49|\u0e2a\u0e48\u0e07|\u0e17\u0e33|\u0e14\u0e39)"
    r"|\u0e08\u0e30(?:\u0e40\u0e23\u0e34\u0e48\u0e21|\u0e25\u0e2d\u0e07|\u0e15\u0e23\u0e27\u0e08|\u0e41\u0e01\u0e49|\u0e2a\u0e48\u0e07|\u0e17\u0e33|\u0e14\u0e39|\u0e23\u0e31\u0e19|\u0e22\u0e34\u0e07))"
    r"[^.!?\n\u3002\uff01\uff1f]{0,160}(?:[.:\u2026]+)?\s*$",
    re.IGNORECASE,
)


def promoted_reasoning_announces_action(text: str) -> bool:
    """Whether promoted reasoning ENDS on a first-person plan to act (stall, not an answer).

    No overall length cap: the reasoning block of a stalled model is often 300-1600 chars of
    planning monologue; only the tail decides.
    """
    t = (text or "").strip()
    if not t:
        return False
    return bool(_PROMOTED_REASONING_PLAN_TAIL_RE.search(t[-240:]) or _SPANISH_PLAN_TAIL_RE.search(t))


_INTENT_ACK_ON = {"true", "always", "yes", "on"}
_INTENT_ACK_OFF = {"false", "never", "no", "off"}


def intent_ack_continuation_mode(agent) -> str:
    """Intent-ack continuation mode: ``"off"``, ``"codex_only"`` (workspace acks on codex_responses)
    or ``"all"``. Mirrors ``agent.tool_use_enforcement``: ``"auto"`` -> codex_only; true-ish -> all;
    false-ish -> off; ``list`` -> all when a substring matches the active model name, else off."""
    mode = getattr(agent, "_intent_ack_continuation", "auto")
    if mode is True or (isinstance(mode, str) and mode.lower() in _INTENT_ACK_ON):
        return "all"
    if mode is False or (isinstance(mode, str) and mode.lower() in _INTENT_ACK_OFF):
        return "off"
    if isinstance(mode, list):
        model_lower = (agent.model or "").lower()
        return "all" if any(p.lower() in model_lower for p in mode if isinstance(p, str)) else "off"
    # "auto" or any unrecognised value: historical codex-only behavior.
    return "codex_only" if agent.api_mode == "codex_responses" else "off"

