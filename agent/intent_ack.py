"""Clarification-aware helpers for intent-acknowledgment recovery.

Holds the detector (``looks_like_codex_intermediate_ack``, re-exported by
``agent_runtime_helpers``) and the helpers that decide when an answered ``clarify`` turn
is still unstarted work and when background work already owns the request."""

import json
import re
from typing import Any, Dict, List, Optional


_ACK_LEAD = (
    r"(?:^|[.!]\s+)(?:(?:sure|understood|okay|ok)[,!.—\s]+)?"
    r"(?:i['’]ll|i\s+will|i(?:['’]m|\s+am)\s+going\s+to|let\s+me)\s+"
    r"(?:(?:first|now)\s+)?(?:start\s+by\s+)?"
)
_ACK_TERMINAL_RE = re.compile(
    r"\b(?:let me know|if you|would you|feel free|happy to help|"
    r"approval|permission|confirmation|credentials?|password|token|"
    r"(?:once|when|after|until) you|go.ahead|"
    r"wait|waiting|awaiting|blocked|unable|cannot|can['’]t|won['’]t|"
    r"never|do not|don['’]t|cancel(?:led|ed)?|stop|hold off|"
    r"done|finished|complete(?:d)?|already|passed|next week|later|"
    r"background|still running|in flight)\b"
)
_EDIT_REQUEST_RE = re.compile(
    r"^(?:please\s+)?(?:(?:proceed|continue)\s+with\s+|go ahead and\s+)?"
    r"(?:implement(?:ing)?|fix(?:ing)?|chang(?:e|ing)|updat(?:e|ing)|"
    r"edit(?:ing)?|build(?:ing)?|add(?:ing)?|remov(?:e|ing))\b"
)
_RESUME_REQUEST_RE = re.compile(r"\b(?:proceed|continue|go ahead)[.!\s]*$")
_REPORTED_REQUEST_RE = re.compile(r"\b(?:said|says|told|reported|quote|quoted|example)\b")
_TASK_COMPLETION_RE = re.compile(r"\b(?:done|finished|complete(?:d)?|no (?:changes|work) remain)\b")
_CONTRASTIVE_ANSWER_RE = re.compile(r"^no,\s+[^.!?;]+,\s+not\s+[^.!?;]+[.!]?$")
_CLARIFIED_ACTION_RE = re.compile(
    _ACK_LEAD + r"(?:implement|change|update|edit|add|remove|show|hide|keep)\b"
)
_DECLINED_RE = re.compile(
    r"(?:^no\b|\b(?:cancel(?:led|ed)?|abort|stop|wait|declin(?:e|ed)|deny|skip|hold off|not yet|"
    r"approval|permission|confirmation|credentials?|password|nothing|not now|not to|"
    r"do not|don['’]t|did not provide a response)\b)"
)
# A clarification answer that narrows the request to a non-execution deliverable
# ("Design only, please", "just a plan for now", "no code") is not a decline, but it
# does not authorize the edit either. An answer that still names an execution verb
# ("implement the design only, not the tests") keeps the request.
_DELIVERABLE = (
    r"(?:design|plan|proposal|outline|spec(?:ification)?|write-?up|summary|explanation|"
    r"analysis|recommendation|advice|discussion|review)"
)
_SCOPE_ONLY_RE = re.compile(
    r"\b(?:" + _DELIVERABLE + r"s?\s+only|"
    r"(?:only|just)\s+(?:an?\s+|the\s+)?" + _DELIVERABLE + r"s?\b|"
    r"(?:only|just)\s+(?:design|plan|outline|explain|describe|discuss|summari[sz]e|document|analy[sz]e)\b|"
    r"no\s+(?:code|coding|implementation|edits?|changes?)|"
    r"without\s+(?:implementing|changing|editing|coding|touching))\b"
)
_EXECUTION_VERB_RE = re.compile(
    r"\b(?:implement|build|change|update|edit|fix|add|remove|refactor|apply|write the code|code it)\b"
)


def _scope_narrowed(text: str) -> bool:
    """The answer asks for a deliverable other than the work: no authorization to act."""
    return bool(_SCOPE_ONLY_RE.search(text)) and not _EXECUTION_VERB_RE.search(text)


def _answered_clarification(content: Any) -> bool:
    """True only for a submitted form whose every question has a usable answer.

    Mirrors ``tools.clarify_tool._result``: ``{"responses": [{"status",
    "user_response"}], "outcome"}``. Cancelled, timed-out, undelivered, partial and
    skipped forms never prove the user authorized the work; nor does an answer that
    declines it or narrows it to a design, plan or explanation (#69779, @kvnloo).
    """
    try:
        result = json.loads(content)
    except (TypeError, ValueError):
        return False
    if not isinstance(result, dict) or result.get("outcome") != "submitted":
        return False
    responses = result.get("responses")
    if not isinstance(responses, list) or not responses:
        return False
    for response in responses:
        if not isinstance(response, dict) or response.get("status") != "answered":
            return False
        answer = response.get("user_response")
        answers = answer if isinstance(answer, list) else [answer]
        if not answers:
            return False
        for answer in answers:
            if not isinstance(answer, str) or not answer.strip():
                return False
            text = answer.strip().lower()
            # A contrastive answer corrects the question's premise; a bare No
            # still declines. Explicit stop/wait language wins in either form.
            if _CONTRASTIVE_ANSWER_RE.fullmatch(text):
                text = text[3:].lstrip()
            if _DECLINED_RE.search(text) or _scope_narrowed(text):
                return False
    return True


def _unquoted_prose(text: str) -> str:
    return re.sub(
        r'''`[^`]*`|"[^"]*"|“[^”]*”|(?<!\w)'[^\n]*?'(?!\w)|‘[^\n]*?’(?!\w)|(?m:^>.*$)''',
        "", text,
    )


def _clarification_task_request(user_text: str, messages: list[dict[str, Any]]) -> str:
    """Resolve a direct request or its adjacent, still-unstarted continuation.

    This selects context for a bounded re-query, not permission to execute. Never
    mine tool output or assistant prose for authority, or jump over performed work
    to resurrect a stale request. A quoted/reported 'proceed' is not a continuation.
    """
    from agent.conversation_compression import _is_real_user_message, _message_text

    text = _unquoted_prose(user_text).strip()
    if "?" in text or _DECLINED_RE.search(text) or _REPORTED_REQUEST_RE.search(text):
        return ""
    if _EDIT_REQUEST_RE.search(text):
        return text
    if not _RESUME_REQUEST_RE.search(text):
        return ""
    latest = next((i for i in range(len(messages) - 1, -1, -1)
                   if isinstance(messages[i], dict) and messages[i].get("role") == "user"), 0)
    for row in reversed(messages[:latest]):
        if not isinstance(row, dict):
            continue
        if row.get("role") == "tool" or row.get("tool_calls"):
            return ""
        if row.get("role") == "assistant" and _TASK_COMPLETION_RE.search(
            _unquoted_prose(_message_text(row).lower())
        ):
            return ""
        if row.get("role") == "user":
            if not _is_real_user_message(row):
                return ""
            request = _unquoted_prose(_message_text(row).lower()).strip()
            return request if (
                _EDIT_REQUEST_RE.search(request) and "?" not in request
                and not _DECLINED_RE.search(request) and not _REPORTED_REQUEST_RE.search(request)
            ) else ""
    return ""


def has_current_turn_clarification(messages: list[dict[str, Any]]) -> bool:
    """Route even denied clarification through the same visible-ack safety gate."""
    for row in reversed(messages):
        if not isinstance(row, dict):
            continue
        if row.get("role") == "user":
            break
        if row.get("role") == "tool" and row.get("name") == "clarify":
            return True
        if any(call.get("function", {}).get("name") == "clarify"
               for call in row.get("tool_calls") or [] if isinstance(call, dict)):
            return True
    return False


def _clarification_only_turn(messages: list[dict[str, Any]]):
    """Return answered clarification count, or None if execution may have begun.

    Pair calls with results rather than trusting a result's display name. Unknown
    tools, missing results, and partial/declined answers conservatively end recovery.
    """
    pending = set()
    count = 0
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], dict) and messages[i].get("role") == "user"), 0)
    if start:
        from agent.conversation_compression import _is_real_user_message, _message_text

        latest = messages[start - 1]
        text = _message_text(latest).strip()
        # Other recovery/verification rounds are not new, unstarted human work.
        if (not _is_real_user_message(latest)
                or text.startswith(("[System:", "[OUT-OF-BAND USER MESSAGE"))
                or _DECLINED_RE.search(text.lower())):
            return None
    for msg in messages[start:]:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict) or call.get("function", {}).get("name") != "clarify":
                return None
            call_id = call.get("id")
            if not call_id or call_id in pending:
                return None
            pending.add(call_id)
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id")
            if call_id not in pending or not _answered_clarification(msg.get("content")):
                return None
            pending.remove(call_id)
            count += 1
    return None if pending else count


def has_live_ack_work(agent) -> bool:
    """An earlier turn may still own background work; do not duplicate it."""
    from tools.async_delegation import has_live_for_session
    from tools.process_registry import process_registry

    if has_live_for_session(parent_session_id=getattr(agent, "session_id", "") or ""):
        return True
    owners = getattr(agent, "_process_owner_task_ids", ())
    return bool(owners) and any(
        row["owner_task_id"] in owners and row["status"] == "running"
        for row in process_registry.list_sessions()
    )


def has_unresolved_clarification(messages: list[dict[str, Any]]) -> bool:
    """A ``clarify`` call this turn that is unanswered, declined, timed out or partial.

    The user has not authorized anything yet, so no announcement should be retried.
    """
    start = next((i + 1 for i in range(len(messages) - 1, -1, -1)
                  if isinstance(messages[i], dict) and messages[i].get("role") == "user"), 0)
    pending = set()
    for msg in messages[start:]:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict) and call.get("function", {}).get("name") == "clarify":
                pending.add(call.get("id"))
        if msg.get("role") == "tool" and msg.get("tool_call_id") in pending:
            if _answered_clarification(msg.get("content")):
                pending.discard(msg.get("tool_call_id"))
    return bool(pending)


def clarification_ack(user_text: str, messages: list[dict[str, Any]]) -> Optional[str]:
    """Resolve an answered-clarification turn for the detector.

    Returns the task request to judge the announcement against, or ``None`` when
    this is not a clarification-only turn or the answer/request forbids retrying.
    """
    if not _clarification_only_turn(messages):
        return None
    if _DECLINED_RE.search(user_text):
        return None
    return _clarification_task_request(user_text, messages) or None


def clarified_ack_signal(prose: str) -> bool:
    """Announcement shapes that count only after an answered clarification."""
    from agent.agent_runtime_helpers import trailing_continue_intent

    return bool(_CLARIFIED_ACTION_RE.search(prose) or trailing_continue_intent(prose))


def terminal_or_question(assistant_text: str) -> bool:
    return "?" in assistant_text or bool(_ACK_TERMINAL_RE.search(assistant_text))


def unquoted_prose(text: str) -> str:
    return _unquoted_prose(text)

def user_declined(user_text: str) -> bool:
    """Did the user explicitly refuse, defer or gate the work this turn?

    ``_DECLINED_RE`` was already applied inside :func:`clarification_ack`, which
    only runs on an answered-clarification turn. The general detector path never
    consulted it, so "do not inspect the project yet" / "I'll inspect the files"
    produced a continuation signal (#69779, reported by @cygnostik). Exposed here
    so both paths share one definition rather than a second copy of the pattern.
    """
    return bool(_DECLINED_RE.search(user_text))


_ACK_WORKSPACE_MARKERS = (
    "directory", "current directory", "current dir", "cwd", "repo", "repository", "codebase",
    "project", "folder", "filesystem", "file tree", "files", "path",
)
# The markers above, matched as whole words (plurals allowed) rather than as substrings:
# "repo" must not match "report", "files" must not match "profiles", "path" must not match
# "sympathy". Reported on #69779 (and the same bug class as #101868's "read" in "bread").
_ACK_WORKSPACE_RE = re.compile(
    r"\b(?:current\s+dir(?:ectory)?|director(?:y|ies)|cwd|repos?|repositor(?:y|ies)|"
    r"codebases?|projects?|folders?|filesystems?|file\s+trees?|files|paths?)\b"
)


def _mentions_workspace(text: str) -> bool:
    return bool(_ACK_WORKSPACE_RE.search(text))


# Action verbs an ack may announce. Word-boundaried with suffix control so a
# stem does not match an unrelated word: creat!=creative/creature, kill!=skill/
# killer, install!=installment, curl!=curly, boot!=booties/bootstrap/reboot,
# generat!=generational, provision!=provisional.
_ACK_ACTION_ALT = (
    r"execut(?:e|es|ed|ing|ion)?|look\s+(?:into|at)|inspect(?:s|ed|ing|ion)?|"
    r"scan(?:s|ned|ning)?|check(?:s|ed|ing)?(?!\s+in\b)|analy[sz](?:e|es|ed|ing|is)?|"
    r"review(?:s|ed|ing)?|explor(?:e|es|ed|ing)|read(?:s|ing)?(?!\s+(?:up|you|to)\b)|"
    r"open(?:s|ed|ing)?(?!\s+(?:with|to|by)\b)|"
    r"run(?:s|ning)?(?!\s+(?:with|through)\b)|test(?:s|ed|ing)?|fix(?:es|ed|ing)?|"
    r"debug(?:s|ged|ging)?|search(?:es|ed|ing)?|find(?:s|ing)?|walk\s*through|report\s+back|"
    r"summari[sz](?:e|es|ed|ing)?|migrat(?:e|es|ed|ing|ion)?|"
    r"deleg(?:ate|ates|ated|ating|ation)?|deploy(?:s|ed|ing|ment)?|"
    r"install(?:s|ed|ing|ation)?|build(?:s|ing|ed)?(?!\s+on\b)|compil(?:e|es|ed|ing)?|"
    r"scaffold(?:s|ed|ing)?|generat(?:e|es|ed|ing|ion)?|creat(?:e|es|ed|ing|ion)?|"
    r"set(?:s|ting)?\s+(?:\w+\s+){0,3}up|"
    r"spin(?:s|ning)?\s+(?:\w+\s+){0,3}up|"
    r"bring(?:s|ing)?\s+(?:\w+\s+){0,3}up|launch(?:es|ed|ing)?|"
    r"provision(?:s|ed|ing)?|configur(?:e|es|ed|ing|ation)?|verif(?:y|ies|ied|ying)|"
    r"curl|boot(?:s|ed|ing)?|"
    r"kill(?:s|ed|ing)?|remov(?:e|es|ed|ing)|delet(?:e|es|ed|ing|ion)?|"
    r"tear\s+down|teardown|clean(?:\s+|-)up|shut\s+down|shutdown"
)
_ACK_ACTION_RE = re.compile(r"\b(?:" + _ACK_ACTION_ALT + r")\b")

# First-person imminent-action lead-ins. Excludes "let me know" (a sign-off/
# offer) and drops the phantom-prone "on it"/"let's" (which matched "suite on
# it", "let's talk").
_ACK_FUTURE_ACK_RE = re.compile(
    r"\b(?:i['’]ll|i\s+will|i['’]?m\s+going\s+to|i\s+am\s+going\s+to|"
    r"i(?:['’]?m|\s+am)\s+about\s+to|going\s+straight\s+to|let\s+me(?!\s+know)|"
    r"now\s+i(?:['’]ll|['’]m(?=\s+\w+ing\b)|\s+will|\s+am(?=\s+\w+ing\b))|"
    r"proceeding|executing|getting\s+started|kicking\s+off|here\s+goes|"
    r"starting\s+now|i\s+can\s+do\s+that|i\s+can\s+help\s+with\s+that)\b"
)

# Governed announcement: a first-person lead-in directly governing an action
# verb within the SAME clause (no sentence punctuation between them). This is
# the precise "I'll deploy it" / "let me set up the stack" signal, and it
# excludes cross-clause coincidences like "I'll admit, building rapport" (comma)
# or "in the long run" (no lead-in at all).
_ACK_ANNOUNCE_RE = re.compile(
    r"\b(?:i['’]ll|i\s+will|i['’]?m\s+going\s+to|i\s+am\s+going\s+to|"
    r"i(?:['’]?m|\s+am)\s+about\s+to|going\s+straight\s+to|let\s+me(?!\s+know)|"
    r"now\s+i(?:['’]ll|['’]m(?=\s+\w+ing\b)|\s+will|\s+am(?=\s+\w+ing\b)))\b"
    r"(?:(?!\b(?:never|not)\b)[^,.:;!?\n]){0,40}?\b(?:" + _ACK_ACTION_ALT + r")\b"
    # A negated object right after the verb is a stand-down, not an announcement:
    # "I'll run no further commands", "I'll make no changes", "I'll touch nothing".
    r"(?!\s+(?:no|not|nothing|nor)\b)"
)

# Sign-off / conditional-offer / question-for-input: the model is offering
# future help, reporting finished work, or asking the user — not announcing an
# imminent action. Suppress even when an action word co-occurs.
_ACK_OFFER_RE = re.compile(
    r"\b(?:let me know|if you['’]?d\s+like|if you\s+want|if you['’]?d\s+prefer|"
    r"would you like|feel free to|happy to help|"
    r"another time|some other time|at a later|later on|next week|not now)\b"
)

# The model is blocked on the user (approval, credentials, a go-ahead) or reports the
# request complete: the announcement is conditional or moot, not an unstarted action.
# A question mark ends a question only before whitespace/closer, not inside a URL query.
_ACK_QUESTION_RE = re.compile(r"\?(?=\s|$|[\"'’”)\\])")
_ACK_WAIT_RE = re.compile(
    r"\b(?:never|(?:once|when|after|until)\s+you|"
    r"(?:once|when|after|until)\s+(?:(?:it|they|this|that)\s+|the\s+[\w\s-]{0,30}?\b)"
    r"(?:finish(?:es|ed)|complet(?:es|ed)|(?:is|are)\s+(?:done|finished|complete))|"
    r"your\s+(?:approval|permission|confirmation)|"
    r"go.ahead|credentials?|password|"
    r"(?:review|task|work|request(?:ed)?\s+\w+)\s+(?:is|was)\s+(?:now\s+)?(?:complete|done|finished))\b"
)


def _pronounless_action(assistant_text: str) -> bool:
    """Terse log-style narration of the next action, with an explicit transition cue."""
    # Some tool-using providers narrate the next action in terse log style: "Brief written. Creating the
    # session now." Bare gerunds are too ambiguous ("Reading the traceback explains the failure" is a final
    # answer), so this path also requires an explicit transition cue such as "now", "on <ACP agent> via acpx",
    # "actual log", or "then launching". Completed reports and questions stay final.
    action_clause_pattern = re.compile(
        r"(?:^|[.!…—–]\s+|\n+\s*|,\s+then\s+)"
        r"(?:(?P<transition>then)\s+)?"
        r"(?P<action>(?:re)?launching|(?:re)?starting|creating|"
        r"checking|running|writing|opening|reading|inspecting|reviewing|"
        r"testing|debugging|searching|fixing)\b"
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
                (is_launch_action and launch_now_tail_pattern.fullmatch(clause_tail))
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
            return True
    return False


def _clarified_turn_ack(agent, clar_text: str, assistant_content: str, require_workspace: bool) -> bool:
    """The announcement after an answered ``clarify`` turn, judged against the resolved request.

    The request is still unstarted (the only tool work this turn was the clarification), so the
    after-tool gate does not apply; decline/wait/terminal wording still ends the turn.
    """
    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False
    if terminal_or_question(assistant_text):
        return False
    prose = unquoted_prose(assistant_text)
    if not (clarified_ack_signal(prose) or _ACK_ANNOUNCE_RE.search(prose)):
        return False
    if not require_workspace:
        return True
    return _mentions_workspace(clar_text) or "/" in clar_text or _mentions_workspace(prose)


def looks_like_codex_intermediate_ack(
    agent, user_message: Any, assistant_content: str, messages: list[dict[str, Any]],
    require_workspace: bool = True,
) -> bool:
    """Detect a planning/ack message that should continue instead of ending the turn.

    ``require_workspace`` (default True) keeps the original codex-coding scope:
    the ack must reference a filesystem/repo workspace. The conversation loop
    passes ``require_workspace=False`` when the user has explicitly opted into
    intent-ack continuation for all api_modes (``agent.intent_ack_continuation``
    is ``true`` or a model-list), so general autonomous workflows ("I'll run a
    health check on the server", "I'll start the deployment") — which carry a
    future-ack and an action verb but no filesystem reference — are caught too.

    In opted-in ``all`` mode the fire condition is a GOVERNED announcement (a
    first-person lead-in directly governing an action verb in the same clause,
    :data:`_ACK_ANNOUNCE_RE`) or a colon-truncated ending carrying a future-ack
    and an action verb. A shared sign-off/offer guard (:data:`_ACK_OFFER_RE`),
    word-boundaried action verbs, and the ``let me``-not-``know`` future-ack keep
    ordinary prose from tripping it. codex_only additionally requires a workspace
    reference and a current-turn after-tool gate. Known residual (bounded by the
    loop's max-2 continuations): in ``all`` mode a first-person lead governing an
    everyday verb with a conversational object ("I'll review your essay") still
    fires — the verb match cannot see the object; the mode is meant for
    autonomous task sessions, and codex_only's workspace requirement avoids it.
    """
    from agent.codex_responses_adapter import _summarize_user_message_for_log as _sum

    _user_text = _sum(user_message).strip().lower()
    # An explicit refusal governs the whole turn, so this is checked before the
    # clarification and general paths diverge. The guard already existed inside
    # clarification_ack, which only runs on an answered-clarification turn; the
    # general path never consulted it, so a declined request still produced a
    # continuation signal. Reported by @cygnostik on #69779: at 14b330c3 both
    # require_workspace values returned False for the repro, at 40193d60 both
    # returned True.
    if user_declined(_user_text):
        return False
    if has_unresolved_clarification(messages):
        return False
    _clar_text = clarification_ack(_user_text, messages)
    if _clar_text is not None:
        return _clarified_turn_ack(agent, _clar_text, assistant_content, require_workspace)

    # Prior-tool gate. NEVER scan the whole `messages` history: the store keeps a
    # role=="tool" row for every past tool call, so a full-history scan dead-shorts
    # the detector in any established session that has ever used a tool.
    #
    # codex_only mode keeps the original "a tool ran THIS turn -> don't continue"
    # behavior (scoped to messages after the last user message).
    #
    # Opted-in "all" mode DROPS the after-tool gate entirely. The dominant
    # local-model failure is "do some work, announce the next step, then stop"
    # ("files are in place; let me launch the stack:") — which lands AFTER a tool
    # call this turn, so gating on it leaves the task one step short of done. The
    # loop's max-consecutive-continuation bound already prevents runaway (the
    # earlier re-delegation loops were the model re-issuing tool calls, which this
    # detector never fires on). So a first-response ack is caught in both modes,
    # and a mid-task "let me finish it:" ack is caught only when opted in.
    if require_workspace:
        # Runtime scaffolding (verification/continuation nudges, compaction summaries)
        # is role="user" but not a new request; it must not hide this turn's tool work.
        from agent.conversation_compression import _is_real_user_message
        _turn_msgs = messages
        for _i in range(len(messages) - 1, -1, -1):
            _m = messages[_i]
            if _is_real_user_message(_m):
                _turn_msgs = messages[_i + 1:]
                break
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in _turn_msgs):
            return False

    assistant_text = agent._strip_think_blocks(assistant_content or "").strip().lower()
    if not assistant_text or len(assistant_text) > 1200:
        return False

    # Quoted examples, blockquotes and code are content, not the model's own commitment.
    assistant_text = unquoted_prose(assistant_text)
    if not assistant_text.strip():
        return False

    # Sign-off / conditional-offer / question-for-input: the model is offering
    # future help, reporting finished work, or asking the user for input — not
    # announcing an imminent action. Suppress in every mode even when an action
    # word co-occurs ("Let me know if you'd like me to create more examples.").
    if (_ACK_QUESTION_RE.search(assistant_text) or _ACK_OFFER_RE.search(assistant_text)
            or _ACK_WAIT_RE.search(assistant_text)):
        return False

    has_pronounless_action = _pronounless_action(assistant_text)
    has_future_ack = bool(_ACK_FUTURE_ACK_RE.search(assistant_text))
    assistant_mentions_action = bool(_ACK_ACTION_RE.search(assistant_text))
    # Truncation signal: a short reply cut off on a colon ("Going straight to
    # execution now:"). Only used alongside a future-ack + action verb below.
    looks_truncated = len(assistant_text) <= 300 and assistant_text.rstrip().endswith(":")


    # Opted-in (all-api_mode) path. Fire on a GOVERNED announcement: a
    # first-person lead-in directly governing an action verb in the same clause
    # ("I'll deploy it", "let me set up the stack"), which excludes cross-clause
    # coincidences ("I'll admit, building rapport"; "in the long run"). Also fire
    # on a truncated colon ending that still carries a future-ack + action verb,
    # for local phrasings that dodge the governed pattern.
    # Both modes share the governed detection; codex_only additionally needs a
    # workspace reference (below).
    if not (
        _ACK_ANNOUNCE_RE.search(assistant_text)
        or (looks_truncated and has_future_ack and assistant_mentions_action)
        or has_pronounless_action
    ):
        return False
    if not require_workspace:
        return True

    # ``user_message`` is typed ``str`` but can arrive as an OpenAI-style
    # multi-part content list (``[{type:"text",...}, {type:"image_url",...}]``)
    # for vision requests routed through the OpenAI-compat API server. A
    # truthy list survives ``(user_message or "")`` and then ``.strip()``
    # raises ``AttributeError`` — flatten to text first.
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message).strip().lower()
    return (
        _mentions_workspace(user_text)
        or "~/" in user_text
        or "/" in user_text
        or _mentions_workspace(assistant_text)
    )
