"""Pure deterministic Work Router rules.

No model, middleware, network call, or semantic inference is used here.  The
caller supplies the registered identity map and the canonical event metadata.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Iterable

from .config import BotRegistry, DEFAULT_PROFILES, RouterConfig
from .meeting import preflight_response_schema_instruction
from .models import CanonicalEvent, RouteAction, RouteDecision, RouterThreadState

DEMIAN_GUIDANCE = "여러 팀을 부르셨습니다. 한 팀씩 불러주시거나, 회의가 필요하면 말씀해 주세요."
_MEETING_AGENDA_RE = re.compile(r"회의 시작 안건:[ \t]*(.+)\Z")
MEETING_APPROVAL_TRIGGERS = ("승인",)
MEETING_STOP_TRIGGERS = ("그만", "멈춰")
_EVERYONE_TRIGGERS = ("모두", "모두들", "전부", "다 같이", "다들")
# Everyone-call only counts when the group word is followed by an
# imperative/ask verb.  "모두들" alone (e.g. "웬디야, '모두들' 규칙은...")
# is a mention of the word, not a call to the whole team.
_EVERYONE_IMPERATIVE_RE = re.compile(
    # Sentence-leading group call: "모두들 ..." (any ask follows).
    r"^\s*(모두들?|전부|다들|다\s*같이)(?![가-힣A-Za-z0-9])"
    # Group word + adjacent imperative.
    r"|(모두들?|전부|다들|다\s*같이)\s*(대답|답해|답변|말해|의견|들려|참여|응답|소리|확인|있어|하고|해봐|해라|줘|봐)"
    r"|(모두들?|전부|다들|다\s*같이)\s*[^\s]{0,25}?\s*(대답|답해|답변|말해|의견|들려|참여|응답|소리|확인|있어|하고|해봐|해라|줘|봐)"
    r"|(모두들?|전부|다들|다\s*같이)\s*(각자|한마디씩|각각)[^\s]{0,20}?\s*(말해|해봐|해라|줘|봐|의견|생각)"
    r"|(들리면|들리는)\s*(모두들?|전부|다들|다\s*같이)",
)
_SLACK_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
_WORK_HEADER_RE = re.compile(r"^\[WORK\] <@([A-Z0-9]+)> :: ([^\r\n]{1,160})$")


@dataclass(frozen=True)
class ParsedWorkDirective:
    title: str
    body: str
    owner_profile: str
    owner_slack_user_id: str
    directive_id: str


@dataclass(frozen=True)
class DirectiveRejection:
    reason: str
    reserved: bool = True


def parse_work_directive(
    event: CanonicalEvent,
    config: RouterConfig,
) -> ParsedWorkDirective | DirectiveRejection:
    """Parse and authorize the exact reserved Demian-only v1 directive."""

    text = event.text or ""
    if not text.startswith("[WORK]"):
        # Lookalike spellings are reserved too: fail closed instead of
        # allowing their real Staff mention to fall through to Advisor routing.
        if re.match(r"^\s*\[work\]", text, re.IGNORECASE):
            return DirectiveRejection("malformed_header")
        return DirectiveRejection("not_directive", reserved=False)
    execution = config.work_execution
    if not execution.enabled:
        return DirectiveRejection("feature_disabled")
    if not execution.ready:
        return DirectiveRejection("feature_not_ready")
    if config.registry is None:
        return DirectiveRejection("registry_unavailable")
    if event.author_kind != "bot" or event.author_profile != config.registry.default:
        return DirectiveRejection("issuer_not_demian")
    demian = config.registry.by_name(config.registry.default)
    if demian is None or event.author_user_id != demian.slack_user_id:
        return DirectiveRejection("issuer_identity_mismatch")
    if event.channel_type.strip().lower() in {"dm", "im", "mpim", "app_home", "home"}:
        return DirectiveRejection("channel_not_allowed")
    if event.channel_id not in execution.channel_allowlist:
        return DirectiveRejection("channel_not_allowed")
    if str(event.metadata.get("slack_team_id") or "") != execution.slack_team_id:
        return DirectiveRejection("workspace_mismatch")
    slack_subtype = event.metadata.get("slack_event_subtype")
    if slack_subtype not in {None, "", "bot_message"} or any(
        event.metadata.get(key)
        for key in ("subtype", "edited", "deleted", "file_only", "forwarded")
    ):
        return DirectiveRejection("unsupported_event")
    first_line, separator, body = text.partition("\n")
    match = _WORK_HEADER_RE.fullmatch(first_line)
    if match is None:
        return DirectiveRejection("malformed_header")
    target_id, raw_title = match.groups()
    target = config.registry.by_user_id(target_id)
    if target is None or target.name == config.registry.default:
        return DirectiveRejection("invalid_target")
    title = raw_title.strip()
    if not title or len(title) > 160:
        return DirectiveRejection("invalid_title")
    body = body if separator else ""
    if len(text.encode("utf-8")) > execution.max_body_bytes:
        return DirectiveRejection("body_too_large")
    registered_mentions = tuple(
        mention[2:-1]
        for mention in _SLACK_MENTION_RE.findall(text)
        if config.registry.by_user_id(mention[2:-1]) is not None
    )
    if registered_mentions != (target_id,):
        return DirectiveRejection("additional_staff_mention")
    team_id = str(event.metadata.get("slack_team_id") or "")
    digest = hashlib.sha256(
        f"v1\0{team_id}\0{event.event_id}".encode("utf-8")
    ).hexdigest()
    return ParsedWorkDirective(title, body, target.name, target.slack_user_id, digest)

# Boundary-safe aliases.  These are deterministic spellings, not semantic
# aliases; a caller can still use a real Slack mention for an unambiguous call.
_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "Demian": ("demian", "데미안"),
    "Hans": ("hans", "한스"),
    "Wendy": ("wendy", "웬디"),
    "Tesla": ("tesla", "테슬라"),
    "Turing": ("turing", "튜링"),
    "Mason": ("mason", "메이슨"),
    "Watson": ("watson", "왓슨"),
}
# Include both vocative suffixes and grammatical reference particles in one pattern.
# Longest-first ordering keeps matches deterministic for overlapping suffixes
# like '으로/로'.
_KOREAN_DIRECT_CALL_SUFFIXES: tuple[str, ...] = ("아", "야", "님", "씨", "군", "양")
_KOREAN_REFERENCE_SUFFIXES: tuple[str, ...] = (
    "에게서",
    "한테",
    "으로",
    "에서",
    "보다",
    "처럼",
    "만큼",
    "부터",
    "까지",
    "마저",
    "조차",
    "따라",
    "커녕",
    "치고",
    "밖에",
    "하고",
    "로",
    "에게",
    "과",
    "랑",
    "의",
    "를",
    "을",
    "가",
    "이",
    "은",
    "는",
    "도",
)
_KOREAN_CALL_SUFFIX_ALTERNATIVES = tuple(
    sorted(
        set(_KOREAN_DIRECT_CALL_SUFFIXES + _KOREAN_REFERENCE_SUFFIXES),
        key=lambda suffix: (-len(suffix), suffix),
    )
)
_KOREAN_CALL_SUFFIX_PATTERN = "(?:" + "|".join(_KOREAN_CALL_SUFFIX_ALTERNATIVES) + ")?"

def route_meeting_control(
    state: RouterThreadState,
    event: CanonicalEvent,
    config: RouterConfig,
    *,
    meeting_status: str | None = None,
) -> RouteDecision | None:
    """Recognize only Sinclair's approved-channel meeting entry/approval commands."""

    if event.author_kind != "human" or not config.authorizes_meeting_control(
        event.channel_id,
        event.author_user_id,
        event.channel_type,
    ):
        return None
    normalized = _normalize_meeting_control_text(event.text)
    if state.mode == "meeting" and normalized in MEETING_STOP_TRIGGERS:
        return RouteDecision(
            RouteAction(
                "meeting_stop",
                target_profile="Demian",
                response_kind="meeting_stop_confirmation",
                reason="sinclair_stop",
            ),
            state,
        )
    direct = (
        "\r" not in event.text
        and "\n" not in event.text
        and (match := _MEETING_AGENDA_RE.fullmatch(event.text.strip(" \t"))) is not None
        and bool(match.group(1).strip(" \t"))
    )
    approval = normalized in MEETING_APPROVAL_TRIGGERS
    if state.mode == "work" and direct:
        reason = "sinclair_direct_meeting_request"
    elif state.mode == "meeting" and meeting_status == "pending_approval" and approval:
        reason = "sinclair_approved_meeting_proposal"
    elif (
        state.mode == "meeting"
        and meeting_status == "awaiting_sensitive_approval"
        and approval
    ):
        return RouteDecision(
            RouteAction(
                "meeting_sensitive_approval",
                target_profile="Demian",
                response_kind="meeting_sensitive_approval",
                content=(
                    "Sales·Goldmine 조회 승인을 기록했습니다. "
                    "실제 조회는 아직 실행하지 않았습니다."
                ),
                reason="sinclair_approved_sensitive_lookup",
            ),
            state,
        )
    else:
        return None
    return RouteDecision(
        RouteAction(
            "meeting_preflight",
            target_profile="Demian",
            response_kind="meeting_preflight",
            reason=reason,
        ),
        replace(state, mode="meeting"),
    )


def is_meeting_stop_request(event: CanonicalEvent, config: RouterConfig) -> bool:
    """Return true only for Sinclair's exact approved-channel stop commands."""

    return (
        event.author_kind == "human"
        and config.authorizes_meeting_control(
            event.channel_id,
            event.author_user_id,
            event.channel_type,
        )
        and _normalize_meeting_control_text(event.text) in MEETING_STOP_TRIGGERS
    )


def parse_meeting_participant_injection(
    text: str,
    registry: BotRegistry,
) -> tuple[str, ...]:
    """Parse one explicit request to add a registered Staff to a live meeting."""

    normalized = _normalize_meeting_control_text(text)
    intent_phrases = (
        "추가해",
        "추가해줘",
        "추가해 줘",
        "넣어줘",
        "넣어 줘",
        "포함시켜",
        "참여시켜",
        "의견도 받아봐",
        "의견 받아봐",
        "검증시켜",
    )
    if not any(phrase in normalized for phrase in intent_phrases):
        return ()
    if re.search(
        r"\S+\s+(?!참가자(?:로)?\b)\S+(?:을|를)\s*(?:추가|넣|포함)",
        normalized,
        re.IGNORECASE,
    ):
        return ()
    if re.search(
        r"(?:자료|파일|문서|링크|내용|정보)(?:도)?\s*(?:추가|넣|포함)",
        normalized,
        re.IGNORECASE,
    ):
        return ()

    matches: list[str] = []
    lowered = normalized.casefold()
    for profile in registry.profiles:
        aliases = _NAME_ALIASES.get(profile.name, (profile.name.casefold(),))
        matched = any(
            _contains_boundary_name(lowered, alias.casefold())
            for alias in aliases
        )
        if not matched:
            matched = any(
                re.search(
                    r"(?<![\w])"
                    + re.escape(alias.casefold())
                    + r"(?:와|과|랑|하고|및)(?![\w])",
                    lowered,
                    re.IGNORECASE,
                )
                is not None
                for alias in aliases
            )
        if matched:
            matches.append(profile.name)

    if len(matches) != 1 or matches[0] == registry.default:
        return ()
    return (matches[0],)


def _normalize_meeting_control_text(text: str) -> str:
    without_mentions = _SLACK_MENTION_RE.sub(" ", text)
    return " ".join(without_mentions.split())


def build_meeting_preflight_prompt(agenda: str) -> str:
    """Build Demian's fail-closed, evidence-first stage-one preflight instruction."""

    schema = preflight_response_schema_instruction()
    return f"""Demian 자료 preflight 요청

안건: {agenda.strip()}

회의 라운드를 시작하지 말고 아래 순서로 내부 자료를 확인하세요.
1. Notion 프로젝트 DB
2. Kanban
3. 공유 파일
4. 외부 공개 검색 connector는 단계 2c 범위 밖이므로 실행하지 않음

확보된 agenda 사실은 facts에, 조회 범위와 자료 없음·조회 불가 근거는 inspection_evidence에 분리하세요. missing 또는 unavailable source에 facts를 넣지 마세요.

계약, 법률, 선행 절차, 일정, 자원, 운영 제약을 모두 평가하세요. 라이선스 이전·전환·증설·재사용 안건은 기존 계약·라이선스 범위와 제조사 승인·발급 절차를 확인하고, 구축·이전·마이그레이션 안건은 작업 승인·일정·인력·장비·권한·중단 가능 시간·접근 제약을 확인하세요. 고객 데이터·개인정보 처리 안건은 법률·규제·고객 내부정책 적용 여부를 확인하세요. 순수 아이디어 탐색은 실행·계약·고객 확정을 내리지 않는다는 agenda 또는 source 근거가 있을 때만 관련 항목을 not_applicable로 판정하세요. 고객별 계약 조항이나 기능 가능 여부를 모델 지식으로 만들지 마세요.

없는 자료가 하나라도 회의 판단에 필요한 blocking premise이면 있는 자료만으로 회의를 시작하지 마세요. 필요한 자료를 blocking premise별 ordered array로 반환하고 정지하세요.

자료가 충분하면 안건과 역할 적합성에 따라 Staff 3~4명을 선정하세요. Demian은 moderator이므로 참석자에 포함하지 마세요.
Sales·Goldmine이 꼭 필요하면 실제 조회하지 말고 대상·항목·이유를 적은 승인 요청만 반환하세요.
모든 blocking premise가 통과하기 전에는 참석자를 확정하지 마세요. 후보 호출과 회의 라운드는 만들지 마세요.

{schema}"""


def route(
    state: RouterThreadState,
    event: CanonicalEvent,
    registry: BotRegistry,
    *,
    handoff_count: int = 0,
    now: float | None = None,
) -> RouteDecision:
    """Return one deterministic action and a state intent.

    The returned state is not persisted by this function.  A store must apply
    it using a durable lease and compare-and-swap.
    """

    if state.mode != "work":
        return _silence(state, "unsupported_mode")
    if event.author_kind == "human":
        return _route_human(state, event, registry)
    registered_author = registry.by_name(event.author_profile or "")
    if (registered_author is None
            or (event.author_user_id and event.author_user_id != registered_author.slack_user_id)
            or (event.author_profile == registry.default
                and event.author_user_id != registered_author.slack_user_id)):
        return _silence(state, "untrusted_bot_identity")
    advisor_profile = state.advisor_stack[-1] if state.advisor_stack else None
    if advisor_profile and event.author_profile == advisor_profile and state.owner:
        # The Advisor's own response is an inbound completion event.  It does
        # not trigger another outbound post; it only returns speaking state to
        # the original Owner after the Advisor message has already succeeded.
        return RouteDecision(
            RouteAction(
                "advisor_complete",
                target_profile=state.owner,
                response_kind="advisor_completion",
                reason="advisor_single_response_completed",
            ),
            transition_advisor_completion(state, advisor_profile),
        )
    return _route_bot(state, event, registry, handoff_count=handoff_count)


def transition_advisor_completion(
    state: RouterThreadState,
    advisor_profile: str,
    *,
    now: float | None = None,
) -> RouterThreadState:
    """Build the post-confirmed-success return-to-owner state.

    This helper is deliberately separate from :func:`route`: an outbound
    advisor response is only allowed to pop the stack after Slack confirms it.
    """

    if not state.advisor_stack or state.advisor_stack[-1] != advisor_profile:
        return state
    timestamp = state.updated_at if now is None else now
    return replace(
        state,
        active_team=state.owner,
        advisor_stack=state.advisor_stack[:-1],
        updated_at=timestamp,
        expires_at=state.expires_at,
    )


def extract_human_team_calls(
    text: str,
    registry: BotRegistry,
    mentioned_user_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return distinct registered calls in stable seven-profile order."""

    calls: set[str] = set()
    for profile in registry.profiles:
        for alias in _NAME_ALIASES.get(profile.name, ()):
            if _contains_boundary_name(text, alias):
                # "@alias", "alias이/의/를/가/는/에게/과…" are references,
                # not direct calls.  Skip them so that only actual
                # imperatives / standalone names trigger multi-team dispatch.
                if not _is_reference_not_call(text, alias):
                    calls.add(profile.name)
                break
    # Only add @-mentions when NO team was found in the text.
    # When the text already has an explicit team call (e.g. "데미안,
    # @메이슨에게 전달해줘"), the @mention is part of the instruction,
    # not a direct call to that bot — avoid multi-team dispatch.
    mention_ids = tuple(mentioned_user_ids)
    if len(set(mention_ids)) != len(mention_ids):
        return ()
    if not calls:
        for user_id in mention_ids:
            profile = registry.by_user_id(user_id)
            if profile is None:
                return ()
            calls.add(profile.name)
    return tuple(name for name in DEFAULT_PROFILES if name in calls)


def _route_human(
    state: RouterThreadState,
    event: CanonicalEvent,
    registry: BotRegistry,
) -> RouteDecision:
    if event.text.strip() == "여기까지":
        partial = replace(
            state,
            active_team=None,
            advisor_stack=(),
            last_human_target=None,
        )
        return RouteDecision(
            RouteAction("partial_reset", response_kind="silence", reason="exact_termination"),
            partial,
        )

    calls = extract_human_team_calls(event.text, registry, event.mentioned_user_ids)

    # "모두들 답해" — all staff bots respond. This overrides any explicit
    # single-team name call: when the user asks "모두", every team speaks.
    # Demian is included too — "모두들" means the whole team.
    # Only counts when the group word is an actual imperative (e.g.
    # "모두들 대답해"), not when it's quoted or merely mentioned.
    if _EVERYONE_IMPERATIVE_RE.search(event.text):
        calls = tuple(name for name in DEFAULT_PROFILES)

    if len(calls) >= 2:
        # Multi-team: fan out to every mentioned team directly.  Each team is
        # dispatched with the original user text; no Demian relay.
        import json as _json
        payload = _json.dumps({"teams": list(calls), "text": event.text})
        return RouteDecision(
            RouteAction(
                "multi_team_dispatch",
                target_profile=None,
                response_kind="multi_team_dispatch",
                content=payload,
                reason="multiple_distinct_human_team_calls",
            ),
            state,
        )
    if len(calls) == 1:
        target = calls[0]
        next_state = replace(
            state,
            active_team=target,
            last_human_target=target,
        )
        return RouteDecision(
            RouteAction(
                "dispatch",
                target,
                response_kind="team",
                reason="single_human_team_call",
            ),
            next_state,
        )

    target = state.active_team or state.owner or state.last_human_target
    if target is None:
        # No explicit Team context yet in this thread. Use the default
        # conversational coordinator (Demian).
        target = registry.default
        return RouteDecision(
            RouteAction("dispatch", target, response_kind="team", reason="no_active_team_or_owner"),
            state,
        )
    return RouteDecision(
        RouteAction("dispatch", target, response_kind="team", reason="unnamed_human_follow_up"),
        state,
    )


def _route_bot(
    state: RouterThreadState,
    event: CanonicalEvent,
    registry: BotRegistry,
    *,
    handoff_count: int,
) -> RouteDecision:
    # Plain text team names in bot output are never calls.  Only the currently
    # registered owner bot may create an advisor turn through real mentions.
    # Exception: when no owner is set yet (e.g. Mason calls @Turing in a
    # collaboration thread), allow the @mention to dispatch directly.
    mention_ids = tuple(event.mentioned_user_ids)
    # Only the registry-bound PM identity may explicitly fan out. Textual names
    # and unregistered/spoofed bot identities are not delegation authority.
    trusted_pm = (
        event.author_profile == registry.default
        and event.author_user_id == registry.user_id_for(registry.default)
    )
    if trusted_pm and mention_ids and len(set(mention_ids)) == len(mention_ids):
        targets = [registry.by_user_id(uid) for uid in mention_ids]
        if all(p is not None and p.name != registry.default for p in targets):
            if handoff_count >= 5:
                return _handoff_summary(state, registry)
            import json
            names = [p.name for p in targets if p is not None]
            return RouteDecision(
                RouteAction(
                    "multi_team_dispatch", response_kind="delegation",
                    content=json.dumps({"teams": names, "text": event.text}),
                    reason="trusted_pm_explicit_delegation",
                ),
                replace(state, owner=state.owner or registry.default),
                handoff_count_delta=1,
            )
    if event.author_profile != state.owner or not state.owner:
        # No current owner, but there's a bot @mention — dispatch to the
        # mentioned bot directly (start a new ownership chain).
        if (not state.owner and event.author_profile
                and mention_ids
                and len(set(mention_ids)) == len(mention_ids)):
            mentioned = [registry.by_user_id(user_id) for user_id in mention_ids]
            if all(profile is not None for profile in mentioned):
                profiles = [profile.name for profile in mentioned if profile is not None]
                others = [n for n in profiles if n != event.author_profile]
                if len(others) == 1:
                    if handoff_count >= 5:
                        return _handoff_summary(state, registry)
                    target = others[0]
                    next_state = replace(
                        state,
                        owner=event.author_profile,
                        active_team=target,
                        last_human_target=target,
                    )
                    return RouteDecision(
                        RouteAction("dispatch", target,
                                    response_kind="team",
                                    reason="bot_mention_no_owner"),
                        next_state,
                        handoff_count_delta=1,
                    )
        return _silence(state, "bot_not_current_owner")

    if not mention_ids or len(set(mention_ids)) != len(mention_ids):
        return _silence(state, "missing_or_duplicate_real_advisor_mention")
    mentioned = [registry.by_user_id(user_id) for user_id in mention_ids]
    if any(profile is None for profile in mentioned):
        return _silence(state, "unregistered_bot_mention")
    profiles = [profile.name for profile in mentioned if profile is not None]
    advisors = [name for name in profiles if name != state.owner]
    if len(advisors) != 1 or len(profiles) != 1:
        return _silence(state, "advisor_mention_not_unambiguous")

    if handoff_count >= 5:
        return _handoff_summary(state, registry)

    advisor = advisors[0]
    next_state = replace(
        state,
        active_team=advisor,
        advisor_stack=state.advisor_stack + (advisor,),
    )
    return RouteDecision(
        RouteAction("dispatch", advisor, response_kind="advisor", reason="owner_real_advisor_mention"),
        next_state,
        handoff_count_delta=1,
    )


def _handoff_summary(state: RouterThreadState, registry: BotRegistry) -> RouteDecision:
    return RouteDecision(
        RouteAction(
            "summary", target_profile=registry.default,
            response_kind="demian_summary", reason="sixth_handoff_summary_only",
            content="[handoff_limited] 자동 위임 한도에 도달했어. 작업 완료가 아니야. "
                    "PM이 현재 결과와 대기 항목을 검토하고 명시적으로 재개해야 해.",
        ), state,
    )


def _silence(state: RouterThreadState, reason: str) -> RouteDecision:
    return RouteDecision(RouteAction("silence", response_kind="silence", reason=reason), state)



def _is_reference_not_call(text: str, alias: str) -> bool:
    """Return True when `alias` in `text` is a reference, not a direct call."""
    if not text or not alias:
        return True

    import re as _re

    normalized = _re.sub(r"[_*~]", " ", text)
    pattern = _re.compile(
        r"(?<![\w@])" + _re.escape(alias) + r"(?P<suffix>" + _KOREAN_CALL_SUFFIX_PATTERN + r")" + r"(?![\w])",
        _re.IGNORECASE,
    )

    for m in pattern.finditer(normalized):
        suffix = (m.group("suffix") or "").strip()
        if not suffix:
            # Bare names are treated as direct calls, including '웬디' alone.
            return False

        if suffix in _KOREAN_DIRECT_CALL_SUFFIXES:
            # Vocative suffixes are direct call forms.
            return False

        # Reference-only grammatical forms (e.g. alias의/를/가/은/는/도/에게...)
        # remain references.
        continue

    return True

def _contains_boundary_name(text: str, alias: str) -> bool:
    """Match a literal name or Korean call form outside identifiers/words.

    Slack markdown (``_italic_``, ``*bold*``, ``~strike~``) is normalised
    to spaces so that ``_메이슨_`` still matches a boundary check.
    """

    normalized = re.sub(r"[_*~]", " ", text or "")
    pattern = re.compile(
        r"(?<![\w])"
        + re.escape(alias)
        + _KOREAN_CALL_SUFFIX_PATTERN
        + r"(?![\w])",
        re.IGNORECASE,
    )
    return bool(pattern.search(normalized))
