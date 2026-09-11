"""Fixed copy templates for agent-authored Collective Wisdom notifications.

The hierarchy, headers, check lines and button labels are fixed here; the
agent supplies only the bracketed prose. Rendering is plain text with a
structured ``actions`` list so each platform adapter (Telegram, Slack,
Desktop) can attach native buttons. The word "Pass" is deliberately never
emitted: safety checks render as ``✓``/``✗`` lines or a single "Yes"/"No".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SHARE_HEADER = "Hermes Collective Wisdom"
SHARE_WHY_HEADER = "Why we ask"
SHARE_SAFE_YES = "Checks complete: no issues detected."
SHARE_SAFE_NO = "Checks found issues. Review the details before sharing."
SHARE_SAFE_UNAVAILABLE = "Checks unavailable. Review is required before sharing."
SHARE_QUESTION = "Would you like to share it?"
SHARE_CHECK_LINES: tuple[str, ...] = (
    "No profanity or abusive language",
    "No hate or harassment",
    "No sexual or graphic content",
    "No detected credentials or private keys",
)
SHARE_BUTTONS: tuple[tuple[str, str], ...] = (
    ("not_now", "Not now"),
    ("review", "Review"),
    ("share", "Share"),
)

TEAMMATE_HEADER = "New skill from your team"
TEAMMATE_SECURITY_LINE = (
    "✓ Security check complete (credentials, private keys, organization policy)"
)
TEAMMATE_POPULAR_LINE = "Popular with your team: {count}+ installations"
TEAMMATE_BUTTONS: tuple[tuple[str, str], ...] = (
    ("mute", "Mute"),
    ("view", "View"),
    ("install", "Install"),
)

PUBLISHED_OPEN_HEADER = "Published to {organization}."
PUBLISHED_OPEN_BODY = "Your teammates can now find and install this skill."
PUBLISHED_MODERATED_HEADER = "Sent for review."
PUBLISHED_MODERATED_BODY = (
    "It is not available to your organization yet. "
    "You will be notified when the review is complete."
)
PUBLISHED_BUTTONS: tuple[tuple[str, str], ...] = (("view_portal", "View in Portal"),)

UPDATE_HEADER = "Update available for {skill_name}"
UPDATE_MATTERS = "This matters for your work because {reason}."
UPDATE_SETUP_IMPACT = "Setup impact: {impact}"
UPDATE_BUTTONS: tuple[tuple[str, str], ...] = (
    ("mute", "Mute"),
    ("view_changes", "View changes"),
    ("update", "Update"),
)

MUTE_OPTIONS: tuple[tuple[str, str, int | None], ...] = (
    ("1d", "1 day", 1),
    ("1w", "1 week", 7),
    ("30d", "30 days", 30),
    ("forever", "Forever", None),
)

FORBIDDEN_WORDS: tuple[str, ...] = ("Pass",)
FORBIDDEN_LABELS: tuple[str, ...] = ("New notification",)


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    primary: bool = False
    target: str | None = None  # opaque callback/URL target bound by the adapter


@dataclass(frozen=True)
class RenderedNotice:
    kind: Literal["share", "teammate", "published", "update", "mute_options"]
    title: str
    lines: list[str]
    actions: list[Action]
    hints: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join([self.title, *self.lines])

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "lines": list(self.lines),
            "text": self.text,
            "actions": [
                {"id": a.id, "label": a.label, "primary": a.primary, "target": a.target}
                for a in self.actions
            ],
            "hints": dict(self.hints),
        }


def _actions(
    spec: tuple[tuple[str, str], ...], *, primary: str, targets: dict[str, str] | None
) -> list[Action]:
    targets = targets or {}
    return [
        Action(id=aid, label=label, primary=(aid == primary), target=targets.get(aid))
        for aid, label in spec
    ]


def _require_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _require_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a real non-negative integer")
    return value


def assert_clean_copy(text: str) -> None:
    """Fail fast if fixed copy or agent prose reintroduces forbidden wording."""
    for word in FORBIDDEN_WORDS:
        for token in text.replace("\n", " ").split(" "):
            if token.strip(".,:;!?()") == word:
                raise ValueError(f"forbidden word in notification copy: {word!r}")
    for label in FORBIDDEN_LABELS:
        if label in text:
            raise ValueError(f"forbidden label in notification copy: {label!r}")


def render_share(
    *,
    editorial_name: str,
    description: str,
    count_7d: int,
    specific_work: str,
    audience: str,
    reason: str,
    checks_passed: bool | None = None,
    failed_checks: list[str] | None = None,
    window_days: int = 7,
    targets: dict[str, str] | None = None,
) -> RenderedNotice:
    name = _require_text(editorial_name, "editorial_name")
    desc = _require_text(description, "description")
    n = _require_count(count_7d, "count_7d")
    work = _require_text(specific_work, "specific_work")
    who = _require_text(audience, "audience")
    why = _require_text(reason, "reason").rstrip(".")
    lines = [
        name,
        f"    {desc}",
        "",
        SHARE_WHY_HEADER,
        (
            f"You used this skill {n} times in the last {window_days} days. "
            f"It has helped with {work}. I think it could help {who} because {why}."
        ),
        "",
        SHARE_SAFE_YES if checks_passed is True else SHARE_SAFE_NO
        if checks_passed is False else SHARE_SAFE_UNAVAILABLE,
    ]
    failed = set(failed_checks or [])
    for check in SHARE_CHECK_LINES:
        if check in failed:
            lines.append(f"⚠ Review: {check.removeprefix('No ')}")
        elif checks_passed is True:
            lines.append(f"✓ {check}")
        else:
            lines.append(f"➖ Unavailable: {check.removeprefix('No ')}")
    lines.append("These checks are not a security certification.")
    lines += ["", SHARE_QUESTION]
    notice = RenderedNotice(
        kind="share",
        title=SHARE_HEADER,
        lines=lines,
        actions=_actions(SHARE_BUTTONS, primary="share", targets=targets),
        hints={"name_emphasis": "editorial_name", "description_indent": True},
    )
    assert_clean_copy(notice.text)
    return notice


def render_teammate(
    *,
    editorial_name: str,
    outcome_one_liner: str,
    publisher_name: str,
    specific_work: str,
    count_7d: int,
    recipient_reason: str,
    installation_count: int | None = None,
    popular_threshold: int = 10,
    window_days: int = 7,
    targets: dict[str, str] | None = None,
) -> RenderedNotice:
    name = _require_text(editorial_name, "editorial_name")
    outcome = _require_text(outcome_one_liner, "outcome_one_liner")
    publisher = _require_text(publisher_name, "publisher_name")
    work = _require_text(specific_work, "specific_work")
    n = _require_count(count_7d, "count_7d")
    why = _require_text(recipient_reason, "recipient_reason").rstrip(".")
    lines = [
        name,
        outcome,
        f"Published by {publisher}",
        "",
        (
            f"{publisher} uses this for {work}. It was used {n} times in the last "
            f"{window_days} days. I think it could help you with {why}."
        ),
        "",
        TEAMMATE_SECURITY_LINE,
    ]
    popular = installation_count is not None and installation_count >= popular_threshold
    if popular:
        lines.append(TEAMMATE_POPULAR_LINE.format(count=popular_threshold))
    notice = RenderedNotice(
        kind="teammate",
        title=TEAMMATE_HEADER,
        lines=lines,
        actions=_actions(TEAMMATE_BUTTONS, primary="install", targets=targets),
        hints={"name_emphasis": "editorial_name", "name_size": "largest", "popular": popular},
    )
    assert_clean_copy(notice.text)
    return notice


def render_published(
    *,
    organization_name: str,
    moderated: bool,
    targets: dict[str, str] | None = None,
) -> RenderedNotice:
    if moderated:
        title, body = PUBLISHED_MODERATED_HEADER, PUBLISHED_MODERATED_BODY
    else:
        org = _require_text(organization_name, "organization_name")
        title, body = PUBLISHED_OPEN_HEADER.format(organization=org), PUBLISHED_OPEN_BODY
    notice = RenderedNotice(
        kind="published",
        title=title,
        lines=[body],
        actions=_actions(PUBLISHED_BUTTONS, primary="view_portal", targets=targets),
        hints={"moderated": moderated},
    )
    assert_clean_copy(notice.text)
    return notice


def render_update(
    *,
    skill_name: str,
    summary: str,
    reason: str,
    setup_impact: str | None = None,
    targets: dict[str, str] | None = None,
) -> RenderedNotice:
    name = _require_text(skill_name, "skill_name")
    text = _require_text(summary, "summary")
    why = _require_text(reason, "reason").rstrip(".")
    impact = str(setup_impact or "").strip() or "None"
    notice = RenderedNotice(
        kind="update",
        title=UPDATE_HEADER.format(skill_name=name),
        lines=[
            text,
            UPDATE_MATTERS.format(reason=why),
            UPDATE_SETUP_IMPACT.format(impact=impact),
        ],
        actions=_actions(UPDATE_BUTTONS, primary="update", targets=targets),
    )
    assert_clean_copy(notice.text)
    return notice


def render_mute_options(*, targets: dict[str, str] | None = None) -> RenderedNotice:
    targets = targets or {}
    actions = [
        Action(id=f"mute:{key}", label=label, target=targets.get(key))
        for key, label, _days in MUTE_OPTIONS
    ]
    notice = RenderedNotice(
        kind="mute_options",
        title="Mute suggestions for how long?",
        lines=["Muting only pauses proactive messages. You can still browse and install."],
        actions=actions,
    )
    assert_clean_copy(notice.text)
    return notice


def mute_duration_days(key: str) -> int | None:
    """Return the day count for a mute option key; ``None`` means forever."""
    for option_key, _label, days in MUTE_OPTIONS:
        if option_key == key:
            return days
    raise KeyError(key)
