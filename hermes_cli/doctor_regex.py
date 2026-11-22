"""``hermes doctor``: regular expressions the user wrote that the RE2 engine refuses or reads differently.

Hermes is moving its regular expressions from Python's ``re`` to RE2 (``hermes_re``), which has no
backreferences and refuses lookaround except as an edge guard (a one-character lookbehind at the very
start, a lookahead at the very end; a negative lookahead tests one character). Until a setting's code has moved, a pattern RE2 refuses still
works; afterwards the setting stops working in the way its ``if_refused`` phrase says. This check
names those patterns ahead of time. A refused one is a manual issue (doctor exits 1, like the neighbouring
upcoming-change notices); one only read differently is a warning.

``SETTINGS`` holds one row per user setting that takes a regex: a label, a reader that yields
``(where, pattern)`` and the consequence of a refusal. Readers call the product's own collectors
(``iter_configured_hooks``, ``iter_configured_targets``, ``load_gateway_config``, ``filter_regexes``,
the webhook subscription store, ``get_scoped_secret``); none parses ``config.yaml`` itself.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from gateway.platform_env import csv_list, mention_patterns
from hermes_cli.doctor_report import Finding, check_info, check_ok, check_warn, doctor_check, warn_on_error

#: Which place an adapter reads a setting from first (see ``_platform_setting``).
First = Literal["env", "yaml", "yaml_key"]


@dataclass(frozen=True)
class Inputs:
    """What the readers share: the CLI config and each configured platform's ``extra`` mapping."""

    config: dict = field(default_factory=dict)
    extras: dict[str, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class Setting:
    label: str
    read: Callable[[Inputs], Iterable[tuple[str, str]]]
    if_refused: str
    ignore_case: bool = False
    bridged: bool = False  # YAML and env var name one value: report a pattern once per setting


@dataclass(frozen=True)
class UserPattern:
    where: str
    pattern: str
    if_refused: str
    ignore_case: bool


def _load_inputs() -> Inputs:
    from gateway.config import load_gateway_config
    from hermes_cli.config import load_config
    extras: dict[str, dict] = {}
    with warn_on_error("Platform settings could not be read", "({e})"):
        extras = {platform.value: dict(cfg.extra or {}) for platform, cfg in load_gateway_config().platforms.items()}
    return Inputs(config=load_config(), extras=extras)


def _hook_matchers(blocking: bool) -> Callable[[Inputs], Iterator[tuple[str, str]]]:
    """Hook matchers whose refusal blocks every tool call (``blocking``: fail_closed on a blocking event,
    as the runtime decides it) or only stops the hook."""
    def read(inputs: Inputs) -> Iterator[tuple[str, str]]:
        from agent.shell_hooks import _BLOCKING_EVENTS, iter_configured_hooks
        hooks = inputs.config.get("hooks")
        for event, entries in (hooks.items() if isinstance(hooks, dict) else ()):
            for index, entry in enumerate(entries if isinstance(entries, list) else ()):
                # One entry at a time so the label carries its index; never the command, which can carry a token.
                for spec in iter_configured_hooks({"hooks": {event: [entry]}}):
                    if spec.matcher and bool(spec.fail_closed and spec.event in _BLOCKING_EVENTS) is blocking:
                        yield f"hooks.{spec.event}[{index}] matcher", spec.matcher
    return read


def _outbound_matchers(inputs: Inputs) -> Iterator[tuple[str, str]]:
    from agent.outbound_webhooks import iter_configured_targets
    hooks = inputs.config.get("hooks")
    entries = hooks.get("outbound") if isinstance(hooks, dict) else None
    for index, entry in enumerate(entries if isinstance(entries, list) else ()):
        # The product's parser, one entry at a time, so the label carries the entry's index in hooks.outbound
        # (as the parser's own warnings do). No other config value: a webhook URL, host included, can carry
        # its token, and doctor output gets pasted into issues.
        for target in iter_configured_targets({"hooks": {"outbound": [entry]}}):
            if target.matcher:
                yield f"hooks.outbound[{index}] matcher", target.matcher


def _route_regexes(kind: str, routes: object) -> Iterator[tuple[str, str]]:
    from gateway.platforms.webhook_filters import filter_regexes
    for name, route in (routes.items() if isinstance(routes, dict) else ()):
        if isinstance(route, dict):
            for pattern in filter_regexes(route.get("filters")):
                yield f"webhook {kind} '{name}' filter regex", pattern


def _static_routes(inputs: Inputs) -> dict:
    routes = inputs.extras.get("webhook", {}).get("routes")
    return routes if isinstance(routes, dict) else {}


def _static_route_regexes(inputs: Inputs) -> Iterator[tuple[str, str]]:
    return _route_regexes("route", _static_routes(inputs))


def _subscription_regexes(inputs: Inputs) -> Iterator[tuple[str, str]]:
    from hermes_cli.webhook import _load_subscriptions
    static = _static_routes(inputs)
    # The adapter drops a subscription that a static route of the same name shadows: its filters never run.
    return _route_regexes("subscription", {name: route for name, route in _load_subscriptions().items()
                                           if name not in static})


def _one(value: str) -> list[str]:
    return [value]


def _json_or_lines(value: str) -> object:
    """Telegram / DingTalk / WhatsApp env form: JSON, else one pattern per line (commas kept), else comma-separated."""
    try:
        return json.loads(value)
    except ValueError:
        return [line.strip() for line in value.splitlines() if line.strip()] or csv_list(value)


def _wake_words(value: str) -> object:
    """Photon / BlueBubbles form (``compile_mention_patterns``): a JSON list, else every line split on commas."""
    try:
        loaded = json.loads(value)
    except ValueError:
        loaded = None
    return loaded if isinstance(loaded, list) else [part for line in value.splitlines() for part in line.split(",")]


def _texts(value: object, split: Callable[[str], object], verbatim: bool) -> list[str]:
    """A setting value as pattern texts: a string goes through ``split``, a list is taken item by item.
    ``verbatim``: the adapter compiles each non-blank string item as it stands and skips other items
    (Telegram, Slack, WhatsApp, DingTalk); otherwise every item is ``str(item).strip()`` (Photon,
    BlueBubbles, Matrix). Doctor must check the text the adapter compiles: ``"x(?=y) "`` is refused, ``"x(?=y)"`` is not."""
    if isinstance(value, str):
        value = split(value)
    if isinstance(value, str):  # a JSON string literal
        value = [value]
    if not isinstance(value, list):
        return []
    if verbatim:
        return [item for item in value if isinstance(item, str) and item.strip()]
    return [text for text in (str(item).strip() for item in value) if text]


def _platform_setting(platform: str, key: str, env: str, *, first: First, yaml_split: Callable[[str], object],
                      env_split: Callable[[str], object], gate_env: bool = False, verbatim: bool = False,
                      ) -> Callable[[Inputs], Iterator[tuple[str, str]]]:
    """Reader for one adapter setting that yields only the value the adapter uses, chosen with that
    adapter's precedence (``first``): ``"env"`` (a non-blank env var of the active profile wins, else the
    YAML), ``"yaml"`` (a YAML value that is not null wins, else the env var) or ``"yaml_key"`` (a YAML key
    that is present wins). ``gate_env`` reads the env var as the gate readers do (Telegram). ``verbatim``: see ``_texts``."""
    def read(inputs: Inputs) -> Iterator[tuple[str, str]]:
        from gateway.platforms._shared import get_scoped_secret, platform_gate_env
        extra = inputs.extras.get(platform, {})
        env_value = (platform_gate_env(env, "") if gate_env else str(get_scoped_secret(env, "") or "")).strip()
        yaml_set = key in extra if first == "yaml_key" else extra.get(key) is not None
        if (bool(env_value) if first == "env" else not yaml_set):
            yield from ((env, pattern) for pattern in _texts(env_value, env_split, verbatim))
        else:
            yield from ((f"{platform}.{key} (config.yaml)", pattern)
                        for pattern in _texts(extra.get(key), yaml_split, verbatim))
    return read


def _mention_patterns(platform: str, env: str, *, first: First = "yaml", yaml_string_is_one_pattern: bool = True,
                      env_split: Callable[[str], object] = mention_patterns, gate_env: bool = False) -> Setting:
    return Setting(
        f"{platform} mention patterns",
        _platform_setting(platform, "mention_patterns", env, first=first, env_split=env_split, gate_env=gate_env,
                          yaml_split=_one if yaml_string_is_one_pattern else env_split,
                          verbatim=yaml_string_is_one_pattern),
        "the pattern will be skipped, so that wake word stops matching", ignore_case=True, bridged=True)


# One row per user setting that takes a regular expression. Each platform row reads the value where its
# adapter reads it (base text): Telegram and Matrix env first; Slack, WhatsApp, DingTalk YAML first (unless
# null); Photon and BlueBubbles YAML first whenever the key is present.
# ``if_refused`` is what happens once the setting's own code runs on RE2, not today's fallback for an invalid
# pattern (a matcher compared as plain text, a filter that answers False, a skipped entry): there a refused hook
# matcher stops its hook and blocks tool calls if ``fail_closed``, a refused route filter stops the webhook
# platform (a subscription is skipped), and a refused Matrix ignore entry stops the adapter, because the old
# fallbacks would let through what those settings exist to keep out.
SETTINGS: tuple[Setting, ...] = (
    Setting("shell hook matcher (fail_closed: true, pre_tool_call)", _hook_matchers(True),
            "the hook will be refused and every tool call blocked until the matcher is fixed"),
    Setting("shell hook matcher", _hook_matchers(False), "the hook will be refused: its command will not run"),
    Setting("outbound webhook matcher", _outbound_matchers, "the target will stop receiving tool events"),
    Setting("Matrix ignore_user_patterns",
            _platform_setting("matrix", "ignore_user_patterns", "MATRIX_IGNORE_USER_PATTERNS", first="env",
                              yaml_split=csv_list, env_split=csv_list),
            "the Matrix adapter will not start", bridged=True),
    Setting("webhook route filter", _static_route_regexes, "the route will not load: the webhook platform will not start"),
    Setting("webhook subscription filter", _subscription_regexes, "the subscription will not load (requests get 404)"),
    _mention_patterns("telegram", "TELEGRAM_MENTION_PATTERNS", first="env", env_split=_json_or_lines, gate_env=True),
    _mention_patterns("slack", "SLACK_MENTION_PATTERNS"),
    _mention_patterns("whatsapp", "WHATSAPP_MENTION_PATTERNS", env_split=_json_or_lines),
    _mention_patterns("whatsapp_cloud", "WHATSAPP_MENTION_PATTERNS", env_split=_json_or_lines),
    _mention_patterns("dingtalk", "DINGTALK_MENTION_PATTERNS", env_split=_json_or_lines),
    _mention_patterns("photon", "PHOTON_MENTION_PATTERNS", first="yaml_key", yaml_string_is_one_pattern=False,
                      env_split=_wake_words),
    _mention_patterns("bluebubbles", "BLUEBUBBLES_MENTION_PATTERNS", first="yaml_key", yaml_string_is_one_pattern=False,
                      env_split=_wake_words),
)


def collect_user_patterns() -> list[UserPattern]:
    """Every regex the active profile configures, each once per place it is set (hook, route, ...). A
    reader that fails is reported and skipped; the others still run."""
    inputs = _load_inputs()
    found: list[UserPattern] = []
    seen: set[tuple[str, str]] = set()  # a YAML value bridged to its env var, or one env var read by two adapters
    for setting in SETTINGS:
        with warn_on_error(f"{setting.label}: could not be read", "({e})"):
            for where, pattern in setting.read(inputs):
                keys = {(where, pattern)} | ({(setting.label, pattern)} if setting.bridged else set())
                if not keys & seen:
                    found.append(UserPattern(where, pattern, setting.if_refused, setting.ignore_case))
                seen.update(keys)
    return found


def _engine_unavailable(exc: ImportError) -> str:
    """Why the engine cannot load. The facade reports every import failure as a missing ``re2``; the
    cause tells a package that is absent from one that is installed but broken (e.g. a source build
    linked against a missing ``libre2``), which needs a different fix."""
    cause = exc.__cause__ or exc
    if isinstance(cause, ModuleNotFoundError) and cause.name == "re2":
        return "google-re2 not installed"
    return f"google-re2 is installed but cannot be imported: {type(cause).__name__}: {cause}"


@doctor_check("Regex settings check skipped", "({e})")
def _check_user_regex_settings(should_fix: bool, f: Finding) -> None:
    """A pattern the engine will refuse is a manual issue (doctor exits 1, like the other
    upcoming-change notices); one that is only read differently is a warning."""
    try:
        import hermes_re
        hermes_re.build("a")  # the engine is loaded on the first build
    except ImportError as exc:
        check_warn("Regex settings", f"(skipped: {_engine_unavailable(exc)})")
        return
    patterns = collect_user_patterns()
    flagged = refused = 0
    for item in patterns:
        flags = hermes_re.IGNORECASE if item.ignore_case else 0
        try:
            hermes_re.build(item.pattern, flags)
            offsets = hermes_re.inexact_boundaries(item.pattern, flags)
        except hermes_re.error as exc:
            flagged += 1
            refused += 1
            check_warn(f"{item.where}: {item.pattern!r} is not a valid RE2 pattern", f"({exc})")
            check_info(f"After the regex engine change {item.if_refused}. Fix the pattern (RE2 has no backreferences; "
                       "lookaround is refused except a one-character lookbehind at the very start and one lookahead "
                       "at the very end, and a negative lookahead tests one character).")
            continue
        if offsets:
            flagged += 1
            check_warn(f"{item.where}: {item.pattern!r} may match differently",
                       f"(the \\b at offset {', '.join(map(str, offsets))} will follow ASCII word rules)")
            check_info("Still accepted after the regex engine change; next to a non-ASCII letter or digit its word boundary can move.")
    if refused:
        f.manual_issues.append(f"Fix {refused} regex setting(s) the RE2 engine will refuse (listed under Regex Settings).")
    if not flagged:
        check_ok(f"{len(patterns)} user regex setting(s) work with the RE2 engine" if patterns else "No user regex settings")
