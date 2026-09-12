"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields``
renders exactly as before), ``cwd`` (home-relative). ``gateway/run.py`` appends the footer to the
final response only (never to tool-progress or streaming partials); when streaming already
delivered the text, it goes out as a trailing message via ``send_trailing_footer()``."""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


def _home_relative_cwd(cwd: str) -> str:
    r"""Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset.

    The prefix test is compared through ``os.path.normcase``, which folds case
    on Windows and is a no-op everywhere else — so this is case-insensitive on
    Windows and remains case-sensitive on POSIX, where two paths differing only
    in case really are different paths. (macOS filesystems are usually
    case-insensitive, but ``posixpath.normcase`` does not fold case, so the
    comparison stays case-sensitive there too.)

    ``abspath`` normalizes separators but NOT case — that is ``normcase``'s job
    — so a case-sensitive comparison silently fails on Windows for any cwd
    whose casing differs from the canonical profile path (``c:\users\me\src``
    against a home of ``C:\Users\me``). The collapse then no-ops and the footer
    publishes the absolute path, including the OS account name, to whatever
    chat surface the reply is delivered to.

    ``expanduser("~")`` has the same gap one level up: it returns whatever
    ``HOME``/``USERPROFILE`` literally contains, without collapsing redundant
    ``..``/``.`` components — unlike ``cwd``, which always goes through
    ``abspath`` here. A home value like ``C:\Users\decoy\..\me`` and a cwd of
    ``C:\Users\me\src`` name the same directory, but the un-normalized home
    string never prefix-matches the normalized cwd, so the redaction no-ops
    for that account regardless of the case fix above. Normalizing home
    through the same ``abspath`` call closes this the same way.

    Only the comparison is normalized; the tail is sliced from the original
    ``p`` so the displayed path keeps its real casing.
    """
    if not cwd:
        return ""
    try:
        home = os.path.abspath(os.path.expanduser("~"))
        p = os.path.abspath(cwd)
        # normcase folds case on Windows only; it is identity on POSIX
        # (including macOS), so behaviour there is unchanged.
        norm_p = os.path.normcase(p)
        norm_home = os.path.normcase(home)
        if home and (
            norm_p == norm_home or norm_p.startswith(norm_home + os.sep)
        ):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    return model.rsplit("/", 1)[-1] if model else ""


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def resolve_footer_config(user_config: dict[str, Any] | None, platform_key: str | None = None) -> dict[str, Any]:
    """Resolve effective footer config: defaults (enabled=False) <
    ``display.runtime_footer`` < ``display.platforms.<platform_key>.runtime_footer``."""
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}
    plat_cfg = (cfg.get("platforms") or {}).get(platform_key) if platform_key else None
    sections = [cfg.get("runtime_footer"), plat_cfg.get("runtime_footer") if isinstance(plat_cfg, dict) else None]
    for section in sections:
        if not isinstance(section, dict):
            continue
        if "enabled" in section:
            resolved["enabled"] = bool(section.get("enabled"))
        if isinstance(section.get("fields"), list) and section["fields"]:
            resolved["fields"] = [str(f) for f in section["fields"]]
    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS) -> str:
    """Render the footer line, or "" if no fields have data. Fields whose data is missing (and
    unknown field names) are skipped silently — a partial footer beats ``?%`` or empty slots."""
    def context_pct() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"{max(0, min(100, round((context_tokens / context_length) * 100)))}%"
        return ""

    renderers = {
        "model": lambda: _model_short(model),
        "context_pct": context_pct,
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
    }
    return _SEP.join(v for field in fields if (render := renderers.get(field)) and (v := render()))


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS)
