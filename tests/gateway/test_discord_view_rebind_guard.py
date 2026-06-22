"""Best-effort tripwire: only the known lazy-install test triggers the Discord
view-class rebind.

``plugins.platforms.discord.adapter._define_discord_view_classes()`` rebinds the
5 view-class module globals to fresh class objects. A test that calls it (or
``check_discord_requirements()``, which calls it after a lazy install) must
contain the rebind so it can't leak a stale class identity into a later test
(see ``test_discord_lazy_install_views.py``'s restore fixture).

This guard scans ``tests/gateway/`` for callers of those two functions and
asserts only the one known leaker file invokes them — a NEW caller trips this,
pointing the author at the restore fixture pattern.

**This is a BEST-EFFORT tripwire, NOT a completeness guarantee.** A literal text
scan fails OPEN on indirect/aliased/``getattr`` invocation. The REAL guarantee is
the consumer-side live-reference in ``test_discord_clarify_buttons.py``
(``isinstance(view, _discord_adapter.ClarifyChoiceView)`` resolves the class via
the live module attribute, so a consumer can never disagree with production
regardless of any rebind). A future edit must not demote that live-reference and
leave this grep masquerading as the guard.
"""
from pathlib import Path


_GATEWAY_TESTS_DIR = Path(__file__).resolve().parent
_KNOWN_LEAKER = "test_discord_lazy_install_views.py"
# This guard file itself names the caller strings as DATA (not invocations).
_SELF = Path(__file__).name
_EXEMPT = {_KNOWN_LEAKER, _SELF}
_REBIND_CALLERS = (
    "_define_discord_view_classes(",
    "check_discord_requirements(",
)


def test_only_known_file_triggers_discord_view_rebind():
    offenders = []
    for path in sorted(_GATEWAY_TESTS_DIR.glob("test_*.py")):
        if path.name in _EXEMPT:
            continue
        text = path.read_text(encoding="utf-8")
        # Strip simple comment lines so a doc reference doesn't false-trip.
        code_lines = [
            ln for ln in text.splitlines()
            if not ln.lstrip().startswith("#")
        ]
        code = "\n".join(code_lines)
        for caller in _REBIND_CALLERS:
            if caller in code:
                offenders.append(f"{path.name}: calls {caller}")

    assert not offenders, (
        "A NEW test calls a Discord view-class rebind function "
        "(_define_discord_view_classes / check_discord_requirements) outside the "
        f"known leaker {_KNOWN_LEAKER}. The rebind leaks stale class identities; "
        "add the snapshot/restore autouse fixture from "
        "test_discord_lazy_install_views.py to the new file (and rely on the "
        "consumer-side live-reference for the real guarantee). Offenders:\n  "
        + "\n  ".join(offenders)
    )
