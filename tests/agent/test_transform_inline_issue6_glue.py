"""Issue #6 glue: the REAL privacy-shield hook feeds the REAL CLI fast-path.

Single-process proof of the composed fix (hermes-security-suite workspace): the
plugin's ``transform_llm_output`` hook restores a mid-text TOKEN_* and returns
a ``str`` subclass with ``transform_is_inline=True``; the CLI reprint fast-path
consumes the turn result and paints nothing — the streamed copy stands alone.
Without the flag the previous full-reprint behavior is unchanged.

Cross-run: ``privacy-shield/tests/test_transform_inline_signal.py`` drives the
real plugin hook at the suite side; this file chains it into the real renderer.

``allow_real_home_io``: the marker is for explicit cross-repo source reads —
this test reads the suite plugin sources under the operator workspace and
performs no writes.
"""

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import pytest

from cli import _post_stream_transform_output

SUITE_PS = Path(
    "/Users/minimetro/.hermes/profiles/operator/workspace/hermes-security-suite/privacy-shield"
)

pytestmark = [
    pytest.mark.allow_real_home_io,
    pytest.mark.skipif(not SUITE_PS.is_dir(), reason="hermes-security-suite workspace not present"),
]


def _load_suite_plugin():
    spec = importlib.util.spec_from_file_location(
        "privacy_shield", SUITE_PS / "__init__.py",
        submodule_search_locations=[str(SUITE_PS)],
    )
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("privacy_shield", entry)
    spec.loader.exec_module(entry)
    assert entry.PrivacyShieldPlugin is not None
    plugin_mod = importlib.import_module("privacy_shield.plugin")
    assert "hermes-security-suite" in plugin_mod.__file__, plugin_mod.__file__
    return entry, plugin_mod


def test_issue6_midtext_token_restore_prints_exactly_once():
    entry, plugin_mod = _load_suite_plugin()
    tmp = Path(tempfile.mkdtemp())
    plugin = plugin_mod.PrivacyShieldPlugin(config=plugin_mod.PrivacyShieldConfig(
        enabled=True, mode="limited", storage_path=str(tmp / "m.db"),
        encryption_key="i" * 32, audit_jsonl_path=str(tmp / "a.jsonl"),
        audit_jsonl_enabled=False,
    ))
    plugin.config.restore_allowed_channels = ("cli",)
    entry.PrivacyShieldPlugin = lambda: plugin

    class Ctx:
        def __init__(self):
            self.hooks = {}

        def register_hook(self, name, cb):
            self.hooks[name] = cb

        def register_middleware(self, name, cb):
            pass

    ctx = Ctx()
    entry.register(ctx)
    ctx.hooks["pre_llm_call"](
        user_message="Die IP 203.0.113.7 ist betroffen",
        session_id="s-issue6", turn_id="t-issue6", model="m",
        platform="cli", is_first_turn=True,
    )
    sanitized = plugin._staged_shields["t-issue6"]["sanitized"]
    match = re.search(r"TOKEN_[A-Z_]+_[0-9a-f]{8,10}_\d{3}", sanitized)
    assert match, sanitized
    token = match.group(0)
    streamed = f"Analysis done. Host {token} reachable."

    flagged = ctx.hooks["transform_llm_output"](
        response_text=streamed, session_id="s-issue6", model="m",
        platform="cli", target_channel="cli",
    )
    assert isinstance(flagged, str)
    assert getattr(flagged, "transform_is_inline", False) is True
    assert "203.0.113.7" in str(flagged)

    # Core per-turn result shape after apply_llm_output_transform ->
    # finalize_turn: exactly-once display for the mid-text substitution.
    result = {
        "response_transformed": True,
        "pre_transform_response": streamed,
        "transform_is_inline": bool(getattr(flagged, "transform_is_inline", False)),
    }
    assert _post_stream_transform_output(str(flagged), result) == ""

    # Regression: no flag -> the previous full-reprint fallback stands.
    unflagged = dict(result)
    unflagged["transform_is_inline"] = False
    assert _post_stream_transform_output(str(flagged), unflagged).startswith("\n")
