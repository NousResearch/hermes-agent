"""Public Codex commentary is a display projection, never a replay mutation."""

from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages
from agent.history_commentary import visible_commentary
from hermes_constants import get_hermes_home_override

import asyncio
import pytest


def _row(text="I will inspect the files."):
    return {
        "role": "assistant",
        "content": "",
        "reasoning": f"Private summary.\n\n{text}",
        "reasoning_content": f"Private summary.\n\n{text}",
        "codex_message_items": [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": text}],
            },
            {
                "type": "message",
                "role": "assistant",
                "phase": "analysis",
                "content": [{"type": "output_text", "text": "private analysis"}],
            },
        ],
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": "{}"},
            }
        ],
        "timestamp": 1,
    }


def test_rest_history_separates_public_commentary_without_changing_source():
    row = _row()
    projected = _project_for_display([row])[0]

    assert projected["display_commentary"] == ["I will inspect the files."]
    assert projected["display_reasoning"] == "Private summary."
    assert projected["codex_message_items"] == row["codex_message_items"]
    assert projected["tool_calls"] == row["tool_calls"]
    assert row["reasoning"] == "Private summary.\n\nI will inspect the files."


def test_gateway_history_uses_same_display_projection():
    row = _row()
    projected = _history_to_messages([row])[0]

    assert projected["display_commentary"] == ["I will inspect the files."]
    assert projected["display_reasoning"] == "Private summary."
    assert projected["codex_message_items"] == row["codex_message_items"]
    assert row["reasoning"].endswith("I will inspect the files.")


def test_rest_history_shows_a_steer_as_the_users_own_words():
    from agent.prompt_builder import steer_user_row

    row = steer_user_row("fix the header too")
    projected = _project_for_display([row])[0]

    # Same words session.resume shows; the model-facing marker stays in the source.
    assert projected["display_content"] == "fix the header too"
    assert _history_to_messages([row])[0]["text"] == "fix the header too"
    assert row["content"].startswith("[OUT-OF-BAND USER MESSAGE")


def test_history_applies_live_stripping_and_redaction_without_changing_raw_items(
    monkeypatch,
):
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: True)
    raw = (
        "Start. <think>private scratchpad</think> Key=" + "sk-" + "demo0123456789abcdef"
    )
    row = _row(raw)

    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_commentary"] == [visible_commentary(raw)]
        assert "<think>" not in projected["display_commentary"][0]
        assert "sk-demo0123456789abcdef" not in projected["display_commentary"][0]
        assert projected["display_reasoning"] == "Private summary."
        assert projected["codex_message_items"] == row["codex_message_items"]


@pytest.mark.parametrize(
    "disabled",
    [
        {"show_commentary": False},
        {"show_commentary": "false"},
        {"interim_assistant_messages": False},
    ],
)
def test_owner_profile_settings_apply_to_rest_and_rpc(monkeypatch, tmp_path, disabled):
    from hermes_cli import config as config_mod

    enabled_home, disabled_home = tmp_path / "enabled", tmp_path / "disabled"
    observed = []

    def config():
        home = get_hermes_home_override()
        observed.append(str(home))
        return {"display": {} if str(home) == str(enabled_home) else disabled}

    monkeypatch.setattr(config_mod, "load_config", config)
    row = _row()
    for adapter in (
        lambda home: _project_for_display([row], home=home)[0],
        lambda home: _history_to_messages([row], profile_home=home)[0],
    ):
        assert adapter(enabled_home)["display_commentary"] == [
            "I will inspect the files."
        ]
        hidden = adapter(disabled_home)
        assert hidden["display_commentary"] == []
        assert hidden["display_reasoning"] == "Private summary."
        assert hidden["reasoning"] == row["reasoning"]
        assert adapter(enabled_home)["display_reasoning"] == "Private summary."
    assert observed == [str(enabled_home), str(disabled_home), str(enabled_home)] * 2


def test_rest_pages_bind_the_history_owner_for_messages_and_around(
    monkeypatch, tmp_path
):
    from hermes_cli import config as config_mod
    from hermes_cli.web_routers import sessions
    import hermes_state_timeline

    row = _row()
    homes = {name: tmp_path / name for name in ("visible", "hidden")}
    monkeypatch.setattr(
        sessions, "_cron_profile_home", lambda profile: (profile, homes[profile])
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "display": {
                "show_commentary": str(get_hermes_home_override())
                == str(homes["visible"])
            }
        },
    )

    class FakeDB:
        def resolve_session_id(self, sid):
            return sid

        def resolve_resume_session_id(self, sid):
            return sid

        def get_messages(self, *args, **kwargs):
            return [row]

    monkeypatch.setattr(
        sessions, "_with_db", lambda profile, fn, read_only: fn(FakeDB())
    )
    monkeypatch.setattr(sessions, "_timeline_session_id", lambda db, sid, owner: sid)
    monkeypatch.setattr(
        hermes_state_timeline,
        "get_session_messages_around",
        lambda *args, **kwargs: {"messages": [row], "pagination": {}},
    )

    async def exercise():
        for profile, expected in (
            ("visible", ["I will inspect the files."]),
            ("hidden", []),
        ):
            page = await sessions.get_session_messages(
                "sid",
                profile=profile,
                limit=120,
                offset=0,
                order="latest",
                include_compacted=True,
            )
            around = await sessions.get_session_messages_around(
                "sid", row_id=1, profile=profile, limit=120
            )
            assert page["messages"][0]["display_commentary"] == expected
            assert around["messages"][0]["display_commentary"] == expected
            assert page["profile"] == around["profile"] == profile

    asyncio.run(exercise())


def test_unscoped_rest_history_uses_custom_home_of_its_database(monkeypatch, tmp_path):
    from hermes_cli import config as config_mod
    from hermes_cli.web_routers import sessions
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    import hermes_state_timeline

    custom, default = tmp_path / "custom", tmp_path / "default"
    monkeypatch.setattr(
        sessions, "_cron_profile_home", lambda profile: ("default", default)
    )
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "display": {
                "show_commentary": str(get_hermes_home_override()) != str(custom)
            }
        },
    )
    row = _row()

    class FakeDB:
        def resolve_session_id(self, sid):
            return sid

        def resolve_resume_session_id(self, sid):
            return sid

        def get_messages(self, *args, **kwargs):
            return [row]

    monkeypatch.setattr(
        sessions, "_with_db", lambda profile, fn, read_only: fn(FakeDB())
    )
    monkeypatch.setattr(sessions, "_timeline_session_id", lambda db, sid, owner: sid)
    monkeypatch.setattr(
        hermes_state_timeline,
        "get_session_messages_around",
        lambda *args, **kwargs: {"messages": [row], "pagination": {}},
    )

    async def exercise():
        page = await sessions.get_session_messages(
            "sid",
            profile=None,
            limit=120,
            offset=0,
            order="latest",
            include_compacted=True,
        )
        around = await sessions.get_session_messages_around(
            "sid",
            row_id=1,
            profile=None,
            limit=120,
        )
        assert page["messages"][0]["display_commentary"] == []
        assert around["messages"][0]["display_commentary"] == []

    token = set_hermes_home_override(str(custom))
    try:
        asyncio.run(exercise())
    finally:
        reset_hermes_home_override(token)


def test_cold_resume_uses_the_resumed_session_home(monkeypatch, tmp_path):
    from hermes_cli import config as config_mod
    from tui_gateway.server import _Resume

    home = tmp_path / "disabled"
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "display": {"show_commentary": str(get_hermes_home_override()) != str(home)}
        },
    )
    resume = _Resume.__new__(_Resume)
    resume.omit_messages = False
    resume.profile_home = home

    assert resume.messages([_row()])[0]["display_commentary"] == []


def test_joined_canonical_commentary_uses_sanitized_display_copy(monkeypatch):
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: True)
    raw = "Checking. Key=" + "sk-" + "demo0123456789abcdef"
    row = _row(raw)
    row["content"] = raw

    visible = _project_for_display([row])[0]
    assert visible["display_content"] == visible_commentary(raw)
    assert visible["content"] == raw
    assert visible["display_reasoning"] == "Private summary."

    from agent.history_commentary import project_history_commentary
    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": False}}
    )
    hidden = project_history_commentary([row])[0]
    assert hidden["display_content"] == visible_commentary(raw)
    assert hidden["display_commentary"] == []


def test_only_exact_delimiter_bounded_copy_is_removed():
    row = _row("Same phrase")
    row["reasoning"] = "Summary mentioning Same phrase as an incidental substring."
    projected = _project_for_display([row])[0]
    assert projected["display_reasoning"] == row["reasoning"]

    row["reasoning"] = "Before.\n\nSame phrase\n\nAfter."
    assert _project_for_display([row])[0]["display_reasoning"] == "Before.\n\nAfter."


def test_foreign_and_malformed_items_are_not_public():
    row = _row()
    row["codex_message_items"][0]["role"] = "user"
    assert "display_commentary" not in _project_for_display([row])[0]
    row["codex_message_items"] = "{invalid JSON"
    assert "display_commentary" not in _project_for_display([row])[0]


def test_mixed_canonical_content_is_sanitized_without_truncating_possible_final(
    monkeypatch,
):
    raw = "Checking. <think>hidden</think> key=" + "sk-" + "demo0123456789abcdef"
    row = _row(raw)
    row["content"] = raw + "\n\nFinal result."
    monkeypatch.setattr("agent.redact._redact_enabled", lambda: True)

    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_commentary"] == []
        assert projected["display_content"] == visible_commentary(row["content"])
        assert raw not in projected["display_reasoning"]

    from hermes_cli import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": False}}
    )
    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_commentary"] == []
        assert projected["display_content"] == visible_commentary(row["content"])
    assert row["content"].startswith(raw)


@pytest.mark.parametrize("final", ["Checking. The answer is 42.", "Checking."])
def test_stream_recovered_final_without_final_sidecar_is_authoritative(
    monkeypatch, final
):
    from types import SimpleNamespace
    from agent.turn_finalizer import _close_transcript_tail
    from hermes_cli import config as config_mod

    row = _row("Checking.")
    agent = SimpleNamespace(_db_flush_scan_prefix=None)
    _close_transcript_tail(agent, [row], final, False, True)
    assert row["content"] == final
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert (
                projected.get(
                    "display_content", projected.get("content", projected.get("text"))
                )
                == final
            )


@pytest.mark.parametrize("final", ["Checking. The answer is 42.", "Checking."])
@pytest.mark.parametrize("phase", ["final", "final_answer", None])
def test_canonical_content_matching_final_item_is_never_stripped(
    monkeypatch, final, phase
):
    from hermes_cli import config as config_mod

    row = _row("Checking.")
    row["content"] = final
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": final}],
    }
    if phase is not None:
        item["phase"] = phase
    row["codex_message_items"].append(item)
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert (
                projected.get(
                    "display_content", projected.get("content", projected.get("text"))
                )
                == final
            )
            assert projected["display_commentary"] == (["Checking."] if visible else [])


def test_mixed_canonical_preserves_final_repetition_of_a_commentary_phrase(monkeypatch):
    from hermes_cli import config as config_mod

    row = _row("Checking")
    row["content"] = "Checking\n\nFinal: Checking completed."
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert projected["display_content"] == row["content"]
            assert projected["display_commentary"] == []

    row["codex_message_items"].append({
        "type": "message",
        "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "Second update"}],
    })
    row["content"] = "Checking\n\nSecond update\n\nFinal: Checking completed."
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": True}}
    )
    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_content"] == row["content"]
        assert projected["display_commentary"] == []


def test_legacy_newline_and_ambiguous_repeats_do_not_leak_public_text_into_thinking():
    row = _row("Public update.")
    row["reasoning"] = "Private before.\nPublic update.\nPrivate after."
    assert (
        _project_for_display([row])[0]["display_reasoning"]
        == "Private before.\nPrivate after."
    )

    row["reasoning"] = (
        "Private before.\n\nPublic update.\n\nPrivate after.\n\nPublic update."
    )
    assert (
        _project_for_display([row])[0]["display_reasoning"]
        == "Private before.\n\nPrivate after."
    )
    assert row["reasoning"].endswith("Public update.")


def test_overlapping_public_items_do_not_leave_partial_text_in_final_or_thinking(
    monkeypatch,
):
    from hermes_cli import config as config_mod

    row = _row("Start")
    row["codex_message_items"].insert(
        1,
        {
            "type": "message",
            "role": "assistant",
            "phase": "commentary",
            "content": [{"type": "output_text", "text": "Start\nTail public"}],
        },
    )
    row["reasoning"] = "Private.\n\nStart\nTail public"
    row["content"] = "Start\nTail public\n\nFinal answer."

    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert projected["display_content"] == row["content"]
            assert projected["display_reasoning"] == "Private."
            assert projected["display_commentary"] == []

    # Without an attributable leading commentary span, this belongs to the final.
    row["content"] = "Leading Start\nTail public trailing"
    for visible in (True, False):
        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda visible=visible: {"display": {"show_commentary": visible}},
        )
        for projected in (
            _project_for_display([row])[0],
            _history_to_messages([row])[0],
        ):
            assert projected["display_content"] == "Leading Start\nTail public trailing"
            assert projected["display_commentary"] == (
                ["Start", "Start\nTail public"] if visible else []
            )

    row["content"] = "Start\nTail publicFinal answer."
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"display": {"show_commentary": True}}
    )
    for projected in (_project_for_display([row])[0], _history_to_messages([row])[0]):
        assert projected["display_content"] == row["content"]
        assert projected["display_commentary"] == []


def test_history_commentary_keeps_whitespace_part_before_memory_fence():
    """Replayed commentary is assembled like live commentary: a newline-only part keeps an
    echoed <memory-context> block on its own line, so the fence drops it (#81312)."""
    row = _row()
    row["codex_message_items"][0]["content"] = [
        {"type": "output_text", "text": "Checking the inbox."},
        {"type": "output_text", "text": "\n"},
        {"type": "output_text", "text": "<memory-context>\nSENTINEL_note\n</memory-context>"},
    ]
    projected = _project_for_display([row])[0]

    assert projected["display_commentary"] == ["Checking the inbox."]


def test_stale_memory_context_module_self_heals_on_lazy_import():
    """If a long-lived serve process has a pre-c6f6de65b4 agent.memory_context cached in
    sys.modules without fence_recalled_context, visible_commentary and fence_result_reply
    reload agent.memory_context in-place instead of crashing with ImportError."""
    import agent.memory_context as mc_mod
    from agent.turn_finalizer import fence_result_reply

    delattr(mc_mod, "fence_recalled_context")
    assert not hasattr(mc_mod, "fence_recalled_context")
    assert visible_commentary("Hello\n<memory-context>\nsecret\n</memory-context>") == "Hello"
    assert hasattr(mc_mod, "fence_recalled_context")

    delattr(mc_mod, "fence_recalled_context")
    assert not hasattr(mc_mod, "fence_recalled_context")
    res = {"final_response": "Reply\n<memory-context>\nsecret\n</memory-context>"}
    fence_result_reply(res)
    assert res["final_response"] == "Reply"
    assert hasattr(mc_mod, "fence_recalled_context")


_OCC_PLUGIN_PATH = (
    __import__("pathlib").Path.home() / ".hermes" / "plugins" / "or-credit-clamp" / "__init__.py"
)
_OCC_PLUGIN_SOURCE = (
    _OCC_PLUGIN_PATH.read_text(encoding="utf-8") if _OCC_PLUGIN_PATH.is_file() else None
)


def _load_or_credit_clamp_plugin():
    import types

    if _OCC_PLUGIN_SOURCE is None:
        pytest.skip("or-credit-clamp plugin not installed in ~/.hermes/plugins")
    assert isinstance(_OCC_PLUGIN_SOURCE, str)
    mod = types.ModuleType("_occ_test_mod")
    exec(compile(_OCC_PLUGIN_SOURCE, "<or_credit_clamp_plugin>", "exec"), mod.__dict__)
    return mod


def test_or_credit_clamp_missing_disk_defs_two_way_polarity(tmp_path):
    """_module_missing_disk_defs detects missing Assign/tuple-Assign/AnnAssign/ImportFrom/Import
    symbols and new ClassDef methods on disk (positive polarity) and returns False when all are
    present in memory (negative polarity)."""
    import importlib.util
    import sys

    occ = _load_or_credit_clamp_plugin()
    mod_file = tmp_path / "sample_defs_mod.py"
    mod_file.write_text(
        "import json as _json_alias\n"
        "from os import path as _os_path\n"
        "PLAIN_CONST = 1\n"
        "TUP_A, TUP_B = 2, 3\n"
        "TYPED_CONST: int = 4\n"
        "class Svc:\n"
        "    def existing_m(self):\n"
        "        return 1\n"
        "    def new_m(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("sample_defs_mod", mod_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sample_defs_mod"] = mod
    try:
        spec.loader.exec_module(mod)
        # Negative polarity: all disk definitions are present on the loaded module
        assert occ._module_missing_disk_defs(mod) is False

        # Positive polarity across each AST construct
        for attr in ("PLAIN_CONST", "TUP_B", "TYPED_CONST", "_os_path", "_json_alias"):
            saved = getattr(mod, attr)
            delattr(mod, attr)
            if hasattr(mod, "_or_credit_clamp_defs_sig"):
                delattr(mod, "_or_credit_clamp_defs_sig")
            assert occ._module_missing_disk_defs(mod) is True, f"missed missing {attr}"
            setattr(mod, attr, saved)

        # Positive polarity for a new method added to an existing class
        delattr(mod.Svc, "new_m")
        if hasattr(mod, "_or_credit_clamp_defs_sig"):
            delattr(mod, "_or_credit_clamp_defs_sig")
        assert occ._module_missing_disk_defs(mod) is True
    finally:
        sys.modules.pop("sample_defs_mod", None)


def test_or_credit_clamp_reload_preserves_contextvar_locks_and_caches(tmp_path, monkeypatch):
    """_reload_if_changed preserves ContextVar identity (so in-flight tokens reset cleanly),
    Lock/RLock identity (top-level and inside tuples/dicts), and populated runtime dict/set caches."""
    import importlib
    import sys

    occ = _load_or_credit_clamp_plugin()
    monkeypatch.syspath_prepend(str(tmp_path))
    mod_file = tmp_path / "sample_stateful_mod.py"
    mod_file.write_text(
        "import contextvars, threading\n"
        "_ACTIVE_CV = contextvars.ContextVar('_ACTIVE_CV', default='idle')\n"
        "_MOD_LOCK = threading.Lock()\n"
        "_MOD_RLOCK = threading.RLock()\n"
        "_TUPLE_LOCKS = (threading.Lock(),)\n"
        "_DICT_LOCKS = {'lk': threading.Lock()}\n"
        "_RUNTIME_CACHE = {}\n"
        "_RUNTIME_SET = set()\n"
        "def version():\n"
        "    return 'v1'\n",
        encoding="utf-8",
    )
    importlib.invalidate_caches()
    mod = importlib.import_module("sample_stateful_mod")
    try:
        orig_cv = mod._ACTIVE_CV
        orig_lock = mod._MOD_LOCK
        orig_rlock = mod._MOD_RLOCK
        orig_tup_lock = mod._TUPLE_LOCKS[0]
        orig_dict_lock = mod._DICT_LOCKS["lk"]
        mod._RUNTIME_CACHE["k"] = "live"
        mod._RUNTIME_SET.add("item")
        token = orig_cv.set("in_flight")

        mod_file.write_text(
            mod_file.read_text(encoding="utf-8").replace("return 'v1'", "return 'v2'"),
            encoding="utf-8",
        )
        assert occ._reload_if_changed(mod, force=True) is True
        assert mod.version() == "v2"
        assert mod._ACTIVE_CV is orig_cv
        assert mod._MOD_LOCK is orig_lock
        assert mod._MOD_RLOCK is orig_rlock
        assert mod._TUPLE_LOCKS[0] is orig_tup_lock
        assert mod._DICT_LOCKS["lk"] is orig_dict_lock
        assert mod._RUNTIME_CACHE == {"k": "live"}
        assert mod._RUNTIME_SET == {"item"}
        # Token created before reload must reset cleanly without ValueError
        mod._ACTIVE_CV.reset(token)
        assert mod._ACTIVE_CV.get() == "idle"
    finally:
        sys.modules.pop("sample_stateful_mod", None)


def test_or_credit_clamp_topological_sort_handles_pkg_and_relative_imports(tmp_path):
    """_topological_sort_modules resolves `from pkg import sub_b`, relative imports
    (`from . import sub_b`, `from .sub_b import X`), and guarantees zero inversions
    across _HOTFIX_TOPOLOGICAL_MODULES."""
    import importlib.util
    import sys

    occ = _load_or_credit_clamp_plugin()
    pkg_dir = tmp_path / "topo_pkg"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    (pkg_dir / "sub_b.py").write_text("VAL = 42\n", encoding="utf-8")
    (pkg_dir / "caller_a.py").write_text("from topo_pkg import sub_b\n", encoding="utf-8")
    (pkg_dir / "rel_c.py").write_text("from . import sub_b\nfrom .sub_b import VAL\n", encoding="utf-8")

    created = []
    try:
        for mname, mpath in (
            ("topo_pkg", pkg_dir / "__init__.py"),
            ("topo_pkg.sub_b", pkg_dir / "sub_b.py"),
            ("topo_pkg.caller_a", pkg_dir / "caller_a.py"),
            ("topo_pkg.rel_c", pkg_dir / "rel_c.py"),
        ):
            spec = importlib.util.spec_from_file_location(mname, mpath)
            assert spec is not None and spec.loader is not None
            m = importlib.util.module_from_spec(spec)
            sys.modules[mname] = m
            created.append(mname)

        ordered = occ._topological_sort_modules(
            ["topo_pkg.caller_a", "topo_pkg.rel_c", "topo_pkg.sub_b"]
        )
        assert ordered.index("topo_pkg.sub_b") < ordered.index("topo_pkg.caller_a")
        assert ordered.index("topo_pkg.sub_b") < ordered.index("topo_pkg.rel_c")
    finally:
        for mname in created:
            sys.modules.pop(mname, None)


def test_relay_shared_metrics_idempotent_register_and_reload_rebind_two_way_polarity(monkeypatch):
    """#CF-89 two-way polarity:
    1. _Runtime.__init__ deregisters and re-registers cleanly when native Relay.subscribers.register
       raises RuntimeError('... subscriber already exists'), and still re-raises unrelated RuntimeErrors.
    2. _get_runtime rebinds a pre-reload _Runtime instance across class redefinition without re-running
       _Runtime.__init__.
    """
    from typing import Any
    import hermes_cli.observability.relay_shared_metrics as rsm

    calls = []

    class _FakeSubscribers:
        def __init__(self, fail_mode: str):
            self.fail_mode = fail_mode
            self.calls = 0

        def register(self, name, sub):
            self.calls += 1
            calls.append(("register", name, self.calls))
            if self.fail_mode == "already_exists_once" and self.calls == 1:
                raise RuntimeError(f"already exists: {name} subscriber already exists")
            if self.fail_mode == "other_error":
                raise RuntimeError("fatal relay failure")

        def deregister(self, name):
            calls.append(("deregister", name, self.calls))

    class _FakeHost:
        runtime_id = "rt_test_cf89"
        relay = type("_R", (), {"subscribers": _FakeSubscribers("already_exists_once")})()

        def retain_managed_execution(self, name):
            calls.append(("retain", name))

        def release_managed_execution(self, name):
            calls.append(("release", name))

    monkeypatch.setattr(rsm.atexit, "register", lambda fn: None)
    fake_host: Any = _FakeHost()
    # Positive polarity: 'already exists' triggers deregister + second register without raising
    rt = rsm._Runtime(host=fake_host)
    assert rt._registered is True
    assert [c[0] for c in calls] == ["register", "deregister", "register", "retain"]

    # Negative polarity: unrelated RuntimeError still raises
    class _BadHost(_FakeHost):
        relay = type("_R", (), {"subscribers": _FakeSubscribers("other_error")})()

    bad_host: Any = _BadHost()
    with pytest.raises(RuntimeError, match="fatal relay failure"):
        rsm._Runtime(host=bad_host)

    # Class-identity rebind across reload: pre-reload _Runtime subclass/copy in _RUNTIMES is rebound in-place
    OldRuntime = type("_Runtime", (), {})
    old_inst: Any = OldRuntime()
    old_inst.host = fake_host
    old_inst._active = True
    monkeypatch.setattr(rsm.relay_runtime, "current_profile_key", lambda: "prof_cf89")
    monkeypatch.setitem(rsm._RUNTIMES, "prof_cf89", old_inst)
    got = rsm._get_runtime(retry_failed=True, host=fake_host)
    assert got is old_inst
    assert isinstance(got, rsm._Runtime)

