"""Regression tests for routed-profile terminal configuration scope (#68559)."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from unittest import mock

import pytest


_TERMINAL_ENV_PREFIX = "TERMINAL_"


@pytest.fixture(autouse=True)
def _reset_terminal_scope_and_environment():
    from tools import terminal_tool

    original_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_TERMINAL_ENV_PREFIX)
    }
    for key in list(original_env):
        os.environ.pop(key, None)
    bridge_attempted = terminal_tool._terminal_config_bridge_attempted
    terminal_tool._terminal_config_bridge_attempted = False
    scope_setter = getattr(terminal_tool, "set_terminal_config_scope", None)
    token = scope_setter(None) if scope_setter is not None else None
    try:
        yield
    finally:
        if token is not None:
            terminal_tool.reset_terminal_config_scope(token)
        terminal_tool._terminal_config_bridge_attempted = bridge_attempted
        for key in list(os.environ):
            if key.startswith(_TERMINAL_ENV_PREFIX):
                os.environ.pop(key, None)
        os.environ.update(original_env)


def _write_profile(home: Path, terminal: dict | None = None) -> None:
    home.mkdir()
    config = {} if terminal is None else {"terminal": terminal}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


def _gateway_and_routed_homes(tmp_path, gateway_terminal, routed_terminal=None):
    gateway_home = tmp_path / "gateway"
    routed_home = tmp_path / "routed"
    _write_profile(gateway_home, gateway_terminal)
    _write_profile(routed_home, routed_terminal)
    return gateway_home, routed_home


def _seed_gateway_terminal_config(monkeypatch, gateway_home):
    from tools import terminal_tool

    monkeypatch.setenv("HERMES_HOME", str(gateway_home))
    gateway_config = terminal_tool._get_env_config()
    assert terminal_tool._terminal_config_bridge_attempted is True
    return gateway_config


def test_routed_profile_uses_its_terminal_backend(tmp_path, monkeypatch):
    """A routed docker profile must not inherit the gateway's local backend."""
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        {"backend": "docker", "docker_image": "routed-image"},
    )
    gateway_config = _seed_gateway_terminal_config(monkeypatch, gateway_home)
    assert gateway_config["env_type"] == "local"

    with _profile_runtime_scope(routed_home):
        routed_config = terminal_tool._get_env_config()

    assert routed_config["env_type"] == "docker"
    assert routed_config["docker_image"] == "routed-image"


def test_gateway_terminal_config_is_restored_after_scope(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        {"backend": "docker"},
    )
    before = _seed_gateway_terminal_config(monkeypatch, gateway_home)

    with _profile_runtime_scope(routed_home):
        assert terminal_tool._get_env_config()["env_type"] == "docker"

    assert terminal_tool._get_env_config() == before


def test_routed_local_profile_does_not_inherit_docker_settings(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {
            "backend": "docker",
            "docker_image": "gateway-image",
            "docker_volumes": ["/host:/container"],
            "docker_network": False,
        },
        {"backend": "local"},
    )
    gateway_config = _seed_gateway_terminal_config(monkeypatch, gateway_home)
    assert gateway_config["env_type"] == "docker"
    assert gateway_config["docker_image"] == "gateway-image"

    with _profile_runtime_scope(routed_home):
        routed_config = terminal_tool._get_env_config()

    assert routed_config["env_type"] == "local"
    assert routed_config["docker_image"] != "gateway-image"
    assert routed_config["docker_volumes"] == []
    assert routed_config["docker_network"] is True


def test_single_profile_path_still_uses_process_environment(tmp_path, monkeypatch):
    from tools import terminal_tool

    home = tmp_path / "single"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "37")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    with mock.patch.object(
        terminal_tool,
        "_ensure_terminal_env_bridged",
        wraps=terminal_tool._ensure_terminal_env_bridged,
    ) as bridge:
        config = terminal_tool._get_env_config()

    # The bridge is driven by get_terminal_setting() rather than by a single
    # call at the top of _get_env_config, so every read on the unscoped path
    # goes through it. It is a one-shot latch, so the count is not the
    # contract — that it runs at all on this path is.
    assert bridge.called
    assert bridge.call_args_list == [mock.call()] * bridge.call_count
    assert config["env_type"] == "local"
    assert config["timeout"] == 37
    assert config["cwd"] == str(tmp_path)


def test_routed_profile_path_never_bridges_into_process_environment(
    tmp_path, monkeypatch
):
    """The scoped path must not run the process-global config→env bridge.

    Bridging writes one profile's ``terminal.*`` into ``os.environ``, which is
    the exact mechanism #68559 is about: whichever profile trips the latch
    first defines the backend every later profile inherits.
    """
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        {"backend": "docker", "docker_image": "routed-image"},
    )
    _seed_gateway_terminal_config(monkeypatch, gateway_home)

    with mock.patch.object(
        terminal_tool, "_ensure_terminal_env_bridged"
    ) as bridge:
        with _profile_runtime_scope(routed_home):
            config = terminal_tool._get_env_config()

    bridge.assert_not_called()
    assert config["env_type"] == "docker"


def test_terminal_scope_propagates_through_copy_context(tmp_path):
    from tools import terminal_tool

    scope = {
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_IMAGE": "thread-image",
    }
    token = terminal_tool.set_terminal_config_scope(scope)
    try:
        context = copy_context()
        with mock.patch.object(
            terminal_tool,
            "_ensure_terminal_env_bridged",
            side_effect=AssertionError("active terminal scope used the global bridge"),
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                config = executor.submit(context.run, terminal_tool._get_env_config).result()
    finally:
        terminal_tool.reset_terminal_config_scope(token)

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "thread-image"


def test_missing_terminal_section_fails_open(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        None,
    )
    gateway_config = _seed_gateway_terminal_config(monkeypatch, gateway_home)

    with _profile_runtime_scope(routed_home):
        routed_config = terminal_tool._get_env_config()

    assert routed_config == gateway_config


# --- Guard: no module may reintroduce a process-global TERMINAL_* read -------
#
# Every fix above is a conversion of one direct ``os.environ`` read into
# ``get_terminal_setting``.  Nothing structural stops a later change from
# adding a fresh ``os.getenv("TERMINAL_...")`` next to the converted ones and
# silently reopening the cross-profile leak — the resulting bug is invisible
# in a single-profile test run, which is how the original holes survived.
# This test greps the AST rather than the text so comments and docstrings that
# merely mention the env vars don't register as reads.

# Modules that must resolve terminal settings through the accessor.
_GUARDED_MODULES = (
    "tools/terminal_tool.py",
    "tools/browser_tool.py",
    "tools/env_probe.py",
    "agent/prompt_builder.py",
)

# The only functions allowed to touch the process environment for TERMINAL_*:
# the accessor that defines the policy, and the one-shot config→env bridge it
# delegates to on the unscoped path.
_ENV_READ_ALLOWED_IN = frozenset(
    {"get_terminal_setting", "_ensure_terminal_env_bridged"}
)


def _terminal_env_reads(source: str):
    """Yield (lineno, function, expression) for each direct TERMINAL_* read."""
    import ast

    def _is_terminal_literal(node) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str) and (
            node.value.startswith(_TERMINAL_ENV_PREFIX)
        )

    def _dotted(node) -> str:
        if isinstance(node, ast.Attribute):
            return f"{_dotted(node.value)}.{node.attr}"
        if isinstance(node, ast.Name):
            return node.id
        return "?"

    tree = ast.parse(source)
    # Map every node to its enclosing function so violations name a call site.
    enclosing: dict[int, str] = {}
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                enclosing.setdefault(id(child), parent.name)

    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.Call) and node.args:
            callee = _dotted(node.func)
            if callee in {"os.getenv", "os.environ.get", "getenv"}:
                if _is_terminal_literal(node.args[0]):
                    target = f"{callee}({node.args[0].value!r})"
        elif isinstance(node, ast.Subscript):
            if _dotted(node.value) == "os.environ" and _is_terminal_literal(node.slice):
                target = f"os.environ[{node.slice.value!r}]"
        if target is None:
            continue
        function = enclosing.get(id(node), "<module>")
        if function in _ENV_READ_ALLOWED_IN:
            continue
        yield node.lineno, function, target


def test_no_module_reads_terminal_config_from_process_environment():
    repo_root = Path(__file__).resolve().parents[2]

    violations = []
    for relative in _GUARDED_MODULES:
        path = repo_root / relative
        assert path.exists(), f"guarded module moved or was renamed: {relative}"
        for lineno, function, expression in _terminal_env_reads(
            path.read_text(encoding="utf-8")
        ):
            violations.append(f"{relative}:{lineno} in {function}(): {expression}")

    assert not violations, (
        "TERMINAL_* settings must be read through "
        "tools.terminal_tool.get_terminal_setting() so a multiplexed gateway "
        "resolves them for the routed profile instead of whichever profile "
        "bridged its config into os.environ first (#68559).\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_guard_detects_a_reintroduced_process_environment_read():
    """The guard above is only useful if it actually fires — prove it does."""
    reintroduced = (
        "import os\n"
        "def _is_local_backend():\n"
        "    return os.getenv('TERMINAL_ENV', 'local') == 'local'\n"
    )
    found = list(_terminal_env_reads(reintroduced))
    assert len(found) == 1
    _lineno, function, expression = found[0]
    assert function == "_is_local_backend"
    assert "TERMINAL_ENV" in expression

    # ...and that it stays quiet for the accessor form and for prose mentions.
    compliant = (
        "import os\n"
        "def _is_local_backend():\n"
        '    """Reads os.environ["TERMINAL_ENV"] historically."""\n'
        "    return get_terminal_setting('TERMINAL_ENV', 'local') == 'local'\n"
    )
    assert not list(_terminal_env_reads(compliant))
