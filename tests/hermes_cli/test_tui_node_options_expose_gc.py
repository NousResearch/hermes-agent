"""Regression tests for #17187.

Node refuses to start when ``--expose-gc`` appears in ``NODE_OPTIONS``
(it is a V8 flag, not whitelisted for env-var injection); the user sees
``--expose-gc is not allowed in NODE_OPTIONS`` and the TUI never boots.

The exposed-GC behaviour must instead be passed as a CLI flag to the node
binary inside ``_make_tui_argv`` for node-direct invocations.
"""

from pathlib import Path

import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as mod

    return mod


def test_launch_tui_does_not_inject_expose_gc_into_node_options(monkeypatch, main_mod):
    """``--expose-gc`` in NODE_OPTIONS makes node refuse to start (#17187)."""

    monkeypatch.delenv("NODE_OPTIONS", raising=False)

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "--expose-gc", "dist/entry.js"], Path(".")),
    )

    captured: dict = {}

    def fake_call(argv, cwd=None, env=None):
        captured["env"] = dict(env)
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)

    with pytest.raises(SystemExit):
        main_mod._launch_tui()

    node_options = captured["env"].get("NODE_OPTIONS", "")
    tokens = node_options.split()
    assert "--expose-gc" not in tokens, (
        f"--expose-gc must not be injected into NODE_OPTIONS (#17187); "
        f"got NODE_OPTIONS={node_options!r}"
    )
    assert "--max-old-space-size=8192" in tokens, (
        "max-old-space-size cap must still be set in NODE_OPTIONS to avoid "
        f"V8 OOM on long sessions; got NODE_OPTIONS={node_options!r}"
    )


def test_launch_tui_preserves_user_node_options_without_appending_expose_gc(
    monkeypatch, main_mod
):
    """A user with their own NODE_OPTIONS must not have --expose-gc tacked on."""

    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=12288 --inspect")

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "--expose-gc", "dist/entry.js"], Path(".")),
    )

    captured: dict = {}

    def fake_call(argv, cwd=None, env=None):
        captured["env"] = dict(env)
        return 0

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)

    with pytest.raises(SystemExit):
        main_mod._launch_tui()

    tokens = captured["env"]["NODE_OPTIONS"].split()
    assert "--expose-gc" not in tokens
    assert "--max-old-space-size=12288" in tokens
    assert "--inspect" in tokens
    # Did not double-cap when user already set their own size.
    assert not any(
        t.startswith("--max-old-space-size=") and t != "--max-old-space-size=12288"
        for t in tokens
    )


def test_make_tui_argv_external_dir_passes_expose_gc_as_cli_flag(
    monkeypatch, main_mod, tmp_path
):
    """HERMES_TUI_DIR fast path must inject --expose-gc into node argv (#17187)."""

    ext = tmp_path / "tui-dist"
    (ext / "dist").mkdir(parents=True)
    (ext / "dist" / "entry.js").write_text("// stub", encoding="utf-8")

    fake_node = tmp_path / "node"
    fake_node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_node.chmod(0o755)

    monkeypatch.setenv("HERMES_TUI_DIR", str(ext))
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda p: False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda b: str(fake_node) if b == "node" else None)

    argv, cwd = main_mod._make_tui_argv(tmp_path / "ui-tui", tui_dev=False)

    assert argv[0] == str(fake_node)
    assert "--expose-gc" in argv, (
        "--expose-gc must be passed on the node CLI for node-direct TUI launches; "
        f"got argv={argv}"
    )
    # Flag goes before the script so V8 sees it; otherwise Node treats it as a
    # script argument.
    expose_gc_idx = argv.index("--expose-gc")
    script_idx = argv.index(str(ext / "dist" / "entry.js"))
    assert expose_gc_idx < script_idx
    assert cwd == ext


def test_make_tui_argv_built_dist_passes_expose_gc_as_cli_flag(
    monkeypatch, main_mod, tmp_path
):
    """The built-bundle path also needs --expose-gc on the node argv (#17187)."""

    tui_dir = tmp_path / "ui-tui"
    (tui_dir / "node_modules").mkdir(parents=True)

    fake_node = tmp_path / "node"
    fake_node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_node.chmod(0o755)
    fake_npm = tmp_path / "npm"
    fake_npm.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_npm.chmod(0o755)

    bundled_root = tui_dir / "bundled"
    (bundled_root / "dist").mkdir(parents=True)
    (bundled_root / "dist" / "entry.js").write_text("// stub", encoding="utf-8")

    def _fake_which(b):
        if b == "node":
            return str(fake_node)
        if b == "npm":
            return str(fake_npm)
        return None

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda p: False)
    monkeypatch.setattr(main_mod, "_tui_build_needed", lambda p: False)
    monkeypatch.setattr(main_mod, "_find_bundled_tui", lambda p: bundled_root)
    monkeypatch.setattr(main_mod.shutil, "which", _fake_which)

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=False)

    assert argv[0] == str(fake_node)
    assert "--expose-gc" in argv
    expose_gc_idx = argv.index("--expose-gc")
    script_idx = argv.index(str(bundled_root / "dist" / "entry.js"))
    assert expose_gc_idx < script_idx
    assert cwd == bundled_root
