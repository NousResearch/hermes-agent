"""Phase 6 hardening: standalone entrypoints mirror HERMES_*/HT_* env vars.

The CLI (``hermes_cli.main``) mirrors the brand env namespaces at startup, but
the other entrypoints — the ``hermes-agent`` (``run_agent:main``) and
``hermes-acp`` (``acp_adapter.entry:main``) console scripts, and
``python batch_runner.py`` — can be launched standalone (not as a child of the
mirrored CLI). Each wires ``mirror_brand_env`` with the same guarded pattern so
an ``HT_*``-only environment still works when they run directly.

``acp_adapter.entry.main(["--version"])`` returns immediately after the mirror,
so it exercises the wiring without starting the ACP server. The other two use
the identical pattern (run_agent at the top of ``main``; batch_runner in the
``__main__`` block), verified by inspection — they cannot be driven to an early
return as cheaply.
"""

import os


def test_acp_entry_mirrors_env_before_early_return(monkeypatch, capsys):
    import acp_adapter.entry as entry

    monkeypatch.delenv("HT_PHASE6_ENTRY_PROBE", raising=False)
    monkeypatch.setenv("HERMES_PHASE6_ENTRY_PROBE", "yes")
    # Snapshot pre-existing HT_* keys so we can strip only what the entry adds.
    ht_before = {k for k in os.environ if k.startswith("HT_")}
    try:
        # --version mirrors env, prints the version, and returns — no server.
        entry.main(["--version"])
        # The HERMES_* var was mirrored to its HT_* counterpart at entry.
        assert os.environ.get("HT_PHASE6_ENTRY_PROBE") == "yes"
    finally:
        # mirror_brand_env mutates os.environ directly, outside monkeypatch's
        # tracking, and mirrors every ambient HERMES_* — remove every HT_* key
        # it created so nothing leaks into and pollutes later tests.
        for key in [k for k in os.environ if k.startswith("HT_") and k not in ht_before]:
            os.environ.pop(key, None)

    # Sanity: we actually hit the version early-return path.
    assert capsys.readouterr().out.strip()


def _called_function_names(obj) -> set:
    """Names of every function actually *called* in obj's source.

    AST ``Call`` nodes only — a name that survives merely in an import line,
    a comment, or a docstring does NOT count, unlike the substring check this
    replaces (which passed even when the call itself was deleted)."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_entrypoints_call_mirror_and_migration():
    """Every standalone entrypoint must CALL mirror_brand_env (both env-name
    spellings work when launched directly) and maybe_migrate_home (the
    ~/.hermes -> ~/.ht-ai-agent bridge symlink exists no matter which entry
    launches first — the CLI alone running it is not enough). The acp
    behavioural test above exercises the mirror at runtime; run_agent's and
    batch_runner's mains are too heavy to drive here, so their wiring is
    pinned at the AST level (a deleted call fails; a leftover import cannot
    pass)."""
    import acp_adapter.entry as acp_entry
    import batch_runner
    import run_agent

    for entry in (run_agent.main, acp_entry.main):
        called = _called_function_names(entry)
        assert "mirror_brand_env" in called
        assert "maybe_migrate_home" in called
    # batch_runner wires both in the module __main__ block.
    called = _called_function_names(batch_runner)
    assert "mirror_brand_env" in called
    assert "maybe_migrate_home" in called


def test_acp_entry_provisions_home_bridge_in_fresh_home(tmp_path):
    """End-to-end guard for the entrypoint migration wiring: a standalone
    `hermes-acp --version` in a fresh HOME (outside pytest, which suppresses
    migration in-process) must provision ~/.ht-ai-agent AND the ~/.hermes
    bridge symlink — without the bridge, the hardcoded legacy fallbacks fork
    a second data dir."""
    import subprocess
    import sys
    from pathlib import Path

    scratch = tmp_path / "fresh-home"
    scratch.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HERMES_HOME", "HT_HOME", "PYTEST_CURRENT_TEST", "HT_SKIP_HOME_MIGRATION")
    }
    env["HOME"] = str(scratch)
    proc = subprocess.run(
        [sys.executable, "-c", "import acp_adapter.entry as e; e.main(['--version'])"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    new = scratch / ".ht-ai-agent"
    legacy = scratch / ".hermes"
    assert new.is_dir()
    assert legacy.is_symlink() and legacy.resolve() == new.resolve()
