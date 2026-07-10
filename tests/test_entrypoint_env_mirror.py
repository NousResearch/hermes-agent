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


def test_entrypoints_use_the_shared_mirror_helper():
    """Guard that the entry modules reference the mirror helper, so the wiring
    isn't silently dropped in a refactor (the acp behavioural test covers one
    path; this keeps the other two honest without driving their heavy mains)."""
    import inspect

    import batch_runner
    import run_agent

    assert "mirror_brand_env" in inspect.getsource(run_agent.main)
    # batch_runner wires it in the module __main__ block.
    assert "mirror_brand_env" in inspect.getsource(batch_runner)


def test_entrypoints_wire_home_migration():
    """Every standalone entrypoint must also run maybe_migrate_home(): the
    ~/.hermes -> ~/.ht-ai-agent bridge symlink has to exist no matter which
    entry launches first, or the hardcoded legacy fallbacks fork a second data
    dir on fresh installs (the CLI alone running it is not enough)."""
    import inspect

    import acp_adapter.entry as acp_entry
    import batch_runner
    import run_agent

    assert "maybe_migrate_home" in inspect.getsource(run_agent.main)
    assert "maybe_migrate_home" in inspect.getsource(acp_entry.main)
    assert "maybe_migrate_home" in inspect.getsource(batch_runner)
