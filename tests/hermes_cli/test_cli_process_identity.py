"""CLI process-identity regression coverage for #86204."""


def test_interactive_cli_attaches_its_windows_child_tree(monkeypatch):
    """A crashed classic CLI must take its Python child tree down with it."""
    from hermes_cli import process_identity
    from hermes_cli.cli_process_identity import register_interactive_cli

    calls = []
    monkeypatch.setattr(process_identity, "register_self", lambda purpose: calls.append(("register", purpose)))
    monkeypatch.setattr(process_identity, "attach_self_to_kill_on_close_job", lambda: calls.append(("attach",)))

    register_interactive_cli()

    assert calls == [("register", "cli"), ("attach",)]
