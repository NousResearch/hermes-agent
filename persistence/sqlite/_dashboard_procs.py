"""The Desktop backend argv test the orphan holder reaper runs, as ``hermes_cli.dashboard_procs`` runs it."""


def _dashboard_subcommand_index(argv: list[str]) -> int | None:
    return next((i for i, tok in enumerate(argv) if tok in ("serve", "dashboard")), None)


def _is_ephemeral_port_zero_backend(argv: list[str]) -> bool:
    """True for Desktop-style ``serve|dashboard --port 0`` backends."""
    if _dashboard_subcommand_index(argv) is None:
        return False
    return any((tok == "--port" and i + 1 < len(argv) and str(argv[i + 1]) == "0")
               or (tok.startswith("--port=") and tok.split("=", 1)[1].strip() == "0")
               for i, tok in enumerate(argv))
