"""Plain-language copy for surfaces that cannot start because an optional dependency group is missing.

``hermes dashboard`` (fastapi + uvicorn) and ``hermes acp`` (agent-client-protocol) are installed by
the ``[all]`` / ``[acp]`` extras. A partial install, a pip-less uv venv or an interrupted update can
leave them out; the built-in repair is ``hermes update``, which reinstalls the extras into the same
interpreter. The manual fallback names the checkout directory and interpreter explicitly, via
``managed_pip_install_command`` — Hermes' managed uv (which is off PATH, so a bare ``uv`` would not
resolve) when present, else the interpreter's own pip (a bare ``pip install`` fails on PEP 668 and
in venvs without pip).
"""


def missing_optional_deps_message(surface: str, what: str, extra: str) -> str:
    """``surface`` = "dashboard"/"ACP server"; ``what`` = "its web-server packages"; ``extra`` = "all"/"acp"."""
    from hermes_cli.main import PROJECT_ROOT
    from hermes_cli.managed_uv import managed_pip_install_command

    editable_spec = f"'.[{extra}]'"
    return (
        f"The {surface} can't start: {what} are missing from this install.\n"
        "Run `hermes update` to reinstall dependencies. If that fails, run manually:\n"
        f"  cd {PROJECT_ROOT} && {managed_pip_install_command('-e', editable_spec)}\n"
    )
