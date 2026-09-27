"""Repair guidance for an unavailable declared dependency group."""


def missing_optional_deps_message(surface: str, what: str, extra: str) -> str:
    return (
        f"The {surface} can't start: {what} are missing from this install.\n"
        "Run `hermes pm install` to prepare the declared dependencies.\n"
        "If an installed dependency is damaged, run `hermes pm repair`, then restart Hermes."
    )


def smart_app_control_block_message(error: BaseException) -> "str | None":
    """Guidance for Windows Smart App Control / Application Control blocking the embedded
    Python runtime's ``_ssl`` module, or ``None`` when *error* is not that case.

    fastapi/uvicorn import ``ssl``; when the policy blocks the ``_ssl`` DLL, the import
    fails with the signature ``DLL load failed ... _ssl`` — and the generic
    missing-deps repair loop above can never fix it, so users looped on repair
    (#63796). The desktop's embedded runtime is the usual victim; the CLI and the
    gateway run on the system Python and stay unaffected.
    """
    message = str(error)
    if "DLL load failed" not in message or "_ssl" not in message:
        return None
    return "\n".join([
        "✗ The embedded Python runtime is blocked by Windows security policy.",
        "",
        "Root cause: Python's SSL module (_ssl) could not be loaded:",
        f"  {message}",
        "",
        "This happens when Windows Smart App Control or an Application Control",
        "policy blocks the embedded Python runtime that ships with Hermes Desktop.",
        "A repair / reinstall loop cannot fix this — the packages are not missing,",
        "the runtime's DLL is being blocked.",
        "",
        "Recovery options:",
        "  1. Use a trusted system Python installation instead of the embedded",
        "     runtime (if your organization allows it).",
        "  2. Ask your IT administrator for an exemption for the Hermes Desktop",
        "     application (Smart App Control / Application Control).",
        "  3. Use the CLI or gateway instead (they use your system Python).",
        "",
        "See https://aka.ms/smartappcontrol for Windows Smart App Control details.",
    ])
