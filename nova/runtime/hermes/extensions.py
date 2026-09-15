"""The MCP and plugin inventory, read from the runtime rather than kept by hand.

Two registries the runtime already owns and already publishes:

``hermes_cli.mcp_catalog.list_catalog()`` parses the pinned manifests under
``optional-mcps/``. Presence there is the approval signal — the module says so itself:
there is no community tier and no other trust signal.

``hermes_cli.plugins_discovery.collect_directory_manifests()`` reads every plugin manifest
in full discovery order **without loading or mutating anything**, which is exactly what a
read-only control plane needs. Its sibling ``gate_manifest`` is what decides at runtime
whether a manifest actually loads, and the ``kind``/``source`` pair this module carries
through is what that decision is made of.

This module is in ``nova/runtime/hermes/`` because it imports the runtime, and that is the
only package allowed to (``tests/platform/test_boundaries.py``).
"""

from __future__ import annotations

from typing import Any


def discover() -> dict[str, Any]:
    """Everything this deployment could grant, as plain rows.

    Plain dicts rather than NOVA types so this module stays a translator. Failures are
    reported in ``detail`` rather than raised: a control plane that cannot list MCP servers
    should say so on one panel, not fail to start.
    """
    mcp, mcp_detail = _mcp_rows()
    plugins, plugin_detail = _plugin_rows()
    return {
        "mcp": mcp,
        "plugins": plugins,
        "detail": "; ".join(part for part in (mcp_detail, plugin_detail) if part),
    }


def _mcp_rows() -> tuple[list[dict], str]:
    try:
        from hermes_cli.mcp_catalog import catalog_diagnostics, list_catalog
    except Exception as exc:  # pragma: no cover — runtime not importable
        return [], f"the MCP catalogue could not be read: {exc}"

    try:
        entries = list_catalog()
    except Exception as exc:  # noqa: BLE001
        return [], f"the MCP catalogue could not be read: {exc}"

    rows = []
    for entry in entries:
        auth = getattr(entry, "auth", None)
        transport = getattr(entry, "transport", None)
        rows.append(
            {
                "id": entry.name,
                "description": getattr(entry, "description", "") or "",
                "source": getattr(entry, "source", "") or "",
                "transport": getattr(transport, "type", "") or "",
                "auth": getattr(auth, "type", "none") or "none",
                "url": getattr(transport, "url", "") or "",
                # A manifest with an install block clones and builds on the host when the
                # server is first used. An operator should see that before granting it.
                "installs": getattr(entry, "install", None) is not None,
                # The variables an operator would actually have to set. For an HTTP
                # server authenticating with a key this is NOT the manifest's ``auth.env``
                # — ``_build_server_config`` writes ``Authorization: Bearer
                # ${MCP_<NAME>_API_KEY}`` and the runtime reads that name, so a form
                # offering the manifest's name would store a value nothing looks at.
                "credentials": _credential_rows(entry),
            }
        )

    detail = ""
    try:
        # Manifests this Hermes is too old to parse. Reported rather than silently
        # dropped — "that server isn't in the list" and "your runtime is behind" look
        # identical from the Control Centre otherwise.
        stale = [name for name, kind, _ in catalog_diagnostics() if kind == "future_manifest"]
        if stale:
            detail = (
                f"{len(stale)} MCP manifest(s) are newer than this runtime and were skipped: "
                f"{', '.join(sorted(stale))}"
            )
    except Exception:  # noqa: BLE001 — diagnostics are a nicety, never a failure
        pass
    return rows, detail


def _credential_rows(entry: Any) -> list[dict]:
    """The environment variables one catalogue entry needs, as the runtime names them."""
    auth = getattr(entry, "auth", None)
    if getattr(auth, "type", "none") != "api_key":
        # OAuth stores a consent, not a variable; ``none`` needs nothing. Offering a field
        # for either would invite somebody to paste a token into a file nothing reads.
        return []
    if getattr(getattr(entry, "transport", None), "type", "") == "http":
        try:
            from hermes_cli.mcp_config import _env_key_for_server
        except Exception:  # pragma: no cover
            return []
        return [{
            "name": _env_key_for_server(entry.name),
            "prompt": f"API key for the {entry.name} MCP server",
            "required": True, "secret": True,
        }]
    return [
        {
            "name": var.name,
            "prompt": getattr(var, "prompt", "") or "",
            "required": bool(getattr(var, "required", True)),
            "secret": bool(getattr(var, "secret", True)),
        }
        for var in (getattr(auth, "env", ()) or ())
    ]


def _plugin_rows() -> tuple[list[dict], str]:
    try:
        from hermes_cli.plugins_discovery import collect_directory_manifests
        from hermes_cli.plugins_manifest import manifest_key
    except Exception as exc:  # pragma: no cover — runtime not importable
        return [], f"the plugin registry could not be read: {exc}"

    try:
        manifests = collect_directory_manifests()
    except Exception as exc:  # noqa: BLE001
        return [], f"the plugin registry could not be read: {exc}"

    rows = []
    for manifest in manifests:
        rows.append(
            {
                # The KEY, not the name. ``plugins.enabled`` is matched against both, but
                # the key is what is unique: two categories both ship a ``firecrawl``.
                "id": manifest_key(manifest),
                "name": getattr(manifest, "name", "") or "",
                "description": getattr(manifest, "description", "") or "",
                "kind": getattr(manifest, "kind", "") or "",
                "source": getattr(manifest, "source", "") or "",
                "version": str(getattr(manifest, "version", "") or ""),
                "tools": list(getattr(manifest, "provides_tools", ()) or ()),
                "credentials": [
                    # ``requires_env`` is a list of names; the manifest states nothing
                    # about whether each is a secret, so both default to the safe answer.
                    {"name": str(name), "prompt": "", "required": True, "secret": True}
                    for name in (getattr(manifest, "requires_env", ()) or ())
                ],
            }
        )
    return rows, ""


def server_config(server_id: str) -> tuple[dict[str, Any], str]:
    """The ``mcp_servers.<id>`` block for one catalogue entry, or a refusal.

    Built by the runtime's own :func:`hermes_cli.mcp_catalog._build_server_config` rather
    than assembled here. That function is what ``hermes mcp install`` writes, so a server
    NOVA compiles and a server the CLI installs produce byte-equivalent configuration —
    and an upstream change to the block format follows NOVA automatically instead of
    silently diverging.

    Returns ``({}, reason)`` when the server cannot be compiled. The one case that matters
    is an entry with an ``install:`` block: it bootstraps by cloning a repository and
    running a build on the host, which is a host operation with a real supply-chain
    surface. ``hermes mcp install <name>`` does it deliberately, once, with the operator
    present. NOVA does not do it as a side effect of saving an agent.
    """
    try:
        from hermes_cli.mcp_catalog import _build_server_config, get_entry
    except Exception as exc:  # pragma: no cover — runtime not importable
        return {}, f"the MCP catalogue could not be read: {exc}"

    try:
        entry = get_entry(server_id)
    except Exception as exc:  # noqa: BLE001
        return {}, f"the MCP catalogue could not be read: {exc}"
    if entry is None:
        return {}, f"{server_id!r} is not in this runtime's MCP catalogue"
    if getattr(entry, "install", None) is not None:
        return {}, (
            f"{server_id!r} has to be installed on the host first — it clones and builds a "
            f"repository. Run `hermes mcp install {server_id}` there, then grant it here."
        )
    try:
        config = _build_server_config(entry, None)
    except Exception as exc:  # noqa: BLE001
        return {}, f"{server_id!r} could not be compiled: {exc}"
    if not config:
        return {}, f"{server_id!r} produced no usable configuration"
    return dict(config), ""


def credential_env_for(server_id: str) -> tuple[list[dict[str, Any]], str]:
    """Which environment variables a granted MCP server needs, and their bearer key.

    An HTTP server authenticating with an API key stores ``Authorization: Bearer
    ${MCP_<NAME>_API_KEY}`` in its config block and keeps the value in the profile's
    ``.env`` — so the variable NOVA must be able to write is the one the runtime's own
    :func:`hermes_cli.mcp_config._env_key_for_server` names, not one NOVA invents.
    """
    try:
        from hermes_cli.mcp_catalog import get_entry
        from hermes_cli.mcp_config import _env_key_for_server
    except Exception as exc:  # pragma: no cover
        return [], f"the MCP catalogue could not be read: {exc}"

    entry = get_entry(server_id)
    if entry is None:
        return [], f"{server_id!r} is not in this runtime's MCP catalogue"
    return _credential_rows(entry), ""
