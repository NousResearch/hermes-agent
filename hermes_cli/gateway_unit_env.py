"""Gateway systemd unit ``Environment=`` directives: escaping, LD_LIBRARY_PATH, sealed-install roots.

Extracted from ``hermes_cli/gateway.py``. Bodies read facade helpers through ``_gw()`` (late
binding on ``hermes_cli.gateway``) so the seams tests and callers patch on the facade keep
intercepting the moved code.
"""
from __future__ import annotations

import os


def _gw():
    from hermes_cli import gateway  # late: the facade imports this module
    return gateway


def _systemd_env_line(name: str, value: str) -> str:
    """One ``Environment="NAME=value"`` line: ``\\`` and ``"`` escaped for systemd's quoting, ``%``
    doubled so specifier expansion leaves the value alone."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    return f'Environment="{name}={escaped}"\n'


def _installed_unit_value(name: str, system: bool) -> str:
    """One ``Environment=`` value baked into the installed unit; ``""`` when absent."""
    gw = _gw()
    return gw._unit_environment_value(gw.get_systemd_unit_path(system=system), name) or ""


def _ld_library_path_line(system: bool, target_home_dir: str | None = None) -> str:
    """Carry the installer's LD_LIBRARY_PATH into the unit (glibc reads it only at process start, so
    ~/.hermes/.env is too late for CUDA libs — #14613); system units remap caller-home components.

    The installed unit is the fallback source: the unit is regenerated and compared on every
    start/restart/status, and a shell without the export (ssh, cron, ``sudo`` strips ``LD_*``)
    must not be able to "repair" the line away."""
    raw = os.environ.get("LD_LIBRARY_PATH", "") or _installed_unit_value("LD_LIBRARY_PATH", system)
    components = [p for p in raw.split(":") if p]
    if target_home_dir is not None:
        components = [_gw()._remap_path_for_user(p, target_home_dir) for p in components]
    return _systemd_env_line("LD_LIBRARY_PATH", ":".join(components)) if components else ""


_INSTALL_RESOURCE_ENV = (
    "HERMES_INSTALL_ROOT",
    "HERMES_BUNDLED_SKILLS",
    "HERMES_OPTIONAL_SKILLS",
    "HERMES_BUNDLED_PLUGINS",
    "HERMES_BUNDLED_LOCALES",
    "HERMES_OPTIONAL_MCPS",
)


def _install_env_lines(system: bool, target_home_dir: str | None = None) -> str:
    """Carry a sealed install's resource roots into the unit; system units remap caller-home paths.

    ``_INSTALL_RESOURCE_ENV`` is the set a sealed install (Nix, Homebrew, desktop bundle) exports from
    its launcher wrapper — ``scripts/build/inputs.py``'s ``RESOURCE_ENV`` plus the install root; keep
    the two in lockstep.

    ``ExecStart`` is the store interpreter, not the packaging wrapper that exports these, and systemd
    starts a unit from a clean environment — so without these lines the gateway resolves bundled data
    against a source checkout a sealed install does not have. ``agent.i18n._locales_dir()`` then finds
    no catalog and every string renders as its raw key (``gateway.reset.header_default``), and
    skills/plugins/MCPs silently fall back under ``HERMES_HOME``.

    The installed unit is the fallback source for the same reason ``LD_LIBRARY_PATH`` uses it: the unit
    is regenerated and compared on every start/restart/status, so a later shell that never sourced the
    wrapper must not be able to "repair" the lines away."""
    lines = ""
    for name in _INSTALL_RESOURCE_ENV:
        value = os.environ.get(name, "").strip() or _installed_unit_value(name, system)
        if not value:
            continue
        if target_home_dir is not None:
            value = _gw()._remap_path_for_user(value, target_home_dir)
        lines += _systemd_env_line(name, value)
    return lines
