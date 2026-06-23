"""Regression test for the systemd unit-glob used by ``hermes update``.

The post-update auto-restart phase discovers all running Hermes gateway
systemd units via ``systemctl list-units <glob>``.  The glob must match
both the default service (``hermes-gateway.service``) and profile-named
services (``hermes-<profile>-gateway.service``, e.g.
``hermes-secondbrain-gateway.service``).

A previous version used ``hermes-gateway*``, which silently missed
profile-named gateway services — leaving them running stale code after
an update.  These tests lock the glob down so that regression cannot
recur without failing loudly.

Because the discovery logic is inline inside ``_cmd_update_impl`` and
not factored into a helper, we exercise the *contract*: the glob string
the source passes to ``systemctl list-units`` must match both unit-name
shapes.  We extract the glob from the source to avoid hard-coding drift,
and we also assert against a realistic ``systemctl list-units`` stdout
fixture (the same format the production code parses line-by-line).
"""

from __future__ import annotations

import re
import subprocess

import pytest

from hermes_cli import main as cli_main


# ─── Source extraction ────────────────────────────────────────────────────────


def _extract_gateway_glob_from_source() -> str:
    """Pull the literal glob string passed to ``list-units`` in source.

    The discovery call looks like::

        subprocess.run(scope_cmd + ["list-units", "<glob>", "--plain", ...])

    We regex the source so the test tracks the production value without
    hard-coding it.  If the call shape changes materially, this helper
    raises — better a loud test failure than a silent glob regression.
    """
    import inspect

    src = inspect.getsource(cli_main._cmd_update_impl)
    # Find the list-units call and capture the glob argument that follows.
    m = re.search(r'["\']list-units["\']\s*,\s*\n\s*["\'](hermes[^"\']*gateway[^"\']*)["\']', src)
    if not m:
        pytest.fail(
            "Could not locate the `list-units <glob>` call inside "
            "_cmd_update_impl — has the discovery code been restructured?"
        )
    return m.group(1)


# ─── Realistic systemctl list-units stdout fixture ────────────────────────────


# Format matches `systemctl list-units --plain --no-legend --no-pager`:
#   UNIT                               LOAD   ACTIVE SUB     DESCRIPTION
# Columns are whitespace-separated; production code does line.split()[0].
LIST_UNITS_STDOUT = (
    "hermes-gateway.service             loaded active running Hermes Agent Gateway\n"
    "hermes-secondbrain-gateway.service loaded active running Hermes Second Brain Gateway\n"
)


def _parse_units(stdout: str) -> list[str]:
    """Mirror the production parse: first whitespace token per line."""
    units = []
    for line in stdout.strip().splitlines():
        parts = line.split()
        if parts and parts[0].endswith(".service"):
            units.append(parts[0])
    return units


# ─── Tests ────────────────────────────────────────────────────────────────────


class TestUpdateGatewayServiceGlob:
    """The glob must catch default + profile gateway services alike."""

    def test_glob_matches_default_service(self):
        glob = _extract_gateway_glob_from_source()
        # fnmatch is what systemd uses for unit-name globs.
        import fnmatch

        assert fnmatch.fnmatch("hermes-gateway.service", glob), (
            f"Glob {glob!r} must match the default 'hermes-gateway.service'"
        )

    def test_glob_matches_profile_gateway_service(self):
        glob = _extract_gateway_glob_from_source()
        import fnmatch

        assert fnmatch.fnmatch("hermes-secondbrain-gateway.service", glob), (
            f"Glob {glob!r} must match profile-named "
            "'hermes-secondbrain-gateway.service' — otherwise profile "
            "gateways are left running stale code after an update"
        )

    def test_glob_matches_arbitrary_profile_name(self):
        """A profile with hyphens or numbers in its name must also match."""
        glob = _extract_gateway_glob_from_source()
        import fnmatch

        for name in ("hermes-work-gateway.service", "hermes-coder-2-gateway.service"):
            assert fnmatch.fnmatch(name, glob), (
                f"Glob {glob!r} must match {name!r}"
            )

    def test_full_list_units_fixture_both_discovered(self):
        """End-to-end: parsing realistic systemctl output yields both units."""
        glob = _extract_gateway_glob_from_source()
        import fnmatch

        units = _parse_units(LIST_UNITS_STDOUT)
        matched = [u for u in units if fnmatch.fnmatch(u, glob)]
        assert matched == [
            "hermes-gateway.service",
            "hermes-secondbrain-gateway.service",
        ], (
            "Glob must discover both default and profile gateway services; "
            f"got {matched!r}"
        )

    def test_glob_is_not_the_old_buggy_value(self):
        """The old glob ``hermes-gateway*`` missed profile services — guard it."""
        glob = _extract_gateway_glob_from_source()
        assert glob != "hermes-gateway*", (
            "Glob regressed to the old 'hermes-gateway*' value which misses "
            "profile-named gateway services (e.g. hermes-secondbrain-gateway)"
        )
