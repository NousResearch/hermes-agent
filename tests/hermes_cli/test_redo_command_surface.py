"""Tests for the /redo slash command surface (registry + gateway handler)."""

from __future__ import annotations

import pytest

from hermes_cli.commands import (
    COMMANDS,
    EXACT_MATCH_ONLY_COMMANDS,
    GATEWAY_KNOWN_COMMANDS,
    resolve_command,
)
from hermes_cli.commands_platforms import slack_native_slashes


class TestRegistry:
    def test_redo_is_registered(self):
        cmd = resolve_command("redo")
        assert cmd is not None
        assert cmd.category == "Session"
        assert cmd.args_hint == "[N]"

    def test_redo_is_dispatchable_by_the_gateway(self):
        assert "redo" in GATEWAY_KNOWN_COMMANDS

    def test_redo_sits_beside_undo(self):
        assert "/redo" in COMMANDS and "/undo" in COMMANDS
        assert resolve_command("redo").category == resolve_command("undo").category


class TestPrefixCollision:
    """Adding a SHORT command must not capture an existing abbreviation.

    The CLI expands an unambiguous prefix and breaks ties by "unique shortest
    match". /redo is the shortest of the /re* family, so without the
    exact-match-only guard a bare /re — previously ambiguous, and a plausible
    reach for /reset or /retry — would silently resolve to /redo and mutate
    the transcript.
    """

    def _resolve(self, typed, known):
        matches = [c for c in known if c.startswith(typed)]
        matches = [
            c for c in matches
            if c not in EXACT_MATCH_ONLY_COMMANDS or c == typed
        ]
        if len(matches) > 1:
            exact = [c for c in matches if c == typed]
            if len(exact) == 1:
                return exact[0]
            shortest_len = min(len(c) for c in matches)
            shortest = [c for c in matches if len(c) == shortest_len]
            if len(shortest) == 1:
                return shortest[0]
            return None
        return matches[0] if len(matches) == 1 else None

    def test_redo_is_exact_match_only(self):
        assert "/redo" in EXACT_MATCH_ONLY_COMMANDS

    def test_bare_re_does_not_resolve_to_redo(self):
        assert self._resolve("/re", set(COMMANDS)) is None

    def test_bare_r_does_not_resolve_to_redo(self):
        assert self._resolve("/r", set(COMMANDS)) is None

    def test_red_still_reaches_redraw(self):
        """/redo must not steal an abbreviation that already meant something."""
        if "/redraw" in COMMANDS:
            assert self._resolve("/red", set(COMMANDS)) == "/redraw"

    def test_redo_still_works_when_typed_in_full(self):
        assert self._resolve("/redo", set(COMMANDS)) == "/redo"

    def test_neighbouring_abbreviations_survive(self):
        known = set(COMMANDS)
        for typed, expected in (("/ret", "/retry"), ("/res", "/reset")):
            if expected in known:
                assert self._resolve(typed, known) == expected

    def test_adding_redo_changed_exactly_one_prefix(self):
        """Exhaustive: every prefix of every command, with and without /redo."""
        known = set(COMMANDS)
        baseline = known - {"/redo"}
        prefixes = {c[:i] for c in known for i in range(2, len(c) + 1)}
        prefixes |= {"/r", "/c", "/u", "/p"}
        changed = {
            p
            for p in prefixes
            if self._resolve(p, baseline) != self._resolve(p, known)
        }
        assert changed == {"/redo"}


class TestSlackCap:
    """Slack caps apps at 50 slash commands and the registry is at that
    ceiling, so a new command silently evicts the tail unless it is routed
    through /hermes instead."""

    def test_native_slash_count_stays_within_the_cap(self):
        assert len(slack_native_slashes()) <= 50

    def test_redo_is_routed_via_hermes_not_natively(self):
        names = {
            (c.get("command") if isinstance(c, dict) else c[0])
            for c in slack_native_slashes()
        }
        assert not any("redo" == str(n).strip("/") for n in names)

    def test_usage_survived(self):
        """/usage is the command /redo would otherwise have displaced."""
        names = {
            str(c.get("command") if isinstance(c, dict) else c[0]).strip("/")
            for c in slack_native_slashes()
        }
        assert "usage" in names


class TestRelayManifest:
    def test_redo_is_advertised(self):
        from gateway.relay.command_manifest import build_relay_command_manifest

        names = {c["name"] for c in build_relay_command_manifest()}
        assert "redo" in names
        assert "undo" in names


def _locale_dir():
    """Repository ``locales/`` directory, resolved from this test file."""
    import os

    here = os.path.dirname(os.path.abspath(__file__))       # tests/hermes_cli
    repo_root = os.path.dirname(os.path.dirname(here))      # repo root
    return os.path.join(repo_root, "locales")


class TestLocaleKeys:
    REQUIRED = (
        "nothing",
        "restart_lost",
        "restored",
        "invalid_count",
        "busy",
        "error",
    )

    def test_every_locale_defines_the_redo_keys(self):
        import glob
        import os

        import yaml

        files = sorted([p for p in glob.glob(os.path.join(_locale_dir(), "*.yaml")) if not p.endswith(".tui.yaml")])
        assert files, "no locale files found"
        for path in files:
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            redo = (data.get("gateway") or {}).get("redo")
            assert isinstance(redo, dict), f"{path}: missing gateway.redo"
            missing = set(self.REQUIRED) - set(redo)
            assert not missing, f"{path}: missing {sorted(missing)}"

    def test_placeholders_survive_translation(self):
        import glob
        import os

        import yaml

        for path in sorted([p for p in glob.glob(os.path.join(_locale_dir(), "*.yaml")) if not p.endswith(".tui.yaml")]):
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            redo = (data.get("gateway") or {}).get("redo") or {}
            assert "{ops}" in redo["restored"], path
            assert "{count}" in redo["restored"], path
            assert "{arg}" in redo["invalid_count"], path
