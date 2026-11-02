"""Self-aware gateway lifecycle guard: siblings allowed, self still blocked.

The launchd/systemd branches of ``cron.lifecycle_guard`` were LABEL-BLIND:
``_HERMES_GATEWAY_LABEL_RE`` matched *any* hermes gateway label, so a
supervised gateway running as ``ai.hermes.gateway-rescue`` could not run
``launchctl bootout gui/501/ai.hermes.gateway`` to recover a wedged sibling
profile's gateway — which is the entire purpose of a dedicated break-glass
profile. Operators had to launder such recoveries through ``write_file`` plus
``ssh localhost 'nohup bash script &'`` instead.

The #30719 respawn loop these branches exist to prevent requires the command
to kill THIS process, so a lifecycle command whose only explicit targets are
SIBLING services is safe. Self-targeting, mixed self+sibling, variable-built
labels, and undeterminable identity all stay blocked (fail closed).
"""

from __future__ import annotations

import pytest

from cron import lifecycle_guard
from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command,
    contains_gateway_lifecycle_command_or_referenced_script,
    contains_launchctl_submit_command,
    describe_self_gateway_identity,
)


SELF_LAUNCHD = "ai.hermes.gateway-rescue"
SIBLING_LAUNCHD = "ai.hermes.gateway"
SELF_SYSTEMD = "hermes-gateway-rescue"
SIBLING_SYSTEMD = "hermes-gateway"


@pytest.fixture
def launchd_identity(monkeypatch):
    """Pin self-identity to the rescue launchd job, as launchd would."""
    monkeypatch.setenv("XPC_SERVICE_NAME", SELF_LAUNCHD)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(
        lifecycle_guard, "_profile_derived_self_names", lambda: set()
    )
    return SELF_LAUNCHD


@pytest.fixture
def systemd_identity(monkeypatch):
    """Pin self-identity to the rescue systemd unit, as systemd would."""
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.setenv("INVOCATION_ID", "0123456789abcdef")
    monkeypatch.setattr(
        lifecycle_guard, "_systemd_self_unit", lambda: SELF_SYSTEMD
    )
    monkeypatch.setattr(
        lifecycle_guard, "_profile_derived_self_names", lambda: set()
    )
    return SELF_SYSTEMD


@pytest.fixture
def no_identity(monkeypatch):
    """No determinable identity at all — the guard must fail closed."""
    monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(
        lifecycle_guard, "_profile_derived_self_names", lambda: set()
    )


# ---------------------------------------------------------------------------
# launchd — sibling targets allowed
# ---------------------------------------------------------------------------


class TestLaunchdSiblingAllowed:
    @pytest.mark.parametrize(
        "text",
        [
            # The exact incident command.
            "launchctl bootout gui/501/ai.hermes.gateway",
            "launchctl kickstart -k gui/501/ai.hermes.gateway",
            "launchctl stop ai.hermes.gateway",
            "launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway.plist",
            "launchctl disable gui/501/ai.hermes.gateway-worker",
            # Another profile's gateway, distinct suffix.
            "launchctl bootout gui/501/ai.hermes.gateway-cronus",
            # Our own WATCHDOG is sibling-safe: stopping it does not kill us.
            "launchctl bootout gui/501/ai.hermes.gateway-watchdog",
        ],
    )
    def test_sibling_launchd_lifecycle_allowed(self, text, launchd_identity):
        assert not contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"

    def test_multiple_siblings_allowed(self, launchd_identity):
        text = (
            "launchctl bootout gui/501/ai.hermes.gateway; "
            "launchctl bootout gui/501/ai.hermes.gateway-cronus"
        )
        assert not contains_gateway_lifecycle_command(text)


class TestLaunchdSelfBlocked:
    @pytest.mark.parametrize(
        "text",
        [
            "launchctl bootout gui/501/ai.hermes.gateway-rescue",
            "launchctl kickstart -k gui/501/ai.hermes.gateway-rescue",
            "launchctl stop ai.hermes.gateway-rescue",
            "launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway-rescue.plist",
            "launchctl disable gui/501/ai.hermes.gateway-rescue",
            # Case-insensitive: launchd labels compare case-insensitively.
            "launchctl bootout gui/501/AI.HERMES.GATEWAY-RESCUE",
        ],
    )
    def test_self_launchd_lifecycle_blocked(self, text, launchd_identity):
        assert contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_mixed_self_and_sibling_blocked(self, launchd_identity):
        text = (
            "launchctl bootout gui/501/ai.hermes.gateway; "
            "launchctl bootout gui/501/ai.hermes.gateway-rescue"
        )
        assert contains_gateway_lifecycle_command(text)

    def test_mixed_in_one_segment_blocked(self, launchd_identity):
        text = "launchctl bootout gui/501/ai.hermes.gateway-rescue ai.hermes.gateway"
        assert contains_gateway_lifecycle_command(text)


class TestVariableLabelBlocked:
    """An unexpanded shell value could expand to our own label at runtime.

    Scope note: a command carrying NO literal gateway token at all (e.g.
    ``launchctl bootout gui/501/$LABEL``) was never blocked by this guard —
    every launchd branch is anchored on a literal ``hermes[.-]?gateway``
    token. That pre-existing gap is out of scope here and is asserted
    unchanged in :class:`TestVariableOnlyLabelsAreAPreExistingGap`. What this
    change must not do is let a variable *launder* a sibling-looking command
    into an exemption, which is what these cases cover.
    """

    @pytest.mark.parametrize(
        "text",
        [
            # The 2026-08-02 incident shape: literal labels in an earlier
            # segment, the verb takes a variable. Blocked before this change
            # and must stay blocked.
            (
                "uid=$(id -u); for item in 'ai.hermes.gateway-worker:/a.plist' "
                "'ai.hermes.gateway:/p.plist'; do label=${item%%:*}; "
                'launchctl bootout "gui/$uid/$label"; done'
            ),
            # Sibling label present, but a variable rides along in the same
            # segment — still blocked, it may expand to us.
            "launchctl bootout gui/501/ai.hermes.gateway $EXTRA",
            "launchctl bootout gui/501/ai.hermes.gateway-cronus ${OTHER}",
            "launchctl bootout gui/501/ai.hermes.gateway $(cat /tmp/label)",
            "launchctl bootout gui/501/ai.hermes.gateway `cat /tmp/label`",
            "systemctl --user restart hermes-gateway $UNIT",
        ],
    )
    def test_variable_alongside_sibling_label_blocked(self, text, launchd_identity):
        assert contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_sibling_predicate_rejects_variable_segment(self, launchd_identity):
        assert not lifecycle_guard._lifecycle_targets_only_sibling_gateways(
            "launchctl bootout gui/501/ai.hermes.gateway $EXTRA"
        )


class TestVariableOnlyLabelsAreAPreExistingGap:
    """Parity guard: commands with no literal gateway token are unchanged.

    These were already allowed before the self-awareness change (the guard's
    launchd branches all require a literal ``hermes[.-]?gateway`` token). The
    assertions exist so a future widening of the label patterns has to update
    this file deliberately rather than silently changing the contract.
    """

    @pytest.mark.parametrize(
        "text",
        [
            'launchctl bootout "gui/$uid/$label"',
            "launchctl bootout gui/501/$LABEL",
            "launchctl bootout gui/501/${label}",
            "launchctl bootout gui/501/$(cat /tmp/label)",
        ],
    )
    def test_variable_only_label_unchanged(self, text, launchd_identity):
        assert not contains_gateway_lifecycle_command(text)

    def test_variable_only_label_unchanged_without_identity(self, no_identity):
        """Same verdict with no identity — the exemption is not what allows it."""
        assert not contains_gateway_lifecycle_command(
            "launchctl bootout gui/501/$LABEL"
        )


class TestNoIdentityFailsClosed:
    @pytest.mark.parametrize(
        "text",
        [
            "launchctl bootout gui/501/ai.hermes.gateway",
            "launchctl kickstart -k gui/501/ai.hermes.gateway-cronus",
            "systemctl --user restart hermes-gateway",
        ],
    )
    def test_undeterminable_identity_keeps_blocking(self, text, no_identity):
        assert contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_self_names_empty_when_undeterminable(self, no_identity):
        assert lifecycle_guard._self_gateway_service_names() == set()

    def test_sibling_predicate_false_without_identity(self, no_identity):
        assert not lifecycle_guard._lifecycle_targets_only_sibling_gateways(
            "launchctl bootout gui/501/ai.hermes.gateway"
        )


# ---------------------------------------------------------------------------
# systemd
# ---------------------------------------------------------------------------


class TestSystemdSiblingAllowed:
    @pytest.mark.parametrize(
        "text",
        [
            "systemctl --user restart hermes-gateway",
            "systemctl --user stop hermes-gateway.service",
            "sudo systemctl restart hermes-gateway-worker",
            "systemctl --user stop hermes-gateway-watchdog",
        ],
    )
    def test_sibling_systemd_lifecycle_allowed(self, text, systemd_identity):
        assert not contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"


class TestSystemdSelfBlocked:
    @pytest.mark.parametrize(
        "text",
        [
            "systemctl --user restart hermes-gateway-rescue",
            "systemctl --user stop hermes-gateway-rescue.service",
            "sudo systemctl restart hermes-gateway-rescue",
        ],
    )
    def test_self_systemd_lifecycle_blocked(self, text, systemd_identity):
        assert contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    def test_mixed_self_and_sibling_systemd_blocked(self, systemd_identity):
        text = (
            "systemctl --user restart hermes-gateway && "
            "systemctl --user restart hermes-gateway-rescue"
        )
        assert contains_gateway_lifecycle_command(text)

    def test_launchd_self_does_not_exempt_systemd_sibling_name(
        self, launchd_identity
    ):
        """Under a launchd identity, the default systemd unit is a sibling."""
        assert not contains_gateway_lifecycle_command(
            "systemctl --user restart hermes-gateway"
        )


# ---------------------------------------------------------------------------
# launchctl submit / bootstrap
# ---------------------------------------------------------------------------


class TestSubmitStillBlocked:
    @pytest.mark.parametrize(
        "text",
        [
            "launchctl submit -l ai.hermes.gateway -- /bin/sh helper.sh",
            "launchctl submit -l ai.hermes.gateway-cronus -- /bin/sh helper.sh",
            "launchctl submit -l neutral-name -- /bin/sh helper.sh",
        ],
    )
    def test_submit_blocked_regardless_of_label(self, text, launchd_identity):
        assert contains_launchctl_submit_command(text), f"Should match: {text!r}"
        assert contains_gateway_lifecycle_command_or_referenced_script(text)


class TestBootstrapSiblingPlist:
    def test_sibling_plist_bootstrap_allowed(self, launchd_identity):
        text = (
            "launchctl bootstrap gui/501 "
            "/Users/ace/Library/LaunchAgents/ai.hermes.gateway.plist"
        )
        assert not contains_launchctl_submit_command(text)
        assert not contains_gateway_lifecycle_command_or_referenced_script(text)

    def test_sibling_suffixed_plist_bootstrap_allowed(self, launchd_identity):
        text = (
            "launchctl bootstrap gui/501 "
            "/Users/ace/Library/LaunchAgents/ai.hermes.gateway-cronus.plist"
        )
        assert not contains_launchctl_submit_command(text)

    def test_self_plist_bootstrap_blocked(self, launchd_identity):
        text = (
            "launchctl bootstrap gui/501 "
            "/Users/ace/Library/LaunchAgents/ai.hermes.gateway-rescue.plist"
        )
        assert contains_launchctl_submit_command(text)
        assert contains_gateway_lifecycle_command_or_referenced_script(text)

    def test_unrelated_plist_bootstrap_still_blocked(self, launchd_identity):
        """Only full gateway labels qualify; anything else keeps the block."""
        text = "launchctl bootstrap gui/501 /tmp/ai.hermes.restart-once.plist"
        assert contains_launchctl_submit_command(text)

    def test_variable_plist_bootstrap_blocked(self, launchd_identity):
        text = "launchctl bootstrap gui/501 $PLIST"
        assert contains_launchctl_submit_command(text)

    def test_bootstrap_without_identity_blocked(self, no_identity):
        text = (
            "launchctl bootstrap gui/501 "
            "/Users/ace/Library/LaunchAgents/ai.hermes.gateway.plist"
        )
        assert contains_launchctl_submit_command(text)


# ---------------------------------------------------------------------------
# Referenced-script scanning inherits the same self/sibling logic
# ---------------------------------------------------------------------------


class TestReferencedScriptInheritsSelfAwareness:
    def test_sibling_bootout_in_referenced_script_allowed(
        self, tmp_path, launchd_identity
    ):
        script = tmp_path / "recover_sibling.sh"
        script.write_text(
            "#!/bin/bash\nlaunchctl bootout gui/501/ai.hermes.gateway\n"
        )
        assert not contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {script}", cwd=str(tmp_path)
        )

    def test_self_bootout_in_referenced_script_blocked(
        self, tmp_path, launchd_identity
    ):
        script = tmp_path / "kill_self.sh"
        script.write_text(
            "#!/bin/bash\nlaunchctl bootout gui/501/ai.hermes.gateway-rescue\n"
        )
        assert contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {script}", cwd=str(tmp_path)
        )

    def test_mixed_in_referenced_script_blocked(self, tmp_path, launchd_identity):
        script = tmp_path / "mixed.sh"
        script.write_text(
            "#!/bin/bash\n"
            "launchctl bootout gui/501/ai.hermes.gateway\n"
            "launchctl bootout gui/501/ai.hermes.gateway-rescue\n"
        )
        assert contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {script}", cwd=str(tmp_path)
        )

    def test_variable_label_in_referenced_script_blocked(
        self, tmp_path, launchd_identity
    ):
        """A script whose loop list carries LITERAL labels stays blocked.

        Written as ONE logical line: the order-independent launchctl pass is
        per-line by construction, so a list on line 3 and the verb on line 5
        is a separate pre-existing gap, not something this change touches.
        """
        script = tmp_path / "loop.sh"
        script.write_text(
            "#!/bin/bash\n"
            "uid=$(id -u); for item in 'ai.hermes.gateway-worker:/a.plist' "
            "'ai.hermes.gateway:/p.plist'; do label=${item%%:*}; "
            'launchctl bootout "gui/$uid/$label"; done\n'
        )
        assert contains_gateway_lifecycle_command_or_referenced_script(
            f"bash {script}", cwd=str(tmp_path)
        )

    def test_sh_c_payload_sibling_allowed(self, launchd_identity):
        assert not contains_gateway_lifecycle_command_or_referenced_script(
            "sh -c 'launchctl bootout gui/501/ai.hermes.gateway'"
        )

    def test_sh_c_payload_self_blocked(self, launchd_identity):
        assert contains_gateway_lifecycle_command_or_referenced_script(
            "sh -c 'launchctl bootout gui/501/ai.hermes.gateway-rescue'"
        )


# ---------------------------------------------------------------------------
# Identity resolution + block-message naming
# ---------------------------------------------------------------------------


class TestSelfIdentityResolution:
    def test_xpc_service_name_is_authoritative(self, launchd_identity):
        assert SELF_LAUNCHD in lifecycle_guard._self_gateway_service_names()

    def test_non_gateway_xpc_name_ignored(self, monkeypatch):
        monkeypatch.setenv("XPC_SERVICE_NAME", "com.apple.xpc.launchd.oneshot")
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.setattr(
            lifecycle_guard, "_profile_derived_self_names", lambda: set()
        )
        assert lifecycle_guard._self_gateway_service_names() == set()

    def test_profile_derived_names_used_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("XPC_SERVICE_NAME", raising=False)
        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.setattr(
            lifecycle_guard,
            "_profile_derived_self_names",
            lambda: {SELF_LAUNCHD, SELF_SYSTEMD},
        )
        names = lifecycle_guard._self_gateway_service_names()
        assert names == {SELF_LAUNCHD, SELF_SYSTEMD}

    def test_watchdog_label_is_not_self(self, launchd_identity):
        assert (
            "ai.hermes.gateway-watchdog"
            not in lifecycle_guard._self_gateway_service_names()
        )

    def test_describe_identity_names_the_label(self, launchd_identity):
        described = describe_self_gateway_identity()
        assert SELF_LAUNCHD in described
        assert described.startswith("this gateway runs as")

    def test_describe_identity_names_the_systemd_unit(self, systemd_identity):
        described = describe_self_gateway_identity()
        assert f"{SELF_SYSTEMD}.service" in described

    def test_describe_identity_empty_without_identity(self, no_identity):
        assert describe_self_gateway_identity() == ""


class TestNoRegressionOnNonLabelBranches:
    """Branch A/D keep their existing behaviour — untouched by this change."""

    @pytest.mark.parametrize(
        "text",
        [
            # Branch A: bare `hermes gateway restart` is self-targeting.
            "hermes gateway restart",
            "hermes gateway stop",
            # Branch D: pkill targets by process pattern, not service label.
            "kill $(pgrep -f hermes-gateway)",
            "pkill -f hermes.*gateway",
        ],
    )
    def test_non_label_branches_still_blocked(self, text, launchd_identity):
        assert contains_gateway_lifecycle_command(text), f"Should match: {text!r}"

    @pytest.mark.parametrize(
        "text",
        [
            # Unrelated launchd services stay unblocked, as before.
            "launchctl unload ai.hermes.update-checker.plist",
            "launchctl list",
            # Prose / paths must not match either way.
            "echo after the hermes gateway restarted cleanly",
            "git log -3 -- skills/hermes-harness/safe-gateway-restart/SKILL.md",
        ],
    )
    def test_previously_safe_commands_stay_safe(self, text, launchd_identity):
        assert not contains_gateway_lifecycle_command(text), f"Should NOT match: {text!r}"

    def test_loopback_ssh_self_still_blocked(self, launchd_identity):
        assert contains_gateway_lifecycle_command(
            "ssh localhost 'launchctl bootout gui/501/ai.hermes.gateway-rescue'"
        )
