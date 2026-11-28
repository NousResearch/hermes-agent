"""A sign-in setting that the home's ``.env`` sets as a ``${VAR}`` template reading the old backend's
launch environment: the plan copies no key the ``.env`` sets, so after the move the gateway would resolve it
against its own service environment (a blank password: the Webapp refuses to start behind no sign-in, the
move fails its verify and rolls back). The plan refuses it up front instead, in the dry run and the apply alike.
Follow-up to #157403.
"""

from __future__ import annotations

import pytest

from hermes_cli import gateway_migrate_backends_apply as engine
from tests.hermes_cli.test_gateway_migrate_backends_apply import (
    _BACKEND_PID,
    FakeHost,
    _args,
    _config_files,
    _files,
    _nothing_changed,
    _plan,
)

_LAUNCHER_ONLY = "launcher-only-pw-3f9e"


class TemplateHost(FakeHost):
    """The old backend's unit exports SOMEVAR; the gateway's user service starts with *gateway_env*."""

    def __init__(self, home, *, gateway_env=None, **kw):
        super().__init__(home, **kw)
        self.gateway_env = dict(gateway_env or {})

    def process_env(self, pid):
        return {"PATH": "/usr/bin", "SOMEVAR": _LAUNCHER_ONLY}

    def unit_environment(self, unit, scope):
        # gateway_environment() reads the gateway's unit through this; the old unit is its launcher.
        if unit == "hermes-dashboard.service":
            return self.process_env(_BACKEND_PID), None
        return dict(self.gateway_env), None

    def gateway_environment(self):
        # An installed gateway unit, whatever this machine has (test_gateway_service_environment: not yet installed).
        return self.unit_environment("hermes-gateway.service", "user")


def _dotenv(home, text):
    (home / ".env").write_text(text, encoding="utf-8")


_TEMPLATE = "HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${SOMEVAR}\n"


def _template_blockers(out):
    return [b for b in out["plan"]["blockers"] if "is a ${...} template" in b]


def test_a_template_reading_the_launchers_environment_refuses_the_dry_run_and_the_apply(home, capsys):
    _dotenv(home, _TEMPLATE)
    host = TemplateHost(home)
    before = _files(home)

    code, out = _plan(host, capsys)
    assert code == engine.EXIT_BLOCKED and not out["steps"]
    (blocker,) = _template_blockers(out)
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in blocker and "SOMEVAR" in blocker
    assert "literal value" in blocker and "where the gateway runs" in blocker and "Nothing changed." in blocker
    _nothing_changed(home, before)

    files = _config_files(home)
    code = engine.cmd_apply(host, _args(json=True))
    applied = capsys.readouterr().out
    assert code == engine.EXIT_BLOCKED and host.calls == [] and _config_files(home) == files
    assert blocker in applied and _LAUNCHER_ONLY not in applied + blocker
    assert not (home / "backend_migration.json").exists()


def test_a_template_the_gateways_service_resolves_the_same_way_is_moved(home, capsys):
    _dotenv(home, _TEMPLATE)

    code, out = _plan(TemplateHost(home, gateway_env={"SOMEVAR": _LAUNCHER_ONLY}), capsys)

    assert code == engine.EXIT_OK and not _template_blockers(out)


@pytest.mark.parametrize("text", [
    # A template reading only an earlier literal line of the same file resolves alike everywhere.
    "PW_BASE=from-the-file\nHERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${PW_BASE}\n",
    # Control: a plain .env with the literal value, the normal single-home setup.
    "HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=plain-pw\n",
    # A template in a key the move does not carry is not the plan's business.
    "OPENAI_API_KEY=${SOMEVAR}\n",
])
def test_a_plain_env_or_a_template_that_reads_no_launch_environment_is_moved(home, capsys, text):
    _dotenv(home, text)

    code, out = _plan(TemplateHost(home), capsys)

    assert code == engine.EXIT_OK and not out["plan"]["blockers"]


def test_a_by_hand_move_refuses_the_same_template(home, capsys):
    _dotenv(home, _TEMPLATE)
    host = TemplateHost(home, supervisor="systemd", scope="system")

    code, out = _plan(host, capsys)

    assert code == engine.EXIT_BLOCKED and len(_template_blockers(out)) == 1
    assert all(entry["steps"] == [] for entry in out["plan"]["unsupported"])


class UnknownGatewayEnvHost(TemplateHost):
    """A by-hand (system unit) move where the gateway's environment cannot be read."""

    def __init__(self, home):
        super().__init__(home, supervisor="systemd", scope="system")

    def gateway_environment(self):
        return {}, "the gateway does not run as a service of this machine's service manager"


@pytest.mark.parametrize(("text", "refused"), [
    ("INFILE=fixed\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${INFILE}\n", False),  # resolved from the .env itself
    ("INFILE=fixed\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${INFILE:-default}\n", False),
    ("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${INFILE:-default}\n", True),  # INFILE may come from outside
    ("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${INFILE}\nINFILE=later\n", True),  # read before the .env sets it
    ("INFILE=${SOMEVAR}\nHERMES_DASHBOARD_BASIC_AUTH_PASSWORD=${INFILE}\n", True),  # through another line
    ("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=plain-pw\n", False),  # control: a literal
])
def test_an_unknown_environment_refuses_only_a_value_that_depends_on_one(home, capsys, text, refused):
    _dotenv(home, text)

    code, out = _plan(UnknownGatewayEnvHost(home), capsys)

    assert code == (engine.EXIT_BLOCKED if refused else engine.EXIT_UNSUPPORTED)
    assert bool(_template_blockers(out)) is refused


@pytest.mark.parametrize(("key", "refused"), [
    ("HERMES_DASHBOARD_FILES_ROOT", False),  # not read by any sign-in provider
    ("HERMES_DASHBOARD_BASIC_AUTH_SECRET", True),  # basic: the session-signing key
    ("HERMES_DASHBOARD_OIDC_CLIENT_SECRET", True),  # self-hosted OIDC
    ("HERMES_DASHBOARD_OAUTH_CLIENT_ID", True),  # Nous Portal
])
def test_only_a_template_in_a_sign_in_setting_is_refused(home, capsys, key, refused):
    _dotenv(home, f"{key}=${{SOMEVAR}}\n")

    code, out = _plan(TemplateHost(home), capsys)

    assert (code == engine.EXIT_BLOCKED) is refused and bool(_template_blockers(out)) is refused
