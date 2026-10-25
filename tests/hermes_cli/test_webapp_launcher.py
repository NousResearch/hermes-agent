"""``hermes webapp`` opens the browser-hosted Desktop that the gateway serves.

It builds the Webapp bundle when stale, starts the gateway that serves the current profile (the default
gateway for a profile it serves, else the profile's own), asks that gateway for a one-use launch link
over the control socket and opens it. It never serves anything itself. ``--sign-out`` reports what
the gateway did, waits for a gateway that is still starting, and replaces the stored credential
itself only when no gateway runs.
"""
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import webapp_launcher as launcher

LINK = "http://127.0.0.1:9120/webapp/launch#ticket=" + "t" * 43


def _home(path: Path, config: str = "") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(config, encoding="utf-8")
    return path


def _args(**overrides) -> Namespace:
    values = {"no_open": False, "status": False, "sign_out": False, "build_only": False,
              "force_build": False, "skip_build": False}
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def world(monkeypatch, tmp_path):
    """The launcher's side effects, recorded; the control socket answers from ``answers``."""
    calls = {"build": 0, "ensure": [], "control": [], "opened": []}
    answers = {"webapp-launch": {"ok": True, "result": {"url": LINK, "private": False, "expires_in_seconds": 60}},
               "webapp-status": {"ok": True, "result": {"state": "serving", "url": LINK.split("/webapp")[0],
                                                         "reason": None}},
               "webapp-sign-out": {"ok": True, "result": {"signed_out": True}}}

    def build(args):
        calls["build"] += 1
        return tmp_path / "dist-webapp"

    def control(home, verb, params=None):
        calls["control"].append((Path(home), verb))
        calls.setdefault("params", []).append(params or {})
        answer = answers[verb]
        if isinstance(answer, list):  # one answer per call, in order
            answer = answer.pop(0) if len(answer) > 1 else answer[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(launcher, "_build", build)
    monkeypatch.setattr(launcher, "_ensure", lambda home: calls["ensure"].append(Path(home)))
    monkeypatch.setattr(launcher, "_control", control)
    monkeypatch.setattr(launcher, "_open", lambda url: calls["opened"].append(url) or True)
    states = ["absent"]  # what the gateway's lock and control socket say when it does not answer
    monkeypatch.setattr(launcher, "_gateway_state", lambda home: states.pop(0) if len(states) > 1 else states[0],
                        raising=False)
    monkeypatch.setattr("time.sleep", lambda seconds: calls.setdefault("slept", []).append(seconds))
    root = _home(tmp_path / "root", "webapp:\n  enabled: true\n")
    monkeypatch.setenv("HERMES_HOME", str(root))
    return {"calls": calls, "answers": answers, "root": root, "tmp": tmp_path, "monkeypatch": monkeypatch,
            "states": states}


def _use_home(world, home):
    world["monkeypatch"].setenv("HERMES_HOME", str(home))


def test_the_webapp_opens_from_a_one_use_link_its_gateway_mints(world, capsys):
    assert launcher.cmd_webapp(_args()) == 0
    calls = world["calls"]
    assert calls["build"] == 1
    assert calls["ensure"] == [world["root"]]
    assert calls["control"] == [(world["root"], "webapp-launch")]
    assert calls["opened"] == [LINK]
    out = capsys.readouterr().out
    assert "Opened the Webapp in your browser" in out
    assert "ticket=" not in out


def test_a_named_profile_opens_the_webapp_of_the_default_gateway_on_that_profile(world):
    profile = _home(world["root"] / "profiles" / "work")
    _use_home(world, profile)
    assert launcher.cmd_webapp(_args()) == 0
    assert world["calls"]["ensure"] == [world["root"]]
    assert world["calls"]["control"] == [(world["root"], "webapp-launch")]
    assert world["calls"]["params"] == [{"profile": "work"}]


def test_the_default_profile_opens_without_a_profile_choice(world):
    assert launcher.cmd_webapp(_args()) == 0
    assert world["calls"]["params"] == [{}]


def test_a_standalone_profile_opens_the_webapp_of_its_own_gateway(world):
    profile = _home(world["root"] / "profiles" / "solo",
                    "gateway:\n  standalone: true\nwebapp:\n  enabled: true\n")
    _use_home(world, profile)
    assert launcher.cmd_webapp(_args()) == 0
    assert world["calls"]["ensure"] == [profile]


def test_the_launcher_says_how_to_turn_the_webapp_on(world, capsys):
    (world["root"] / "config.yaml").write_text("", encoding="utf-8")
    world["answers"]["webapp-status"] = {"ok": True, "result": {"state": "off", "url": None, "reason": None}}
    assert launcher.cmd_webapp(_args()) == 1
    out = capsys.readouterr().out
    assert "webapp.enabled: true" in out and str(world["root"] / "config.yaml") in out
    assert (world["calls"]["build"], world["calls"]["ensure"]) == (0, [])


def test_a_gateway_started_before_the_webapp_was_turned_on_asks_for_a_restart(world, capsys):
    world["answers"]["webapp-launch"] = {"ok": False, "error": "RuntimeError: The Webapp is off. Set ..."}
    world["answers"]["webapp-status"] = {"ok": True, "result": {"state": "off", "url": None, "reason": None}}
    assert launcher.cmd_webapp(_args()) == 1
    out = capsys.readouterr().out
    assert "hermes gateway restart" in out
    assert world["calls"]["opened"] == []


def test_a_webapp_the_gateway_could_not_serve_reports_why_and_how_to_retry(world, capsys):
    # The gateway reads the Webapp's settings and sign-in providers when it starts, so a fix only
    # takes effect after a restart.
    reason = "could not listen on 127.0.0.1:9120 (Address already in use)"
    world["answers"]["webapp-launch"] = {"ok": False, "error": f"RuntimeError: The Webapp is unavailable: {reason}"}
    world["answers"]["webapp-status"] = {"ok": True, "result": {"state": "unavailable", "url": None,
                                                               "reason": reason}}
    assert launcher.cmd_webapp(_args()) == 1
    out = capsys.readouterr().out
    assert reason in out and "hermes gateway restart" in out
    assert launcher.cmd_webapp(_args(status=True)) == 0
    out = capsys.readouterr().out
    assert reason in out and "hermes gateway restart" in out


def test_a_launch_refused_for_a_stale_bundle_asks_for_a_build_not_a_restart(world, capsys):
    """A bundle that is missing or that hermes update could not rebuild is served as soon as it is
    built, and a restart keeps the stale mark: the refusal says to build it and never sends the user
    to restart the gateway, as --status already does."""
    from gateway.webapp_listener import NOT_REBUILT
    world["answers"]["webapp-launch"] = {"ok": False, "error": f"RuntimeError: The Webapp is unavailable: {NOT_REBUILT}"}
    world["answers"]["webapp-status"] = {"ok": True, "result": {"state": "unavailable", "url": None,
                                                               "reason": NOT_REBUILT}}
    assert launcher.cmd_webapp(_args(skip_build=True)) == 1
    out = capsys.readouterr().out
    assert "Run hermes webapp to rebuild it." in out and launcher.RESTART_COMMAND not in out


SERVING = {"ok": True, "result": {"state": "serving", "url": "http://127.0.0.1:9120", "reason": None}}


def test_a_webapp_turned_off_in_config_but_still_served_says_a_restart_stops_it(world, capsys):
    """The gateway reads webapp: only when it starts. After the config turns the Webapp off, the running
    gateway still serves it (on the LAN for a wildcard host) until it restarts: hermes webapp says so,
    instead of "The Webapp is off", and opens nothing."""
    (world["root"] / "config.yaml").write_text("webapp:\n  enabled: false\n  host: 0.0.0.0\n", encoding="utf-8")
    world["answers"]["webapp-status"] = SERVING
    assert launcher.cmd_webapp(_args()) == 1
    out = capsys.readouterr().out
    assert "The Webapp is off" not in out
    assert "http://127.0.0.1:9120" in out and launcher.RESTART_COMMAND in out
    assert (world["calls"]["build"], world["calls"]["ensure"], world["calls"]["opened"]) == (0, [], [])


def test_status_of_a_webapp_served_after_the_config_turned_it_off_says_a_restart_stops_it(world, capsys):
    (world["root"] / "config.yaml").write_text("webapp:\n  enabled: false\n", encoding="utf-8")
    world["answers"]["webapp-status"] = SERVING
    assert launcher.cmd_webapp(_args(status=True)) == 0
    out = capsys.readouterr().out
    assert "Serving at http://127.0.0.1:9120" in out and launcher.RESTART_COMMAND in out


def test_behind_the_dashboard_sign_in_the_launcher_opens_the_webapp_to_sign_in(world, capsys):
    url = "http://127.0.0.1:9120/"
    world["answers"]["webapp-launch"] = {"ok": True, "result": {
        "url": url, "private": False, "expires_in_seconds": None, "sign_in": True}}
    assert launcher.cmd_webapp(_args()) == 0
    assert world["calls"]["opened"] == [url]
    assert "sign in" in capsys.readouterr().out.lower()
    assert launcher.cmd_webapp(_args(no_open=True)) == 0
    out = capsys.readouterr().out
    assert url in out and "sign in" in out.lower() and "one-use" not in out


def test_no_open_prints_the_link_instead(world, capsys):
    assert launcher.cmd_webapp(_args(no_open=True)) == 0
    assert world["calls"]["opened"] == []
    out = capsys.readouterr().out
    assert LINK in out and "60 seconds" in out


def test_a_browser_that_will_not_open_leaves_the_link_on_screen(world, capsys):
    world["monkeypatch"].setattr(launcher, "_open", lambda url: False)
    assert launcher.cmd_webapp(_args()) == 0
    assert LINK in capsys.readouterr().out


def test_build_only_builds_without_a_gateway(world):
    assert launcher.cmd_webapp(_args(build_only=True)) == 0
    assert (world["calls"]["build"], world["calls"]["ensure"], world["calls"]["control"]) == (1, [], [])


def test_status_reads_the_running_gateway_without_starting_one(world, capsys):
    assert launcher.cmd_webapp(_args(status=True)) == 0
    assert world["calls"]["ensure"] == []
    assert world["calls"]["control"] == [(world["root"], "webapp-status")]
    assert "http://127.0.0.1:9120" in capsys.readouterr().out


def test_status_of_a_gateway_still_starting_the_webapp_says_so(world, capsys):
    """A gateway answers ``starting`` until its Webapp listener settles: that is neither a failure nor a
    reason to restart it."""
    world["answers"]["webapp-status"] = {"ok": True, "result": {"state": "starting", "url": None, "reason": None}}
    assert launcher.cmd_webapp(_args(status=True)) == 0
    out = capsys.readouterr().out
    assert "still starting" in out and "Unavailable" not in out and launcher.RESTART_COMMAND not in out


OLDER = {"webapp-status": {"ok": False, "error": "unknown verb: 'webapp-status'", "supported_verbs": ["identify"]},
         "webapp-launch": {"ok": False, "error": "unknown verb: 'webapp-launch'", "supported_verbs": ["identify"]},
         "webapp-sign-out": {"ok": False, "error": "unknown verb: 'webapp-sign-out'", "supported_verbs": ["identify"]}}


@pytest.mark.parametrize("action", [{"status": True}, {}, {"sign_out": True}])
def test_a_gateway_older_than_the_webapp_is_named_with_the_restart_that_updates_it(world, capsys, action):
    """A gateway started before an upgrade (a git pull, a service manager that kept the old process)
    does not know the Webapp's commands and answers "unknown verb". The command says the running
    gateway is older and names the restart that loads this version, not the raw protocol error."""
    world["answers"].update(OLDER)
    world["states"][:] = ["ready"]
    assert launcher.cmd_webapp(_args(**action)) == 1
    out = capsys.readouterr().out
    assert "unknown verb" not in out and "older" in out and launcher.RESTART_COMMAND in out


def test_status_with_no_gateway_running_says_so(world, capsys):
    world["answers"]["webapp-status"] = launcher.GatewayUnreachable("no gateway")
    assert launcher.cmd_webapp(_args(status=True)) == 1
    assert "gateway is not running" in capsys.readouterr().out


def test_status_with_no_gateway_running_and_the_webapp_off_says_how_to_turn_it_on(world, capsys):
    (world["root"] / "config.yaml").write_text("", encoding="utf-8")
    world["answers"]["webapp-status"] = launcher.GatewayUnreachable("no gateway")
    assert launcher.cmd_webapp(_args(status=True)) == 1
    out = capsys.readouterr().out
    assert "webapp.enabled: true" in out and str(world["root"] / "config.yaml") in out
    assert "Run hermes webapp" not in out


def test_sign_out_replaces_the_browser_credential_on_the_running_gateway(world, capsys):
    world["answers"]["webapp-sign-out"]["result"] = {
        "signed_out": True, "serving": True, "sign_in_gate": False, "connections_closed": 3}
    assert launcher.cmd_webapp(_args(sign_out=True)) == 0
    assert world["calls"]["control"] == [(world["root"], "webapp-sign-out")]
    assert world["calls"]["ensure"] == []
    out = capsys.readouterr().out
    assert "Signed out of every Webapp tab" in out and "closed 3 browser connections" in out


def test_sign_out_behind_the_sign_in_gate_says_browsers_must_sign_in_again(world, capsys):
    """It says exactly what ended: every browser login and the account tokens it held, which work on
    the Webapp only as that login. Native apps' own sign-ins stay."""
    world["answers"]["webapp-sign-out"]["result"] = {
        "signed_out": True, "serving": True, "sign_in_gate": True, "connections_closed": 1}
    assert launcher.cmd_webapp(_args(sign_out=True)) == 0
    out = capsys.readouterr().out
    assert "closed 1 browser connection." in out and "must sign in again" in out
    assert "tokens" in out and "Native apps" in out and "stay signed in" in out


def test_sign_out_on_a_gateway_not_serving_the_webapp_says_no_browser_was_connected(world, capsys):
    world["answers"]["webapp-sign-out"]["result"] = {
        "signed_out": True, "serving": False, "sign_in_gate": False, "connections_closed": 0}
    assert launcher.cmd_webapp(_args(sign_out=True)) == 0
    out = capsys.readouterr().out
    assert "not serving the Webapp" in out and "Signed out of every" not in out


def test_sign_out_on_a_gateway_that_does_not_report_what_it_closed_asks_for_a_restart(world, capsys):
    # A gateway started before this Hermes signs out loopback tabs only.
    world["answers"]["webapp-sign-out"]["result"] = {"signed_out": True}
    assert launcher.cmd_webapp(_args(sign_out=True)) == 0
    out = capsys.readouterr().out
    assert launcher.RESTART_COMMAND in out and "Signed out of every" not in out


def test_a_sign_out_the_gateway_refuses_is_reported_as_not_done(world, capsys):
    from gateway.webapp_listener import TOKEN_PATH, browser_session_token
    before = browser_session_token(world["root"])
    world["answers"]["webapp-sign-out"] = {"ok": False, "error": "PermissionError: authenticated private request required"}
    assert launcher.cmd_webapp(_args(sign_out=True)) == 1
    out = capsys.readouterr().out
    assert "authenticated private request required" in out and "Signed out" not in out
    assert (world["root"] / TOKEN_PATH).read_text().strip() == before


def test_a_gateway_that_will_not_start_is_reported(world, capsys):
    def refuse(home):
        raise launcher.GatewayUnreachable("The gateway did not start (inaccessible: lock owned by another user).")

    world["monkeypatch"].setattr(launcher, "_ensure", refuse)
    assert launcher.cmd_webapp(_args()) == 1
    assert "lock owned by another user" in capsys.readouterr().out
    assert world["calls"]["control"] == []


def test_signing_out_with_no_gateway_running_still_replaces_the_stored_credential(world, capsys):
    from gateway.webapp_listener import TOKEN_PATH, browser_session_token
    before = browser_session_token(world["root"])
    world["answers"]["webapp-sign-out"] = launcher.GatewayUnreachable("no gateway")
    assert launcher.cmd_webapp(_args(sign_out=True)) == 0
    assert (world["root"] / TOKEN_PATH).read_text().strip() != before
    out = capsys.readouterr().out
    assert "No gateway is running" in out and "sign in again" in out


def test_signing_out_with_no_gateway_and_a_credential_that_cannot_be_stored_says_nobody_was(world, capsys):
    """With no gateway running, sign-out replaces the stored credential itself. When the home cannot
    store one (a stray file where its folder goes, a folder another user owns), it says nobody was
    signed out and names the file."""
    from gateway.webapp_listener import TOKEN_PATH
    (world["root"] / "webapp").write_text("not a folder\n", encoding="utf-8")
    world["answers"]["webapp-sign-out"] = launcher.GatewayUnreachable("no gateway")
    assert launcher.cmd_webapp(_args(sign_out=True)) == 1
    out = capsys.readouterr().out
    assert "Nobody was signed out" in out and str(world["root"] / TOKEN_PATH) in out


STARTING = {"ok": False, "error": "RuntimeError: The Webapp is still starting; try again in a moment."}
SIGNED_OUT = {"ok": True, "result": {"signed_out": True, "serving": True, "sign_in_gate": False,
                                     "connections_closed": 1}}


@pytest.mark.parametrize("first", [STARTING, launcher.GatewayUnreachable("control socket not up yet")])
def test_sign_out_waits_for_a_starting_gateway_and_signs_out_there(world, capsys, first):
    """A starting gateway may already hold the stored credential in memory, so rewriting the file
    would sign nobody out: the command waits for the gateway to answer instead."""
    from gateway.webapp_listener import TOKEN_PATH, browser_session_token
    before = browser_session_token(world["root"])
    world["answers"]["webapp-sign-out"] = [first, first, SIGNED_OUT]
    world["states"][:] = ["starting"]
    assert launcher.cmd_webapp(_args(sign_out=True)) == 0
    out = capsys.readouterr().out
    assert "Signed out of every Webapp tab: closed 1 browser connection." in out
    assert [verb for _, verb in world["calls"]["control"]] == ["webapp-sign-out"] * 3
    assert (world["root"] / TOKEN_PATH).read_text().strip() == before


def test_sign_out_on_a_gateway_that_stays_starting_says_nobody_was_signed_out(world, capsys):
    from gateway.webapp_listener import TOKEN_PATH, browser_session_token
    before = browser_session_token(world["root"])
    world["answers"]["webapp-sign-out"] = [STARTING]
    world["states"][:] = ["starting"]
    assert launcher.cmd_webapp(_args(sign_out=True)) == 1
    out = capsys.readouterr().out
    assert "still starting" in out and "nobody was signed out" in out.lower() and "Signed out" not in out
    assert (world["root"] / TOKEN_PATH).read_text().strip() == before
    assert sum(world["calls"]["slept"]) >= 20  # it waited before giving up


def test_sign_out_on_a_gateway_that_does_not_answer_signs_nobody_out(world, capsys):
    from gateway.webapp_listener import TOKEN_PATH, browser_session_token
    before = browser_session_token(world["root"])
    world["answers"]["webapp-sign-out"] = launcher.GatewayUnreachable("timed out")
    world["states"][:] = ["inaccessible"]
    assert launcher.cmd_webapp(_args(sign_out=True)) == 1
    out = capsys.readouterr().out
    assert "nobody was signed out" in out.lower() and "Signed out" not in out
    assert (world["root"] / TOKEN_PATH).read_text().strip() == before


def test_a_home_whose_gateway_never_ran_reads_as_having_none(tmp_path):
    # Read from the home's real gateway lock and control socket, as --sign-out does.
    assert launcher._gateway_state(_home(tmp_path / "idle")) == "absent"


@pytest.mark.parametrize("deleted", ["removed", "tombstoned"])
@pytest.mark.parametrize("action", [{"sign_out": True}, {"status": True}, {}])
def test_a_hermes_home_at_a_deleted_profile_is_refused_like_its_profile_flag(world, capsys, deleted, action):
    """``HERMES_HOME`` naming a deleted profile is refused exactly as ``-p <name>`` is, so signing
    out never replaces the default gateway's credential instead."""
    from gateway.webapp_listener import TOKEN_PATH, browser_session_token
    from hermes_constants import mark_named_profile_deleted

    gone = world["root"] / "profiles" / "gone"
    if deleted == "tombstoned":
        _home(gone)
        mark_named_profile_deleted(gone)
    before = browser_session_token(world["root"])
    _use_home(world, gone)
    assert launcher.cmd_webapp(_args(**action)) == 1
    refused = capsys.readouterr()
    assert "Error: Profile 'gone' does not exist. Create it with: hermes profile create gone" in refused.err
    assert "Signed out" not in refused.out
    assert (world["calls"]["control"], world["calls"]["ensure"], world["calls"]["build"]) == ([], [], 0)
    assert (world["root"] / TOKEN_PATH).read_text().strip() == before


def test_the_webapp_command_takes_one_action_at_a_time():
    import argparse
    from hermes_cli.subcommands.webapp import build_webapp_parser
    parser = argparse.ArgumentParser()
    seen = []
    build_webapp_parser(parser.add_subparsers(), cmd_webapp=seen.append)
    args = parser.parse_args(["webapp", "--no-open", "--force-build"])
    args.func(args)
    assert vars(seen[0]) | {"func": None} == vars(_args(no_open=True, force_build=True)) | {"func": None}
    for clash in (["--status", "--sign-out"], ["--status", "--build-only"], ["--sign-out", "--build-only"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["webapp", *clash])


def test_a_link_for_the_user_to_open_by_hand_is_printed_and_never_handed_to_a_browser_launcher(world, capsys):
    """Where the gateway cannot check who redeems a launch link (macOS, the BSDs) it marks the link
    private. With no browser to carry it there in a user-only page, the command prints it with its
    lifetime and never passes it to a browser launcher, whose arguments other local users can read."""
    world["answers"]["webapp-launch"] = {"ok": True, "result": {"url": LINK, "private": True,
                                                                "expires_in_seconds": 60}}
    world["monkeypatch"].setattr(launcher, "_open", lambda url: world["calls"]["opened"].append(url) and False)
    assert launcher.cmd_webapp(_args()) == 0
    assert all("ticket" not in url for url in world["calls"]["opened"])
    out = capsys.readouterr().out
    assert LINK in out and "keep it private" in out and "60 seconds" in out and "once" in out
