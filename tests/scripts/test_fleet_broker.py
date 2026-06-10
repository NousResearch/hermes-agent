"""Tests for the fleet broker's allowlist — the security boundary that bounds
what an inbound hermes-ops SSH session can do."""
import importlib.util
from pathlib import Path

import pytest

_BROKER = Path(__file__).resolve().parents[2] / "scripts" / "fleet_broker.py"
_spec = importlib.util.spec_from_file_location("fleet_broker", _BROKER)
fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fb)


# ── parse(): only allowlisted subcommands survive ──

@pytest.mark.parametrize("cmd,expected", [
    ("ps", ("ps", [])),
    ("up acme", ("up", ["acme"])),
    ("logs acme 50", ("logs", ["acme", "50"])),
    ("fleet ps", ("ps", [])),                  # leading "fleet" tolerated
    ("hermes fleet up acme", ("up", ["acme"])),  # full prefix tolerated
])
def test_parse_accepts_allowlisted(cmd, expected):
    assert fb.parse(cmd) == expected


@pytest.mark.parametrize("cmd", [
    "",                       # empty → no command
    "id",                     # not allowlisted
    "rm -rf /",               # not allowlisted
    "cat /etc/shadow",        # not allowlisted
    "status; cat /etc/shadow",  # injection: shlex → 'status;' is unknown
    "'unterminated",          # unparseable quoting
])
def test_parse_rejects_everything_else(cmd):
    with pytest.raises(SystemExit) as e:
        fb.parse(cmd)
    assert e.value.code != 0


def test_injection_tokens_never_reach_a_shell():
    # 'up && reboot' parses to subcommand 'up' with args ['&&', 'reboot'];
    # cmd_up then rejects '&&' as an invalid slug — no shell ever sees it.
    sub, args = fb.parse("up && reboot")
    assert sub == "up"
    with pytest.raises(SystemExit):
        fb.cmd_up(args, {"CLIENTS_COMPOSE": "/nope"})


# ── slug validation ──

@pytest.mark.parametrize("name", ["acme", "a", "client-1", "a0-b1"])
def test_valid_slugs(name):
    assert fb._valid_slug(name) == name


@pytest.mark.parametrize("name", [
    "../etc", "a b", "ACME", "-leading", "a/b", "a;b", "a$b", "x" * 40, "",
])
def test_invalid_slugs_die(name):
    with pytest.raises(SystemExit):
        fb._valid_slug(name)


# ── handlers build fixed argv (never a shell string) ──

@pytest.fixture
def captured_run(monkeypatch):
    calls = []
    monkeypatch.setattr(fb, "_run", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(fb, "DOCKER", "docker")
    return calls


def test_cmd_up_builds_compose_argv(captured_run, tmp_path):
    compose = tmp_path / "docker-compose.clients.yml"
    compose.write_text("services: {}\n")
    fb.cmd_up(["acme"], {"CLIENTS_COMPOSE": str(compose)})
    assert captured_run == [["docker", "compose", "-f", str(compose),
                             "up", "-d", "hermes-acme"]]


def test_cmd_up_refuses_when_compose_missing(captured_run):
    with pytest.raises(SystemExit):
        fb.cmd_up(["acme"], {"CLIENTS_COMPOSE": "/does/not/exist.yml"})
    assert captured_run == []   # nothing ran


def test_cmd_logs_bounds_line_count(captured_run):
    fb.cmd_logs(["acme"], {})                       # default 100
    assert captured_run[-1] == ["docker", "logs", "--tail", "100", "hermes-acme"]
    fb.cmd_logs(["acme", "2000"], {})               # max allowed
    assert captured_run[-1][3] == "2000"
    for bad in ("0", "2001", "abc"):
        with pytest.raises(SystemExit):
            fb.cmd_logs(["acme", bad], {})


def test_status_and_down_target_namespaced_container(captured_run, tmp_path):
    compose = tmp_path / "c.yml"; compose.write_text("x")
    fb.cmd_status(["acme"], {})
    assert "hermes-acme" in captured_run[-1]
    fb.cmd_down(["acme"], {"CLIENTS_COMPOSE": str(compose)})
    assert captured_run[-1] == ["docker", "compose", "-f", str(compose),
                                "rm", "-sf", "hermes-acme"]
