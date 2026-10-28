"""Clients adopt the Home Manager module's launchd job as the gateway service (#143089).

Without it a client finds no service, starts a detached gateway, and the module's job exits 75
beside it forever.
"""
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

LABEL = "org.nix-community.home.hermes-agent"
HERMES = "/nix/store/0000000000000000000000000000000-hermes-agent-0.0.0/bin/hermes"


def _wait4path(script: str) -> list[str]:
    """ProgramArguments as Home Manager's launchd module (modules/launchd/default.nix) writes them."""
    return ["/bin/sh", "-c", "/bin/wait4path /nix/store && exec " + script]


def _launchctl_print(argv: list[str], home: Path, *, supervised: bool = True) -> str:
    """``launchctl print`` of a loaded job, in the layout launchd prints it."""
    env = [f"HERMES_HOME => {home}", "HERMES_MANAGED => home-manager"]
    if supervised:
        env.append("HERMES_SUPERVISED_CHILD => 1")
    return (f"gui/{os.getuid()}/{LABEL} = {{\n\tstate = running\n\n\tprogram = {argv[0]}\n\targuments = {{\n"
            + "".join(f"\t\t{a}\n" for a in argv) + "\t}\n\n\tinherited environment = {\n\t\tPATH => /usr/bin\n\t}\n\n"
            "\tenvironment = {\n" + "".join(f"\t\t{e}\n" for e in env) + "\t}\n}\n")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("script,supervised,reason", [
    (f"{HERMES} gateway run", True, None),
    (f"{HERMES} gateway run --quiet", True, None),
    (f"{HERMES} gateway", True, "service_identity_unverified"),
    (f"{HERMES} gateway run", False, "service_identity_unverified"),
    (f"{HERMES} gateway run; /bin/echo pwned", True, "service_identity_unverified"),
    (f"{HERMES} gateway run && /bin/echo pwned", True, "service_identity_unverified"),
    (f"{HERMES} gateway run $(/bin/echo pwned)", True, "service_identity_unverified"),
    # Only the regenerated quoting catches these: the inner argv still names a `hermes` that runs `gateway run`.
    ("/nix/store/$(id)/bin/hermes gateway run", True, "service_identity_unverified"),
    ("/nix/store/`id`/bin/hermes gateway run", True, "service_identity_unverified"),
    (f"'{HERMES}' gateway run", True, "service_identity_unverified"),
    (f"/bin/echo {HERMES} gateway run", True, "service_identity_unverified"),
])
def test_home_manager_wait4path_job_verifies_only_as_the_exact_wrapper(script, supervised, reason, tmp_path):
    """The wrapper is unwrapped only when Nix's own quoting regenerates the script, and the command
    inside must pass the same gateway identity as an unwrapped job, as loaded and on disk."""
    from hermes_cli.gateway_runtime_service_identity import verify_launchd_loaded, verify_launchd_plist
    home = tmp_path / ".hermes"
    argv = _wait4path(script)
    env = {"HERMES_HOME": str(home), **({"HERMES_SUPERVISED_CHILD": "1"} if supervised else {})}
    checks = (
        lambda: verify_launchd_loaded(_launchctl_print(argv, home, supervised=supervised), home,
                                      uid=os.getuid(), username="owner"),
        lambda: verify_launchd_plist(dict(Label=LABEL, ProgramArguments=argv, EnvironmentVariables=env), LABEL, home),
    )
    for check in checks:
        if reason:
            with pytest.raises(ValueError, match=reason):
                check()
        else:
            check()


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("marker", ["home-manager", None, "nixos"])
def test_client_kickstarts_the_home_manager_job_instead_of_starting_its_own(marker, tmp_path, monkeypatch):
    """A home the module manages is served by its job: discovery returns that job's kickstart. A
    home it does not manage never adopts a job under the module's label."""
    import pwd
    from hermes_cli import gateway_runtime_service as service
    home = tmp_path / ".hermes"
    home.mkdir(mode=0o700)
    if marker:
        (home / ".managed").write_text(marker, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: SimpleNamespace(pw_dir=str(tmp_path), pw_name="owner"))
    loaded = _launchctl_print(_wait4path(f"{HERMES} gateway run"), home)
    queried = []

    def launchctl(argv, *_args, **_kwargs):
        assert argv[:2] == ["launchctl", "print"]
        queried.append(argv[2])
        if argv[2] == f"gui/{os.getuid()}/{LABEL}":
            return SimpleNamespace(returncode=0, stdout=loaded)
        return SimpleNamespace(returncode=113, stdout="")

    monkeypatch.setattr(service, "_run", launchctl)
    found = service.discover_existing_gateway_service(home, deadline=time.monotonic() + 5)
    if marker == "home-manager":
        assert found == service.ExistingService("launchd", ("launchctl", "kickstart", f"gui/{os.getuid()}/{LABEL}"), True)
    else:
        assert found is None
        assert not any(target.endswith("/" + LABEL) for target in queried)
