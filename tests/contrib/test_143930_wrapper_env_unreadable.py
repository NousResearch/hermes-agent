"""RED for #143930: an osascript-wrapped launchd gateway whose own environment is
unreadable (macOS platform binary exposes 0 variables) must still resolve to the
Hermes home it actually serves — via the wrapper's child-process environment, then
the job plist's EnvironmentVariables — instead of silently attributing the gateway
to the root home. Live receipt (strzhao Mac, 2026-10-24): the wrapper's environ
reads as {} and the resolver returns the root home; its stderr_timestamp child
carries the real HERMES_HOME. That silent mis-attribution makes `hermes update`
send every drain request to the default gateway and hard-stop profile gateways.

Run: .venv/bin/python tests/contrib/test_143930_wrapper_env_unreadable.py
"""
import os
import plistlib
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

WRAPPER_PID = 4242
CHILD_PID = 4243
JXA = ('ObjC.import("stdlib"); const status=$.system("exec /usr/local/bin/hermes '
       '--run-module hermes_cli.stderr_timestamp -- /usr/local/bin/hermes gateway '
       'run --external-supervisor");')
WRAPPER_ARGV = ["/usr/bin/osascript", "-l", "JavaScript", "-e", JXA]
CHILD_ENV = {"HERMES_HOME": "/Users/dev/.hermes/profiles/work", "PATH": "/usr/bin:/bin"}


class NoSuch(Exception):
    pass


def install_stubs(child_env, plist_home, child_cmdline=None):
    """child_env: dict the wrapper's *child* exposes (wrapper itself: {})."""
    from hermes_cli import dashboard_procs as dp

    fake_psutil = types.ModuleType("psutil")

    def process(pid):
        if pid == WRAPPER_PID:
            return types.SimpleNamespace(
                cmdline=lambda: list(WRAPPER_ARGV),
                environ=dict,  # platform binary: 0 variables (live receipt)
                children=lambda recursive=False: [process(CHILD_PID)] if child_env else [],
            )
        if pid == CHILD_PID:
            return types.SimpleNamespace(
                cmdline=lambda: list(child_cmdline or
                                     ["/usr/local/bin/hermes", "--run-module",
                                      "hermes_cli.stderr_timestamp"]),
                environ=lambda: dict(child_env or {}),
            )
        raise NoSuch(pid)

    fake_psutil.NoSuch = NoSuch
    fake_psutil.Process = process
    sys.modules["psutil"] = fake_psutil

    # wrapper env unreadable; child readable (plain dict path)
    dp._pid_environ = lambda pid: ({}) if pid == WRAPPER_PID else dict(child_env or {})

    main_dash = types.ModuleType("hermes_cli.main_dashboard")
    main_dash._dashboard_cmdline_for_pid = lambda pid: list(WRAPPER_ARGV)
    sys.modules["hermes_cli.main_dashboard"] = main_dash

    profiles_mod = types.ModuleType("hermes_cli.profiles")
    profiles_mod.get_active_profile = lambda root: "default"
    profiles_mod.normalize_profile_name = lambda name: name or "default"
    profiles_mod.profile_root_for_env_home = lambda env_home, default_home: default_home
    sys.modules["hermes_cli.profiles"] = profiles_mod

    if plist_home is not None:
        gateway_mod = types.ModuleType("hermes_cli.gateway")

        class FakePath(str):
            @property
            def parent(self):
                return FakeParentsDir(str(self).rsplit("/", 1)[0])

            def with_name(self, name):
                return FakePath(str(self).rsplit("/", 1)[0] + "/" + name)

        class FakeParentsDir(str):
            def glob(self, pattern):
                # honor the pattern's stem so mutant globs (zz-nonexistent-*) match nothing
                stem = pattern.split("/")[-1].split("*")[0]
                if "ai.hermes.gateway".startswith(stem) or stem == "":
                    return [FakePath(str(self) + "/" + "ai.hermes.gateway.plist")]
                return []

        gateway_mod.get_launchd_plist_path = lambda: FakePath(
            "/Users/dev/Library/LaunchAgents/ai.hermes.gateway.plist")
        gateway_mod.launchd_gateway_labels_for_install = list
        gateway_mod.legacy_launchd_labels_for_install = lambda exclude=None: []
        gateway_mod._locate_launchd_gateway_service = lambda label: (None, 0)
        sys.modules["hermes_cli.gateway"] = gateway_mod

        real_open = open

        def fake_open(file, *a, **kw):
            if str(file).endswith("ai.hermes.gateway.plist"):
                import io
                return io.BytesIO(b"<plist><dict/></plist>")
            return real_open(file, *a, **kw)

        builtins_mod = __import__("builtins")
        builtins_mod.open = fake_open

        def fake_load(fp, *a, **kw):
            return {"ProgramArguments": list(WRAPPER_ARGV),
                    "EnvironmentVariables": {"HERMES_HOME": plist_home}}

        plistlib.load = fake_load
    return dp


def run_case(dp, name, expect):
    got = dp._hermes_home_for_pid(WRAPPER_PID)
    ok = got == expect
    print(("PASS " if ok else "FAIL ") + name + f" :: got={got!r} expect={expect!r}")
    return ok


def test_wrapper_env_unreadable() -> None:
    """Pytest entry: the manual-case battery must stay green.

    ``main()`` reports via ``sys.exit``; a bare ``SystemExit(0)`` still fails a
    pytest case, so it is caught and asserted instead.
    """
    try:
        main()
    except SystemExit as e:
        assert e.code == 0, "wrapper-env battery failed (see captured stdout)"


def main():
    results = []
    # Case A (guard): child + plist both silent -> unchanged fail-open fallback to the
    # root home (no home can be proven; the plain resolver path must stay intact).
    dp = install_stubs(None, None)
    results.append(run_case(dp, "A.no_proofs_root_fallback_unchanged",
                            os.path.expanduser("~/.hermes")))
    # Case B (RED on main): wrapper env unreadable but child carries HERMES_HOME.
    dp = install_stubs(CHILD_ENV, None)
    results.append(run_case(dp, "B.child_env_carries_home", CHILD_ENV["HERMES_HOME"]))
    # Case D: a non-hermes child carrying HERMES_HOME must NOT be adopted
    # (the child walk only credits hermes processes). Runs before the plist
    # case: install_stubs(_, None) does not undo a previous case's plist
    # stubs, so the no-plist cases must all come first.
    dp = install_stubs(CHILD_ENV, None,
                       child_cmdline=["/usr/bin/osascript", "-l", "JavaScript",
                                      "-e", 'ObjC.import("cocoa"); 1'])
    results.append(run_case(dp, "D.non_hermes_child_not_adopted",
                            os.path.expanduser("~/.hermes")))
    # Case C (RED on main): neither wrapper nor child proves it; plist decides.
    dp = install_stubs(None, "/Users/dev/.hermes")
    results.append(run_case(dp, "C.plist_env_carries_home", "/Users/dev/.hermes"))
    print(f"RESULT {sum(results)}/{len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
