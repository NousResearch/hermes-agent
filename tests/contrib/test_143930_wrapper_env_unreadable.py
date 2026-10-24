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
    """child_env: dict the wrapper's *child* exposes (wrapper itself: {}).

    plist_home: one plist when a string, or one launchd job per list entry —
    every job's ``ProgramArguments`` equal ``WRAPPER_ARGV`` (the JXA is what
    the plist leg matches on), each pinning its own ``HERMES_HOME``.
    """
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
                    # One fake LaunchAgent per matching label (label -> plist file name),
                    # so dedupe cases can pin several jobs with the same JXA.
                    labels = ["ai.hermes.gateway"]
                    if "ai.hermes.gateway-legacy".startswith(stem) or stem == "":
                        labels.append("ai.hermes.gateway-legacy")
                    return [FakePath(str(self) + "/" + label + ".plist") for label in labels]
                return []

        gateway_mod.get_launchd_plist_path = lambda: FakePath(
            "/Users/dev/Library/LaunchAgents/ai.hermes.gateway.plist")
        gateway_mod.launchd_gateway_labels_for_install = list
        gateway_mod.legacy_launchd_labels_for_install = lambda exclude=None: []
        gateway_mod._locate_launchd_gateway_service = lambda label: (None, 0)
        sys.modules["hermes_cli.gateway"] = gateway_mod

        real_open = open

        def fake_open(file, *a, **kw):
            if str(file).endswith(".plist"):
                import io
                # The label marker rides in the bytes so fake_load can tell
                # which LaunchAgent the resolver is reading (path is gone by then).
                marker = "gateway-legacy" if str(file).endswith("gateway-legacy.plist") else ""
                return io.BytesIO(f"<plist><dict/><!--{marker}--></plist>".encode())
            return real_open(file, *a, **kw)

        builtins_mod = __import__("builtins")
        builtins_mod.open = fake_open

        # Real plistlib.loads parses the fake bytes: the label selects which
        # job the resolver is reading, so each label can pin its own home.
        def fake_load(fh, *a, **kw):
            raw = fh.read().decode("utf-8")
            plist_homes = plist_home if isinstance(plist_home, list) else [plist_home]
            if "gateway-legacy" in raw:
                return {"ProgramArguments": list(WRAPPER_ARGV),
                        "EnvironmentVariables": {"HERMES_HOME": plist_homes[-1]}}
            return {"ProgramArguments": list(WRAPPER_ARGV),
                    "EnvironmentVariables": {"HERMES_HOME": plist_homes[0]}}

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
    # Case E: legacy label migration leaves TWO plists with the same JXA and
    # the same HERMES_HOME; the dedupe must still resolve instead of treating
    # the repeat as ambiguity. (Red with the dedupe dropped: homes == [h, h].)
    dp = install_stubs(None, ["/Users/dev/.hermes"])
    results.append(run_case(dp, "E.duplicate_same_home_plists_resolve",
                            "/Users/dev/.hermes"))
    # Case F: two labels pinning DIFFERENT homes with one JXA is genuine
    # ambiguity — the leg must stay silent rather than guess; the resolver
    # falls back to the root home exactly like the no-proof guard case A.
    dp = install_stubs(None, ["/Users/dev/.hermes", "/Users/dev/.hermes/profiles/other"])
    results.append(run_case(dp, "F.conflicting_homes_stay_rejected",
                            os.path.expanduser("~/.hermes")))
    # Case C (RED on main): neither wrapper nor child proves it; plist decides.
    dp = install_stubs(None, "/Users/dev/.hermes")
    results.append(run_case(dp, "C.plist_env_carries_home", "/Users/dev/.hermes"))
    print(f"RESULT {sum(results)}/{len(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
