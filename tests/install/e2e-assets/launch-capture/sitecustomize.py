"""Driver-side spawn interception for hermes desktop E2E legs.

Pre-PM ``hermes`` is a venv console script that imports this module at
startup via ``PYTHONPATH``. PM launchers use -I and instead load it via
the driver-side pm-launch.py before the installed bootstrap. With an
explicit env-var opt-in the module wraps
``subprocess.run`` so the FINAL electron launch call of ``hermes
desktop`` is captured -- argv, cwd, and the fully-constructed ``env``
kwarg written to a JSON spec -- and replaced with a fake success instead
of spawning. Everything before the spawn (build, stamps, integrity gate,
sandbox fixup) runs for real, in the REAL installed code of whatever
version is under test; Playwright's ``_electron.launch`` then owns the
app from spawn using exactly the spec the product would have used.

This ships with the DRIVER, never with the product, so it works
identically on every sampled OLD ref -- version drift in the launch
shapes is the matcher's problem, which lives here, next to the driver
(the same maintenance model as ``installer_supports()``).

Opt-in: ``HERMES_E2E_CAPTURE_LAUNCH=<path>`` -- the spec is written
there, and the marker file ``<path>.captured`` distinguishes "hermes
desktop exited 0 and we captured" from "exited 0 without reaching a
launch" (a version that errors out earlier must FAIL the leg, loudly).

Launch shapes across sampled desktop-era tags (verified against each
tag's own hermes_cli/main.py):

  v2026.6.5    subprocess.run([npm, "exec", "--", "electron", "."], ...)
  v0.20.0+     the npm-exec form AND subprocess.run(launch_command, ...)
               where launch_command[0] is the packaged app executable
               under apps/desktop/release/

  3ddf82fe249+ on Windows: the packaged launch is a detached
               subprocess.Popen(launch_command, creationflags=...) and
               ``hermes desktop`` exits without waiting for it

Every shape passes an explicit ``env=`` kwarg, so both ``subprocess.run``
and ``subprocess.Popen`` are intercepted. npm BUILD calls (``npm run
build`` / ``npm run pack``) carry no ``electron`` token in argv and pass
through untouched -- they must run for real.
"""

import os

_SPEC_PATH = os.environ.get("HERMES_E2E_CAPTURE_LAUNCH")

if _SPEC_PATH:
    import json
    import subprocess

    _SPEC: str = _SPEC_PATH
    _real_run = subprocess.run

    def _basename_noext(token: str) -> str:
        base = os.path.basename(str(token))
        for ext in (".exe", ".cmd", ".bat"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        return base.lower()

    def _match_shape(argv: "list[str]") -> str:
        """Return the launch shape for argv, or '' when it is not a launch."""
        if not argv:
            return ""
        tokens = [str(t) for t in argv]
        head = _basename_noext(tokens[0])
        # Source shape: npm/npx invoking electron ("npm exec -- electron .").
        # Membership, not position: absorb argv drift across versions. Build
        # calls ("npm run pack") carry no bare "electron" token.
        if head in ("npm", "npx"):
            if any(_basename_noext(t) == "electron" for t in tokens[1:]):
                return "source"
            return ""
        # Packaged shape: argv[0] is the packaged app executable under
        # apps/desktop/release/ (win-unpacked/Hermes.exe, linux-unpacked/...,
        # mac*/Hermes.app/Contents/MacOS/...).
        first = tokens[0].replace("\\", "/")
        if "apps/desktop/release/" in first:
            return "packaged"
        return ""

    def _launch_shape(args, kwargs) -> "tuple[list[str], str]":
        argv = args[0] if args else kwargs.get("args")
        if not isinstance(argv, (list, tuple)):
            return [], ""
        tokens = [str(t) for t in argv]
        return tokens, _match_shape(tokens)

    def _capture(tokens: "list[str]", kwargs: dict, shape: str) -> None:
        env = kwargs.get("env")
        spec = {
            "argv": tokens,
            "cwd": str(kwargs.get("cwd") or os.getcwd()),
            # Capture what the child would ACTUALLY get: the constructed
            # env= when present, the ambient environment when not.
            "env": dict(env) if env is not None else dict(os.environ),
            "matchedShape": shape,
        }
        tmp = _SPEC + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2)
        os.replace(tmp, _SPEC)
        with open(_SPEC + ".captured", "w", encoding="utf-8") as fh:
            fh.write(shape)
        print(f"[e2e launch-capture] captured {shape} launch -> {_SPEC} (not spawning)")

    def _capturing_run(*args, **kwargs):
        tokens, shape = _launch_shape(args, kwargs)
        if not shape:
            return _real_run(*args, **kwargs)
        _capture(tokens, kwargs, shape)
        return subprocess.CompletedProcess(tokens, 0, stdout=None, stderr=None)

    _RealPopen = subprocess.Popen

    class _CapturingPopen(_RealPopen):  # type: ignore[misc, valid-type]
        """A launch-shaped Popen records the spec and stands in for an exited child."""

        _e2e_stub = False

        def __init__(self, *args, **kwargs):
            tokens, shape = _launch_shape(args, kwargs)
            if not shape:
                super().__init__(*args, **kwargs)
                return
            _capture(tokens, kwargs, shape)
            self._e2e_stub = True
            self._child_created = False  # Popen.__del__ must not reap a pid we never made.
            self.args = tokens
            self.stdin = self.stdout = self.stderr = None
            self.pid = 0
            self.returncode = 0

        def poll(self):
            return 0 if self._e2e_stub else super().poll()

        def wait(self, timeout=None):
            return 0 if self._e2e_stub else super().wait(timeout)

        def communicate(self, input=None, timeout=None):
            return (None, None) if self._e2e_stub else super().communicate(input, timeout)

        def send_signal(self, sig):
            if not self._e2e_stub:
                super().send_signal(sig)

        def terminate(self):
            if not self._e2e_stub:
                super().terminate()

        def kill(self):
            if not self._e2e_stub:
                super().kill()

    subprocess.run = _capturing_run
    subprocess.Popen = _CapturingPopen  # type: ignore[misc]
