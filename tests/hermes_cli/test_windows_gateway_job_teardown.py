"""Regression tests for #48820 (4th repro): job-object teardown killed the
post-update respawned gateway silently, and the updater printed
"✓ Restarting Windows gateway profile(s)" anyway.

Two fixes under test:

1. ``_spawn_gateway_restart_watcher``'s inlined watcher source must
   (a) route the respawned gateway's stray stdout/stderr to
       ``logs/gateway-stdio.log`` (it was ``DEVNULL`` — a gateway killed by
       parent Job Object teardown left ZERO trace anywhere), and
   (b) stamp ``_HERMES_GATEWAY_BREAKAWAY`` =1/0 on the respawn env exactly
       like the canonical ``gateway_windows._spawn_detached``, so the
       lifecycle/exit-diag records show whether the gateway escaped the
       parent's Job Object.

2. ``_resume_windows_gateways_after_update`` must verify a stable gateway
   process actually exists (via ``gateway_windows._wait_for_gateway_ready``)
   before printing the ✓ — a truthy launch return only proves the watcher
   process was created, not that the respawned gateway survived the
   updater's Job Object teardown.

3. #127089: when the updater itself sits inside a Windows Job Object,
   ``_spawn_gateway_restart_watcher`` hands the respawn to a transient
   Task Scheduler task (outside any Job Object) via
   ``gateway_windows._spawn_gateway_via_transient_task``; on ANY scheduler
   failure it logs to ``logs/gateway-stdio.log`` and falls back to the
   existing breakaway -> non-breakaway Popen chain rather than failing.
"""

import json
import sys
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
from hermes_cli import _subprocess_compat

# ---------------------------------------------------------------------------
# 1. Watcher template contract
# ---------------------------------------------------------------------------

def _captured_watcher_source(monkeypatch) -> str:
    """Spawn the watcher with a mocked Popen and return the inlined -c source."""
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class _P:
            pid = 12345

        return _P()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    # Force the non-job path so the transient handoff never fires here:
    # the template contract is about the breakaway watcher itself.
    monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: False)
    assert gateway._spawn_gateway_restart_watcher(
        999999, ["python", "-m", "hermes_cli.main", "gateway", "run"]
    )
    argv = captured["argv"]
    assert argv[1] == "-c"
    return argv[2]

class TestWatcherRespawnTemplate:

    def test_respawn_source_compiles(self, monkeypatch):
        """The inlined -c template is built via str.format over a
        dedented literal — guard against brace/indentation regressions."""
        src = _captured_watcher_source(monkeypatch)
        compile(src, "<watcher>", "exec")

    def test_respawn_routes_stdio_and_stamps_breakaway(self, monkeypatch):
        """Watcher routes respawn stdio to gateway-stdio.log and stamps
        _HERMES_GATEWAY_BREAKAWAY like _spawn_detached (contract, not snapshot)."""
        src = _captured_watcher_source(monkeypatch)
        assert "gateway-stdio.log" in src
        assert "_HERMES_GATEWAY_BREAKAWAY" in src
        # Both breakaway states are reachable in the template.
        assert '"1"' in src or "'1'" in src or ': "1"' in src or 'BREAKAWAY_ENV: "1"' in src
        assert "windows_detach_flags_without_breakaway" in src

# ---------------------------------------------------------------------------
# 2. Post-update resume liveness gate
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. process_is_in_job() (#127089)
# ---------------------------------------------------------------------------

class TestProcessIsInJob:
    def test_returns_false_on_non_windows(self, monkeypatch):
        """Off Windows the probe is defined to be False (no Job Objects)."""
        monkeypatch.setattr(sys, "platform", "linux")
        assert _subprocess_compat.process_is_in_job() is False

    def test_returns_false_when_probe_reports_not_in_job(self, monkeypatch):
        """IsProcessInJob succeeding with result 0 means not in a job."""
        monkeypatch.setattr(sys, "platform", "win32")

        class FakeKernel:
            def GetCurrentProcess(self):
                return 0xFFFFFFFF

            def IsProcessInJob(self, proc, job, out):
                out._obj.value = 0
                return 1

        class FakeWindll:
            kernel32 = FakeKernel()

        import ctypes

        monkeypatch.setattr(ctypes, "windll", FakeWindll(), raising=False)
        assert _subprocess_compat.process_is_in_job() is False

    def test_returns_true_when_probe_reports_in_job(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")

        class FakeKernel:
            def GetCurrentProcess(self):
                return 0xFFFFFFFF

            def IsProcessInJob(self, proc, job, out):
                out._obj.value = 1
                return 1

        class FakeWindll:
            kernel32 = FakeKernel()

        import ctypes

        monkeypatch.setattr(ctypes, "windll", FakeWindll(), raising=False)
        assert _subprocess_compat.process_is_in_job() is True

    def test_returns_false_when_api_fails(self, monkeypatch):
        """IsProcessInJob returning 0 (failure) fails closed to False."""
        monkeypatch.setattr(sys, "platform", "win32")

        class FakeKernel:
            def GetCurrentProcess(self):
                return 0xFFFFFFFF

            def IsProcessInJob(self, proc, job, out):
                return 0

        class FakeWindll:
            kernel32 = FakeKernel()

        import ctypes

        monkeypatch.setattr(ctypes, "windll", FakeWindll(), raising=False)
        assert _subprocess_compat.process_is_in_job() is False

    def test_returns_false_when_windll_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        import ctypes

        monkeypatch.setattr(ctypes, "windll", None, raising=False)
        assert _subprocess_compat.process_is_in_job() is False


# ---------------------------------------------------------------------------
# 4. Transient task template/behavior (#127089)
# ---------------------------------------------------------------------------

def _launcher_kwargs(task_name="Hermes_GW_Restart_1_abcd1234", temp_dir=r"C:\Temp\hermes-gw-restart-x", watcher_env=None):
    return dict(
        task_name=task_name,
        temp_dir=temp_dir,
        old_pid=4242,
        gateway_cmd=["C:\\venv\\Scripts\\python.exe", "-m", "hermes_cli.main", "gateway", "run"],
        respawn_cwd=r"C:\Users\me\.hermes",
        respawn_env_overlay={"HERMES_HOME": r"C:\Users\me\.hermes", "VIRTUAL_ENV": r"C:\venv"},
        watcher_env=watcher_env,
        project_root=r"C:\hermes\hermes-agent",
        watcher_timeout_s=120.0,
    )


class TestTransientTaskTemplate:
    def test_launcher_source_compiles(self):
        src = gateway_windows._build_transient_task_launcher(**_launcher_kwargs())
        compile(src, "<transient_launcher>", "exec")

    def test_launcher_self_deletes_task_and_cleans_temp(self):
        """Launcher deletes its own task (no orphan triggerless task) and
        removes its temp dir; behavior contract, not a snapshot."""
        src = gateway_windows._build_transient_task_launcher(**_launcher_kwargs())
        assert "schtasks" in src
        assert "/Delete" in src
        assert "/TN" in src
        assert "_TASK_NAME" in src
        # Temp cleanup in finally.
        assert "shutil.rmtree" in src
        assert "_TEMP_DIR" in src
        assert "finally" in src

    def test_launcher_keeps_stdio_and_breakaway_diagnostics(self):
        """Transient launcher keeps the gateway-stdio.log trace and the
        _HERMES_GATEWAY_BREAKAWAY stamp."""
        src = gateway_windows._build_transient_task_launcher(**_launcher_kwargs())
        assert "gateway-stdio.log" in src
        assert "_HERMES_GATEWAY_BREAKAWAY" in src
        assert "windows_detach_flags" in src
        assert "pid_exists_stdlib" in src

    def test_transient_xml_is_triggerless_and_hidden(self):
        xml = gateway_windows._build_transient_task_xml(
            "Hermes_GW_Restart_1_abcd1234", r"C:\venv\Scripts\python.exe", r'"C:\Temp\x\restart_launcher.py"',
            r"DOMAIN\me",
        )
        assert "<Triggers" in xml
        assert "LogonTrigger" not in xml
        assert "<Hidden>true</Hidden>" in xml
        assert "<AllowStartOnDemand>true</AllowStartOnDemand>" in xml
        assert "Hermes_GW_Restart_1_abcd1234" in xml
        assert "python.exe" in xml

    def test_transient_task_name_is_unique(self):
        a = gateway_windows._transient_task_name()
        b = gateway_windows._transient_task_name()
        assert a.startswith("Hermes_GW_Restart_")
        assert a != b


class TestSpawnViaTransientTask:
    def _arrange_success(self, monkeypatch, tmp_path):
        """Stub schtasks success; real temp dir under tmp_path."""
        monkeypatch.setattr(sys, "platform", "win32")
        calls: list[list[str]] = []

        def fake_schtasks(args):
            calls.append(list(args))
            if args[0] == "/Create":
                xml_path = Path(args[args.index("/XML") + 1])
                assert xml_path.exists()
                return (0, "SUCCESS", "")
            if args[0] == "/Run":
                return (0, "SUCCESS", "")
            if args[0] == "/Delete":
                return (0, "SUCCESS", "")
            raise AssertionError(f"unexpected schtasks args: {args}")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)
        monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: r"DOMAIN\me")
        return calls

    def test_create_and_run_success_returns_true(self, monkeypatch, tmp_path):
        calls = self._arrange_success(monkeypatch, tmp_path)
        staging = tmp_path / "staging"
        staging.mkdir()

        import tempfile as _tf

        real_mkdtemp = _tf.mkdtemp

        def fake_mkdtemp(*args, **kwargs):
            kwargs.pop("prefix", None)
            return real_mkdtemp(dir=str(staging))

        monkeypatch.setattr(gateway_windows.tempfile, "mkdtemp", fake_mkdtemp)

        ok = gateway_windows._spawn_gateway_via_transient_task(
            4242, ["C:\\venv\\Scripts\\python.exe", "-m", "hermes_cli.main", "gateway", "run"],
            respawn_cwd="", respawn_env_overlay={}, watcher_env=None, watcher_timeout_s=5,
        )
        assert ok is True
        kinds = [c[0] for c in calls]
        assert kinds[0] == "/Create"
        assert "/Run" in kinds
        # XML staging is removed; the launcher dir is left for the launcher (self-cleans).
        assert not list(staging.rglob("*.task.xml"))
        assert list(staging.rglob("restart_launcher.py"))

    def test_create_failure_raises_and_cleans_temp(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        calls: list[list[str]] = []

        def fake_schtasks(args):
            calls.append(list(args))
            if args[0] == "/Create":
                return (1, "", "Access is denied.")
            raise AssertionError(f"unexpected: {args}")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)
        monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: None)
        import tempfile as _tf

        created: list[str] = []
        real_mkdtemp = _tf.mkdtemp

        def tracking_mkdtemp(**kwargs):
            path = real_mkdtemp(**kwargs)
            created.append(path)
            return path

        monkeypatch.setattr(gateway_windows.tempfile, "mkdtemp", tracking_mkdtemp)
        with pytest.raises(RuntimeError, match="schtasks /Create failed"):
            gateway_windows._spawn_gateway_via_transient_task(
                4242, ["python.exe", "gateway"], watcher_timeout_s=5,
            )
        assert created and not Path(created[0]).exists()

    def test_run_failure_deletes_task_and_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        calls: list[list[str]] = []

        def fake_schtasks(args):
            calls.append(list(args))
            if args[0] == "/Create":
                return (0, "SUCCESS", "")
            if args[0] == "/Run":
                return (1, "", "No valid logon token.")
            if args[0] == "/Delete":
                return (0, "SUCCESS", "")
            raise AssertionError(f"unexpected: {args}")

        monkeypatch.setattr(gateway_windows, "_exec_schtasks", fake_schtasks)
        monkeypatch.setattr(gateway_windows, "_resolve_task_user", lambda: None)
        with pytest.raises(RuntimeError, match="schtasks /Run failed"):
            gateway_windows._spawn_gateway_via_transient_task(
                4242, ["python.exe", "gateway"], watcher_timeout_s=5,
            )
        assert [c[0] for c in calls] == ["/Create", "/Run", "/Delete"]

    def test_rejects_bad_args(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        with pytest.raises(ValueError):
            gateway_windows._spawn_gateway_via_transient_task(0, ["python.exe"])
        with pytest.raises(ValueError):
            gateway_windows._spawn_gateway_via_transient_task(123, [])


# ---------------------------------------------------------------------------
# 5. Restart-watcher Job handoff with fallback (#127089)
# ---------------------------------------------------------------------------

def _fake_popen_factory(calls):
    def fake_popen(argv, **kwargs):
        calls.append((argv, kwargs))

        class _P:
            pid = 9999

        return _P()

    return fake_popen


class TestRestartWatcherJobHandoff:
    def test_no_handoff_when_not_in_job(self, monkeypatch):
        """Not in a job: transient task is never attempted; normal Popen wins."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: False)
        transient_calls: list = []
        monkeypatch.setattr(
            gateway_windows, "_spawn_gateway_via_transient_task",
            lambda *a, **k: transient_calls.append((a, k)) or True,
        )
        monkeypatch.setattr(
            gateway_windows, "windowless_gateway_restart_spec",
            lambda argv: (argv, "", {}),
        )
        calls: list = []
        monkeypatch.setattr(gateway.subprocess, "Popen", _fake_popen_factory(calls))
        assert gateway._spawn_gateway_restart_watcher(111, ["python", "gateway"], host=False) is True
        assert transient_calls == []
        assert len(calls) == 1

    def test_handoff_success_skips_popen(self, monkeypatch):
        """In a job with a working scheduler: transient handoff returns True
        without spawning the in-job watcher."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "windowless_gateway_restart_spec",
            lambda argv: (argv, "", {}),
        )
        transient_calls: list = []
        def fake_transient(old_pid, run_argv, **kwargs):
            transient_calls.append((old_pid, run_argv, kwargs))
            return True

        monkeypatch.setattr(gateway_windows, "_spawn_gateway_via_transient_task", fake_transient)
        calls: list = []
        monkeypatch.setattr(gateway.subprocess, "Popen", _fake_popen_factory(calls))
        assert gateway._spawn_gateway_restart_watcher(222, ["python", "gateway"], host=False) is True
        assert len(transient_calls) == 1
        assert transient_calls[0][0] == 222
        assert calls == []

    def test_handoff_failure_falls_back_and_logs(self, monkeypatch, tmp_path):
        """Scheduler denied (SSH/service, timeout): log to gateway-stdio.log
        and fall back to the breakaway watcher rather than failing."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "windowless_gateway_restart_spec",
            lambda argv: (argv, "", {}),
        )
        def failing_transient(*a, **k):
            raise RuntimeError("schtasks /Run failed (code 1): No valid logon token.")

        monkeypatch.setattr(gateway_windows, "_spawn_gateway_via_transient_task", failing_transient)
        calls: list = []
        monkeypatch.setattr(gateway.subprocess, "Popen", _fake_popen_factory(calls))
        import hermes_constants

        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
        assert gateway._spawn_gateway_restart_watcher(333, ["python", "gateway"], host=False) is True
        # Fallback watcher was spawned.
        assert len(calls) == 1
        # Failure left a trace in the same sidecar log the watcher uses.
        log_path = tmp_path / "logs" / "gateway-stdio.log"
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "transient task handoff failed" in text
        assert "falling back" in text

    def test_no_handoff_on_posix(self, monkeypatch):
        """POSIX never consults the Job probe or the scheduler."""
        monkeypatch.setattr(sys, "platform", "linux")
        probed: list = []
        monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: probed.append(1) or True)
        transient_calls: list = []
        monkeypatch.setattr(
            gateway_windows, "_spawn_gateway_via_transient_task",
            lambda *a, **k: transient_calls.append(1) or True,
        )
        calls: list = []
        monkeypatch.setattr(gateway.subprocess, "Popen", _fake_popen_factory(calls))
        assert gateway._spawn_gateway_restart_watcher(444, ["python", "gateway"], host=False) is True
        assert probed == []
        assert transient_calls == []
        assert len(calls) == 1


@pytest.mark.platforms("windows")
def test_process_is_in_job_live_returns_bool():
    """Native probe smoke: returns a bool on a real Windows host."""
    assert isinstance(_subprocess_compat.process_is_in_job(), bool)


# ---------------------------------------------------------------------------
# 6. Nullable watcher_env launcher execution (#127089 review)
# ---------------------------------------------------------------------------

class TestNullableWatcherEnvExecution:
    def test_none_renders_python_none_literal(self):
        """A named profile passes host=False so watcher_env=None: the
        launcher must render Python-safe ``None``, not JSON ``null``
        (``null`` compiles but raises NameError before _delete_task)."""
        src = gateway_windows._build_transient_task_launcher(**_launcher_kwargs(watcher_env=None))
        assignment = next(
            line for line in src.splitlines() if line.strip().startswith("_WATCHER_ENV =")
        )
        assert assignment.strip() == "_WATCHER_ENV = None"
        assert "null" not in assignment
        # The literal itself evaluates cleanly to None (no NameError).
        namespace: dict = {}
        exec(assignment, namespace)  # noqa: S102 — intentional literal check
        assert namespace["_WATCHER_ENV"] is None

    def test_dict_watcher_env_roundtrips(self):
        """Host handoff supplies a dict: it still renders and evaluates."""
        env = {"HERMES_HOME": r"C:\Users\me\.hermes", "A": "1"}
        src = gateway_windows._build_transient_task_launcher(
            **_launcher_kwargs(watcher_env=env)
        )
        assignment = next(
            line for line in src.splitlines() if line.strip().startswith("_WATCHER_ENV =")
        )
        namespace: dict = {}
        exec(assignment, namespace)  # noqa: S102 — intentional literal check
        assert namespace["_WATCHER_ENV"] == env

    def test_none_launcher_executes_without_nameerror(self, monkeypatch, tmp_path):
        """Behavioral: the full None-path launcher executes past _WATCHER_ENV
        (no NameError), respawns once, and cleans its temp dir in finally."""
        import subprocess as _sp

        import hermes_constants

        temp_dir = tmp_path / "launcher-tmp"
        temp_dir.mkdir()
        (temp_dir / "dummy.txt").write_text("x", encoding="utf-8")
        src = gateway_windows._build_transient_task_launcher(
            task_name="Hermes_GW_Restart_1_abcd1234",
            temp_dir=str(temp_dir),
            old_pid=999999,  # provably dead: wait loop exits immediately
            gateway_cmd=[sys.executable, "-c", "pass"],
            respawn_cwd="",
            respawn_env_overlay={},
            watcher_env=None,
            project_root=str(Path(gateway_windows.__file__).resolve().parent.parent),
            watcher_timeout_s=5.0,
        )
        assert "_WATCHER_ENV = None" in src
        compile(src, "<transient_launcher_none>", "exec")

        popen_calls: list = []
        run_calls: list = []

        def fake_popen(*args, **kwargs):
            popen_calls.append((args, kwargs))

            class _P:
                pid = 1111

            return _P()

        def fake_run(*args, **kwargs):
            run_calls.append((args, kwargs))

            class _R:
                returncode = 0

            return _R()

        monkeypatch.setattr(_sp, "Popen", fake_popen)
        monkeypatch.setattr(_sp, "run", fake_run)
        monkeypatch.setattr(_subprocess_compat, "pid_exists_stdlib", lambda pid: False)
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
        try:
            exec(compile(src, "<transient_launcher_none>", "exec"), {"__name__": "__launcher__"})  # noqa: S102
        except NameError as exc:
            pytest.fail(f"None-path launcher raised NameError: {exc}")
        # Respawn attempted once; task delete attempted early + finally.
        assert len(popen_calls) == 1
        assert len(run_calls) >= 2
        assert any("/Delete" in str(call) for call in run_calls)
        # finally removed the temp dir.
        assert not temp_dir.exists()


# ---------------------------------------------------------------------------
# 7. Scheduler rejection log/warning distinction (#127089 review)
# ---------------------------------------------------------------------------

class TestSchedulerRejectionDiagnostics:
    def test_rejected_handoff_warns_best_effort(self, monkeypatch, tmp_path, caplog):
        """A rejected scheduler handoff falls back to a best-effort in-job
        watcher: the file log keeps the existing diagnostics plus the
        kill-on-close/breakaway warning, and the logger warns (not debug)."""
        import logging

        import hermes_constants

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "windowless_gateway_restart_spec",
            lambda argv: (argv, "", {}),
        )

        def failing_transient(*a, **k):
            raise RuntimeError("schtasks /Run failed (code 1): No valid logon token.")

        monkeypatch.setattr(gateway_windows, "_spawn_gateway_via_transient_task", failing_transient)
        calls: list = []
        monkeypatch.setattr(gateway.subprocess, "Popen", _fake_popen_factory(calls))
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="hermes_cli.gateway"):
            assert gateway._spawn_gateway_restart_watcher(555, ["python", "gateway"], host=False) is True
        assert len(calls) == 1
        log_path = tmp_path / "logs" / "gateway-stdio.log"
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8", errors="replace")
        # Existing diagnostics intact.
        assert "transient task handoff failed" in text
        assert "falling back" in text
        # Refined best-effort fallback warning.
        assert "best-effort" in text
        assert "breakaway" in text
        assert "kill-on-close" in text or "may not survive" in text
        warnings = [
            r for r in caplog.records
            if r.name == "hermes_cli.gateway" and r.levelno >= logging.WARNING
        ]
        assert warnings, "expected a logger.warning for the rejected handoff fallback"
        warned = " ".join(r.getMessage() for r in warnings)
        assert "transient task handoff failed" in warned
        assert "best-effort" in warned
        assert "breakaway" in warned

    def test_accepted_handoff_marks_submission_not_proven(self, monkeypatch, caplog):
        """An accepted schtasks /Run is a submission, not a proven surviving
        handoff: success is logged as accepted pending the liveness poll."""
        import logging

        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(_subprocess_compat, "process_is_in_job", lambda: True)
        monkeypatch.setattr(
            gateway_windows, "windowless_gateway_restart_spec",
            lambda argv: (argv, "", {}),
        )
        monkeypatch.setattr(
            gateway_windows, "_spawn_gateway_via_transient_task", lambda *a, **k: True,
        )
        calls: list = []
        monkeypatch.setattr(gateway.subprocess, "Popen", _fake_popen_factory(calls))
        with caplog.at_level(logging.DEBUG, logger="hermes_cli.gateway"):
            assert gateway._spawn_gateway_restart_watcher(666, ["python", "gateway"], host=False) is True
        assert calls == []
        messages = " ".join(r.getMessage() for r in caplog.records if r.name == "hermes_cli.gateway")
        assert "accepted" in messages
        assert "liveness poll" in messages or "to be verified" in messages


# ---------------------------------------------------------------------------
# 8. Launcher cleanup paths (#127089 review)
# ---------------------------------------------------------------------------

class TestLauncherCleanupContract:
    def test_launcher_preserves_early_and_final_cleanup(self):
        """Early _delete_task (no orphan on kill during wait), final
        _delete_task + temp cleanup, and the stdio/breakaway diagnostics
        all survive the nullable-literal fix."""
        src = gateway_windows._build_transient_task_launcher(**_launcher_kwargs())
        assert "gateway-stdio.log" in src
        assert "_HERMES_GATEWAY_BREAKAWAY" in src
        assert "shutil.rmtree" in src
        # Early delete before the wait loop.
        early = src.index("_delete_task()\ntry:")
        assert early >= 0
        # Final cleanup deletes the task and the temp dir.
        final = src.rindex("finally:")
        tail = src[final:]
        assert "_delete_task()" in tail
        assert "_cleanup_temp()" in tail
        assert "_TEMP_DIR" in src
