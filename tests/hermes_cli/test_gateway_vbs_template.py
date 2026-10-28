"""Tests for the shared Windows gateway VBS template.

The VBS launcher is generated in one place (``hermes_cli.gateway_vbs_template``)
and VERIFIED in another (``hermes_cli.gateway_runtime_service_identity``).
Before this module existed the two hand-maintained copies drifted: the
generator emitted a new supervised template while the verifier still
exact-matched the old one-shot template, so the live deployed launcher was
rejected with ``service_identity_unverified`` and the Windows native-task
service could no longer be discovered or verified.

These tests pin the contract that prevents the drift from recurring.
"""

import pytest

from hermes_cli import gateway_runtime_service_identity as identity
from hermes_cli import gateway_vbs_template as vbs_template
from hermes_cli import gateway_windows


PYTHON = r"C:\Users\paulf\AppData\Local\hermes\tools\python-3.14.9+202****1021-win32-x64\python.exe"
HOME = r"C:\Users\paulf\AppData\Local\hermes"

# The pre-change deployed launcher: a one-shot detached launch. Installations
# that have not yet run ``hermes update`` still carry this file, so the
# verifier must keep recognising it.
LEGACY_VBS = "\n".join(
    [
        "' Hermes Agent Gateway - Messaging Platform Integration",
        "Option Explicit",
        "Dim sh, env, existing_pp",
        'Set sh = CreateObject("WScript.Shell")',
        'Set env = sh.Environment("PROCESS")',
        f'env.Item("HERMES_HOME") = "{HOME}"',
        'env.Item("PYTHONIOENCODING") = "utf-8"',
        'env.Item("HERMES_GATEWAY_DETACHED") = "1"',
        'env.Item("HERMES_SUPERVISED_CHILD") = "1"',
        f'env.Item("VIRTUAL_ENV") = "{HOME}\\tools"',
        'existing_pp = env.Item("PYTHONPATH")',
        "If Len(existing_pp) > 0 Then",
        f'  env.Item("PYTHONPATH") = "{HOME}\\hermes-agent;" & existing_pp',
        "Else",
        f'  env.Item("PYTHONPATH") = "{HOME}\\hermes-agent"',
        "End If",
        f'sh.CurrentDirectory = "{HOME}"',
        f'sh.Run "{PYTHON} -m hermes_cli.main gateway run", 0, False',
        "",
    ]
)


def _render(shape=vbs_template.SUPERVISED, **overrides) -> str:
    """Render a launcher with controlled values so mutations are predictable."""
    values = {
        "pythonpath_with_existing": f"{HOME}\\hermes-agent;",
        "pythonpath": f"{HOME}\\hermes-agent",
        "cwd": HOME,
        "command": f"{PYTHON} -m hermes_cli.main gateway run",
        "delay_ms": vbs_template.BASE_DELAY_MS,
        "retry_code": vbs_template.RETRY_EXIT_CODE,
        "max_attempts": vbs_template.MAX_ATTEMPTS,
    }
    values.update(overrides)
    return vbs_template.render(
        shape,
        description="Hermes Agent Gateway",
        env_items=[
            ("HERMES_HOME", HOME),
            ("HERMES_GATEWAY_DETACHED", "1"),
            ("HERMES_SUPERVISED_CHILD", "1"),
            ("PYTHONIOENCODING", "utf-8"),
            ("VIRTUAL_ENV", f"{HOME}\\.venv"),
        ],
        values=values,
    )


def _generated() -> str:
    """The script the real generator emits."""
    return gateway_windows._build_gateway_vbs_script(PYTHON, HOME, HOME, "")


class TestTemplateRoundTrip:
    """The generator and the verifier must agree by construction."""

    def test_generated_script_parses_as_supervised(self):
        parsed = vbs_template.parse(_generated())
        assert parsed is not None
        assert parsed.shape == vbs_template.SUPERVISED.name
        assert parsed.home == HOME
        assert parsed.env["HERMES_SUPERVISED_CHILD"] == "1"

    def test_generated_script_is_accepted_by_the_recognizer(self):
        """The drift that broke production: generator output the verifier rejected."""
        home, argv = identity._vbs_identity(_generated())
        assert home == HOME
        assert argv[-3:] == ["hermes_cli.main", "gateway", "run"]

    def test_legacy_one_shot_script_is_still_recognized(self):
        """Installations that have not yet run ``hermes update`` must keep working."""
        parsed = vbs_template.parse(LEGACY_VBS)
        assert parsed is not None
        assert parsed.shape == vbs_template.LEGACY.name
        home, argv = identity._vbs_identity(LEGACY_VBS)
        assert home == HOME
        assert argv[-3:] == ["hermes_cli.main", "gateway", "run"]

    def test_the_launcher_verifies_in_both_encodings(self):
        """A launcher must verify whether it was written UTF-16 or UTF-8.

        The current writer emits ``.vbs`` launchers as UTF-16 with a BOM
        (Windows Script Host decodes a BOM-less script with the ANSI code
        page), while launchers written by older builds -- and every install
        that has not run ``hermes update`` -- are UTF-8. The verifier reads
        whichever it finds, so both must verify.
        """
        script = _render()
        for encoding in ("utf-16", "utf-8"):
            text = identity.launcher_text(script.encode(encoding))
            home, argv = identity._vbs_identity(text)
            assert home == HOME, f"the {encoding} launcher did not verify"
            assert argv[-3:] == ["hermes_cli.main", "gateway", "run"]

    def test_a_utf16_launcher_is_written_the_way_the_verifier_reads_it(self):
        """The writer and the reader must agree on the encoding.

        A launcher written UTF-16 but read as UTF-8 (or the reverse) decodes
        to mojibake, and the verifier then rejects a launcher this very
        install just wrote.
        """
        import inspect

        from hermes_cli import gateway_windows

        source = inspect.getsource(gateway_windows._write_task_script)
        assert 'encoding="utf-16"' in source, "the .vbs launcher must be written as UTF-16"

    def test_render_then_parse_preserves_every_value(self):
        parsed = vbs_template.parse(_render())
        assert parsed is not None
        assert parsed.values["max_attempts"] == 3
        assert parsed.values["retry_code"] == 75
        assert parsed.values["delay_ms"] == 3000

    def test_every_shape_round_trips(self):
        """Each declared shape must render and parse back to itself."""
        for shape in vbs_template.SHAPES:
            parsed = vbs_template.parse(_render(shape))
            assert parsed is not None, f"{shape.name} did not parse back"
            assert parsed.shape == shape.name

    def test_retry_code_matches_the_gateway_exit_code(self):
        """The launcher must act on the code the gateway actually returns."""
        from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

        assert vbs_template.RETRY_EXIT_CODE == GATEWAY_SERVICE_RESTART_EXIT_CODE


class TestForeignScriptsRejected:
    """Recognising the vendor template must not mean accepting anything."""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda s: "MsgBox \"hello\"", id="not-a-launcher"),
            pytest.param(lambda s: s.replace("attempts = 0", "attempts = 0\nwscript.Echo \"x\""), id="injected-line"),
            pytest.param(
                lambda s: s.replace("If attempts >= 3 Then Exit Do", "If attempts >= 3 Then Exit Do\n  WScript.Run \"evil.exe\""),
                id="injected-command",
            ),
            pytest.param(
                lambda s: s.replace("Dim sh, env, existing_pp, rc, attempts, delay", "Dim sh, env, existing_pp"),
                id="wrong-declarations",
            ),
            pytest.param(
                lambda s: s.replace('env.Item("HERMES_SUPERVISED_CHILD") = "1"', 'env.Item("EVIL") = "1"'),
                id="missing-marker",
            ),
            pytest.param(lambda s: s.replace('sh.CurrentDirectory', 'sh.Run "calc.exe"'), id="foreign-command"),
            pytest.param(lambda s: s + '\nwscript.Run "evil.exe"\n', id="appended-command"),
            pytest.param(
                lambda s: s.replace('CreateObject("WScript.Shell")', 'CreateObject("evil.Shell")'),
                id="foreign-shell-object",
            ),
            pytest.param(lambda s: s.replace("Option Explicit", "Option Explicit\nDim extra"), id="injected-declaration"),
            pytest.param(lambda s: s.replace("rc = sh.Run(", "rc = sh.Run2("), id="foreign-run-call"),
        ],
    )
    def test_foreign_script_is_rejected(self, mutate):
        script = mutate(_render())
        with pytest.raises(ValueError, match="service_identity_unverified"):
            identity._vbs_identity(script)

    def test_unresolved_environment_variable_is_rejected(self):
        """``%VAR%`` means the launcher would resolve a different command at run time."""
        script = _render(command="%HERMES_PYTHON% -m hermes_cli.main gateway run")
        with pytest.raises(ValueError, match="service_identity_unverified"):
            identity._vbs_identity(script)

    def test_unknown_environment_key_is_rejected(self):
        script = _render().replace(
            'env.Item("HERMES_SUPERVISED_CHILD") = "1"',
            'env.Item("HERMES_SUPERVISED_CHILD") = "1"\nenv.Item("EVIL") = "1"',
        )
        with pytest.raises(ValueError, match="service_identity_unverified"):
            identity._vbs_identity(script)

    def test_supervised_child_marker_is_required(self):
        script = _render().replace(
            'env.Item("HERMES_SUPERVISED_CHILD") = "1"', 'env.Item("HERMES_HOME") = "x"'
        )
        with pytest.raises(ValueError, match="service_identity_unverified"):
            identity._vbs_identity(script)


class TestSupervisedLoopIsBounded:
    """AC1: a gateway that keeps exiting 75 must not spin forever."""

    def test_supervised_loop_has_a_finite_attempt_cap(self):
        script = _render()
        assert "attempts >= 3 Then Exit Do" in script
        assert vbs_template.parse(script).values["max_attempts"] == 3

    def test_supervised_loop_waits_for_the_gateway(self):
        """The launch must be synchronous.

        A detached launch returns immediately, so the scheduled task completes
        in under a second and its ``IgnoreNew`` policy no longer covers the
        gateway's lifetime -- a second launch can then race a draining
        predecessor, which is the hard-kill pattern from 2026-10-26.
        """
        script = _render()
        assert "rc = sh.Run(" in script
        assert ", 0, True)" in script
        assert ", 0, False)" not in script

    def test_supervised_loop_only_retries_the_retry_code(self):
        script = _render()
        assert "If rc <> 75 Then Exit Do" in script

    def test_supervised_loop_backs_off(self):
        """The retry delay must grow, not stay at a flat 3 seconds."""
        script = _render()
        assert "delay = delay * 2" in script
        assert "WScript.Sleep delay" in script

    def test_legacy_shape_stays_detached(self):
        """The legacy one-shot shape must keep its original semantics."""
        assert ", 0, False" in LEGACY_VBS
        assert vbs_template.parse(LEGACY_VBS).shape == vbs_template.LEGACY.name

    def test_absurd_attempt_count_is_rejected(self):
        """A launcher from a build with different tuning must still verify, but
        not one with an absurd attempt count."""
        assert vbs_template.parse(_render(max_attempts=9999)) is None


class TestTemplateParsing:
    def test_unparseable_script_returns_none(self):
        assert vbs_template.parse("") is None
        assert vbs_template.parse("not a vbs file") is None

    def test_parse_is_shape_specific(self):
        assert vbs_template.parse(LEGACY_VBS).shape == vbs_template.LEGACY.name
        assert vbs_template.parse(_render()).shape == vbs_template.SUPERVISED.name

    def test_render_refuses_an_unknown_environment_key(self):
        with pytest.raises(ValueError, match="allowlist"):
            vbs_template.render(
                vbs_template.SUPERVISED,
                description="x",
                env_items=[("EVIL", "1")],
                values={},
            )
