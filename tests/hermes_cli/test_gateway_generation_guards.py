"""Persisted supervisor commands bind to the install root, never a dependency
generation workspace (regression for #131164).

A generation's venv console script runs with PROJECT_ROOT inside
``installs/<key>/environments/<gen>/workspace`` — a GC-able tree whose own
install key has no committed environment, so a definition persisted from
there crash-loops the supervised service with "no dependency environment is
committed". Generation must map back to the owning checkout, and a
definition the mapping cannot vouch for must never be written
(``hermes_cli/gateway_generation_guards.py``).
"""
import hermes_cli.gateway as gateway_cli


class TestGenerationRootedServiceBinding:
    """The two end-to-end contracts: canonicalize the root, refuse the unsafe write."""

    def _workspace_rooted_install(self, tmp_path, monkeypatch):
        from pm.environments import install_state_dir

        home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(home))
        checkout = tmp_path / "checkout"
        (checkout / "pm").mkdir(parents=True)
        (checkout / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        state = install_state_dir(checkout)
        workspace = state / "environments" / ("g" * 32) / "workspace"
        (workspace / "pm").mkdir(parents=True)
        (workspace / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
        (state / "inputs").mkdir(parents=True)
        (state / "inputs" / ".project-root").write_text(str(checkout), encoding="utf-8")
        return checkout, workspace

    def test_generation_rooted_invocation_writes_checkout_bound_definitions(
        self, tmp_path, monkeypatch
    ):
        checkout, workspace = self._workspace_rooted_install(tmp_path, monkeypatch)
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", workspace)
        monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/usr/bin/python3")

        unit = gateway_cli.generate_systemd_unit(system=False)
        exec_lines = [
            line
            for line in unit.splitlines()
            if line.strip().startswith(("ExecStart", "ExecStop"))
        ]
        assert exec_lines, "the unit must carry launch commands"
        assert all(
            str(checkout.resolve()) in line for line in exec_lines
        ), "Exec directives must launch from the owning checkout"
        assert str(workspace) not in unit

        plist = gateway_cli.generate_launchd_plist()
        assert str(workspace) not in plist
        assert str(checkout.resolve()) in plist

    def test_refusal_leaves_an_unsafe_definition_untouched(
        self, tmp_path, monkeypatch, capsys
    ):
        checkout, workspace = self._workspace_rooted_install(tmp_path, monkeypatch)
        monkeypatch.setattr(gateway_cli, "PROJECT_ROOT", workspace)
        monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/usr/bin/python3")
        # A synthetic hermes home: the tmp_path fixture lives under /tmp/pytest-of-, which the
        # pre-existing temp-home guard refuses before the generation guard can be exercised.
        poisoned_exec = (
            "/srv/hermes-home/installs/aabbcc/environments/" + ("g" * 32) + "/workspace/.hermes/bin/hermes"
        )
        ran = []
        monkeypatch.setattr(gateway_cli, "_run_systemctl", lambda args, **kwargs: ran.append(args))

        unit_path = tmp_path / "hermes-gateway.service"
        unit_path.write_text("old unit\n", encoding="utf-8")
        monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unit_path)
        monkeypatch.setattr(
            gateway_cli,
            "generate_systemd_unit",
            lambda system=False, run_as_user=None: f'[Service]\nExecStart="{poisoned_exec}" "gateway" "run"\n',
        )
        monkeypatch.setattr(gateway_cli, "_sync_hermes_home_from_systemd_unit", lambda **_: None)

        assert gateway_cli.refresh_systemd_unit_if_needed(system=False) is False
        assert unit_path.read_text(encoding="utf-8") == "old unit\n"
        assert not any("daemon-reload" in str(args) for args in ran)

        # The same contract on the launchd side; the PATH value carries a generation bin dir
        # that must not alone trip the scan — only the launch command does.
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old content</plist>\n", encoding="utf-8")
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
        monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: False)
        monkeypatch.setattr(
            gateway_cli,
            "generate_launchd_plist",
            lambda: (
                "<plist><key>ProgramArguments</key><array>"
                f"<string>{poisoned_exec}</string>"
                "</array>"
                "<key>EnvironmentVariables</key><dict><key>PATH</key><string>"
                "/usr/bin:/srv/hermes-home/installs/aabbcc/environments/" + ("g" * 32) + "/venv/bin"
                "</string></dict></plist>"
            ),
        )

        assert gateway_cli.refresh_launchd_plist_if_needed() is False
        assert plist_path.read_text(encoding="utf-8") == "<plist>old content</plist>\n"

        out = capsys.readouterr().out
        assert "dependency-generation tree" in out
        assert str(checkout / ".hermes" / "bin" / "hermes") in out
