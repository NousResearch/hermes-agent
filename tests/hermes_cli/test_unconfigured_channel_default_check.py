"""#124584: hermes update --check must say so when it is riding the
unconfigured main-line default, not just print the channel name."""
import subprocess


def _git_checkout(tmp_path, extra_branches=()):
    remote, local = tmp_path / "remote", tmp_path / "checkout"

    def git(*args, cwd=None):
        return subprocess.run(["git", *map(str, args)], cwd=cwd, check=True, capture_output=True, text=True)

    git("init", "-b", "main", remote)
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--allow-empty", "-m", "initial", cwd=remote)
    for branch in extra_branches:
        git("branch", branch, cwd=remote)
    git("clone", remote, local)
    return local


def test_unconfigured_install_warns_it_is_on_the_main_line_default(tmp_path, monkeypatch, capsys):
    from hermes_cli import main
    from hermes_cli.update_cmd import _cmd_update_check

    local = _git_checkout(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", local)
    _cmd_update_check()
    output = capsys.readouterr().out
    assert "Update channel: main" in output
    assert "--set-channel stable" in output


def test_explicit_channel_flag_this_run_needs_no_warning(tmp_path, monkeypatch, capsys):
    from hermes_cli import main
    from hermes_cli.update_cmd import _cmd_update_check

    local = _git_checkout(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", local)
    _cmd_update_check(channel="main")
    output = capsys.readouterr().out
    assert "Update channel: main" in output
    assert "--set-channel stable" not in output


def test_configured_stable_record_needs_no_warning(tmp_path, monkeypatch, capsys):
    from hermes_cli import main
    from hermes_cli.update_cmd import _cmd_update_check
    from hermes_cli.update_channel import set_install_channel

    local = _git_checkout(tmp_path, extra_branches=("stable",))
    monkeypatch.setattr(main, "PROJECT_ROOT", local)
    set_install_channel("stable", local)
    _cmd_update_check()
    output = capsys.readouterr().out
    assert "Update channel: stable" in output
    assert "--set-channel stable" not in output
