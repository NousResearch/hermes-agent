"""Channel dispatch names an exact pushed Git commit; CI allocates and builds."""
import subprocess

import pytest


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


@pytest.fixture
def source(tmp_path):
    origin = tmp_path / "origin.git"
    repo = tmp_path / "source"
    origin.mkdir()
    repo.mkdir()
    git(origin, "init", "--bare", "--quiet")
    git(repo, "init", "--initial-branch=main", "--quiet")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.test")
    (repo / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1.2.3"\n', encoding="utf-8")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "--quiet", "-m", "Initial")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "--quiet", "-u", "origin", "main")
    return repo


def test_dispatch_names_channel_and_binds_to_pushed_source(source):
    from scripts.releases.channel_build import prepare_build
    calls = []
    options = dict(name="custom-branch", revision="main", remote="origin", repo=source,
                   repository="example/hermes-agent", default_branch="main",
                   dispatch=calls.append, bundle_env={"HERMES_GUEST_ONBOARDING": "1"})
    dry = prepare_build(**options)
    assert dry["commit"] == git(source, "rev-parse", "HEAD")
    assert dry["sourceVersion"] == "1.2.3"
    assert not calls
    command = dry["command"]
    assert "desktop-bundled-release.yml" in command
    assert "channel=custom-branch" in command
    assert "build_commit=" + dry["commit"] in command
    assert any(value.startswith("bundle_env=") for value in command)
    assert "--ref" in command and command[command.index("--ref") + 1] == "main"
    admitted = prepare_build(**options, publish=True)
    assert calls == [command]
    assert admitted["commit"] == dry["commit"]
    # An unpublished commit is never dispatched.
    git(source, "commit", "--allow-empty", "--quiet", "-m", "Unpublished")
    with pytest.raises(ValueError, match="pushed"):
        prepare_build(**options, publish=True)
    assert len(calls) == 1


def test_missing_default_branch_is_rejected_without_dispatch(source):
    from scripts.releases.channel_build import prepare_build
    from hermes_cli.release_channels import ChannelError
    with pytest.raises(ChannelError, match="default branch"):
        prepare_build(name="invalid-controller", revision="main", remote="origin", repo=source,
                      repository="example/hermes-agent", default_branch="",
                      dispatch=lambda command: pytest.fail("must not dispatch"), publish=True)


def test_disposable_scope_is_opt_in_and_lease_bound(monkeypatch):
    from scripts.releases import channel_build, r2
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("R2_DISPOSABLE_RUN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", "https://archive.example")
    # No lease and no repository identity check: scoping is opt-in, the
    # unscoped publisher talks production keys only when the caller holds them.
    monkeypatch.setattr(r2, "credentials", lambda: ({"access_key_id": "inert", "secret_key": "inert"}, "https://r2.example", "bucket"))
    unscoped = channel_build.configured_publisher("ethernet8023/hermes-agent")
    assert unscoped.store.scope.key("releases/channels/preview.json") == "releases/channels/preview.json"
    monkeypatch.setenv("R2_DISPOSABLE_RUN", "123")
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "456")
    monkeypatch.setattr(channel_build.commit_build, "output", lambda args: "789")
    with pytest.raises(ValueError, match="another repository"):
        channel_build.configured_publisher("ethernet8023/hermes-agent")
    monkeypatch.setattr(channel_build.commit_build, "output", lambda args: "456")
    publisher = channel_build.configured_publisher("ethernet8023/hermes-agent")
    assert publisher.store.scope.key("releases/channels/preview.json") == "ci-disposable/456/123/releases/channels/preview.json"
    assert publisher.reader.base_url == "https://archive.example/ci-disposable/456/123"


def test_release_parser_preserves_plain_oneoff_dispatch(monkeypatch):
    import sys
    from scripts import release
    from scripts.releases import commit_build
    calls = []
    monkeypatch.setattr(commit_build, "cmd_build_commit", lambda args: calls.append(("oneoff", args.build_commit)))
    monkeypatch.setattr(sys, "argv", ["release.py", "--build-commit", "main"])
    release.main()
    assert calls == [("oneoff", "main")]
    monkeypatch.setattr(sys, "argv", ["release.py", "--channel", "bad/name", "--build-commit", "main"])
    with pytest.raises(SystemExit) as rejected:
        release.main()
    assert rejected.value.code == 2
    from scripts.releases import channel_build
    monkeypatch.setattr(channel_build, "cmd_channel", lambda args: calls.append(("channel", args.channel)))
    monkeypatch.setattr(sys, "argv", ["release.py", "--channel", "unknown-name", "--build-commit", "main"])
    release.main()
    assert calls[-1] == ("channel", "unknown-name")
    monkeypatch.setattr(sys, "argv", ["release.py", "--channels", "--bundle-env", "X=y"])
    with pytest.raises(SystemExit) as incompatible:
        release.main()
    assert incompatible.value.code == 2