"""Channel dispatch freezes a real pushed Git commit before allocating inputs."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

_spec = importlib.util.spec_from_file_location("channel_http_fixture", Path(__file__).with_name("test_release_channels.py"))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


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


def test_dry_run_never_allocates_and_dispatch_is_bound_to_pushed_source(source):
    from scripts.releases.channel_build import prepare_build
    from hermes_cli.release_channels import canonical_json
    with _fixture.object_server() as (url, objects, headers, requests, faults):
        pub = _fixture.publisher(url)
        calls = []
        options = dict(name="custom-branch", revision="main", remote="origin", repo=source,
                       repository="example/hermes-agent", default_branch="main", publisher=pub,
                       dispatch=calls.append, bundle_env={"FEATURE": "a"}, controller_commit="c" * 40)
        dry = prepare_build(**options)
        assert dry["commit"] == git(source, "rev-parse", "HEAD")
        assert not calls and not objects
        admitted = prepare_build(**options, publish=True)
        request = admitted["request"]
        assert request["commit"] == dry["commit"] and request["sourceVersion"] == "1.2.3"
        command = calls.pop()
        assert "desktop-bundled-release.yml" in command
        assert "channel_build=" + request["buildId"] in command
        assert "channel_request_sha256=" + hashlib.sha256(canonical_json(request)).hexdigest() in command
        assert "--ref" in command and command[command.index("--ref") + 1] == "main"
        assert not any(value.startswith("build_commit=") for value in command)
        git(source, "commit", "--allow-empty", "--quiet", "-m", "Unpublished")
        with pytest.raises(ValueError, match="pushed"):
            prepare_build(**options, publish=True)
        assert not calls
        assert pub.request(request["buildId"])["commit"] == dry["commit"]


def test_missing_controller_branch_is_rejected_without_channel_creation(source):
    from scripts.releases.channel_build import prepare_build
    from hermes_cli.release_channels import ChannelError
    with _fixture.object_server() as (url, objects, headers, requests, faults):
        pub = _fixture.publisher(url)
        with pytest.raises(ChannelError, match="default branch"):
            prepare_build(name="invalid-controller", revision="main", remote="origin", repo=source,
                          repository="example/hermes-agent", default_branch="absent", publisher=pub,
                          dispatch=lambda command: pytest.fail("must not dispatch"), publish=True)
        assert not objects


def test_local_fork_requires_scope_before_credentials_or_storage(monkeypatch):
    from scripts.releases import channel_build, r2
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("R2_DISPOSABLE_RUN", raising=False)
    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", "https://archive.example")
    monkeypatch.setattr(r2, "credentials", lambda: pytest.fail("Unscoped fork reached credentials"))
    with pytest.raises(ValueError, match="disposable"):
        channel_build.configured_publisher("ethernet8023/hermes-agent")
    monkeypatch.setenv("R2_DISPOSABLE_RUN", "123-1")
    monkeypatch.setenv("GITHUB_REPOSITORY_ID", "456")
    monkeypatch.setattr(channel_build.commit_build, "output", lambda args: "789")
    with pytest.raises(ValueError, match="another repository"):
        channel_build.configured_publisher("ethernet8023/hermes-agent")
    monkeypatch.setattr(channel_build.commit_build, "output", lambda args: "456")
    monkeypatch.setattr(r2, "credentials", lambda: ({"access_key_id": "inert", "secret_key": "inert"}, "https://r2.example", "bucket"))
    publisher = channel_build.configured_publisher("ethernet8023/hermes-agent")
    assert publisher.store.scope.key("releases/channels/preview.json") == "ci-disposable/456/123-1/releases/channels/preview.json"
    assert publisher.reader.base_url == "https://archive.example/ci-disposable/456/123-1"


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
    monkeypatch.setattr(channel_build, "cmd_channel", lambda args: calls.append(("resume", args.resume_channel_build)))
    monkeypatch.setattr(sys, "argv", ["release.py", "--resume-channel-build", "a" * 32, "--request-sha256", "b" * 64])
    release.main()
    assert calls[-1] == ("resume", "a" * 32)


def test_dispatch_retry_uses_same_immutable_request_and_no_new_sequence(source):
    from scripts.releases import channel_build
    from hermes_cli.release_channels import canonical_json, ChannelError
    with _fixture.object_server() as (url, objects, headers, requests, faults):
        pub = _fixture.publisher(url)
        calls = []
        admitted = channel_build.prepare_build(name="resume", revision="main", remote="origin", repo=source,
                    repository="example/hermes-agent", default_branch="main", publisher=pub,
                    dispatch=calls.append, publish=True)
        request = admitted["request"]
        before = pub._read("resume")[0]
        digest = hashlib.sha256(canonical_json(request)).hexdigest()
        channel_build.resume_build(request["buildId"], digest, publisher=pub, default_branch="main", dispatch=calls.append)
        assert calls[-1] == calls[0]
        assert pub._read("resume")[0] == before
        with pytest.raises(ChannelError, match="SHA256"):
            channel_build.resume_build(request["buildId"], "0" * 64, publisher=pub, default_branch="main", dispatch=calls.append)
