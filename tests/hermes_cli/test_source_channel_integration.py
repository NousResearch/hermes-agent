"""Source channel tracers through real HTTP, Git, and completion boundaries."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main, source_releases, update_cmd
from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_channel import channel_record, set_install_channel
from hermes_cli.config import require_readable_config_before_write


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                          text=True, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.fixture
def source(tmp_path, monkeypatch):
    home, origin, checkout = (tmp_path / name for name in ("home", "origin", "checkout"))
    home.mkdir()
    origin.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Channel Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    commits = []
    for label in ("installed", "published", "unpublished"):
        (origin / "content.txt").write_text(label)
        git(origin, "add", "content.txt")
        git(origin, "commit", "-m", label)
        commits.append(git(origin, "rev-parse", "HEAD"))
    git(tmp_path, "clone", str(origin), str(checkout))
    git(checkout, "checkout", "--detach", commits[0])
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: checkout)
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=main.cmd_update)
    opts = update_cmd._UpdateOptions(pre_update_version=None, gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False)
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    return SimpleNamespace(home=home, origin=origin, root=checkout, commits=commits, parser=parser)


def record(name, *, repository="NousResearch/hermes-agent", state="active", destination=None):
    return {"schema": 1, "name": name, "repository": repository, "policy": "preview",
            "state": state, "identity": {}, "nextSequence": 2, "head": None,
            **({"destination": destination} if destination else {})}


def reader_result(source, name, destination=None, repository="NousResearch/hermes-agent"):
    requested = record(name, repository=repository,
                       state="retired" if destination else "active", destination=destination)
    terminal = record(destination or name, repository=repository)
    terminal["head"] = {"buildId": "build-fixture", "sequence": 1}
    return SimpleNamespace(requested=requested, terminal=terminal, manifest={
        "schema": 1, "request": {"buildId": "build-fixture", "channel": terminal["name"],
        "sequence": 1, "repository": repository, "commit": source.commits[1],
        "sourceVersion": "1.2.3", "version": "0.0.1", "identity": {}, "bundleEnv": {},
        "publicBase": "https://hermes-assets.nousresearch.com"}, "packages": []})


def install_reader(monkeypatch, result):
    # This is the single documented adapter to ChannelReader.resolve's validated
    # result. No source caller should depend on tag tuples or protocol internals.
    monkeypatch.setattr(source_releases, "_resolve_channel", lambda name, repository: deepcopy(result), raising=False)


def saved(source):
    return channel_record(require_readable_config_before_write(source.home / "config.yaml"), source.root)


def test_tagless_channel_check_apply_is_pinned_not_branch_tip(source, monkeypatch, capsys):
    name = "preview-unknown-at-build"
    set_install_channel(name, source.root)
    install_reader(monkeypatch, reader_result(source, name))
    update_cmd._cmd_update_check()
    assert name in capsys.readouterr().out
    assert git(source.root, "rev-parse", "HEAD") == source.commits[0]
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda request: completed.append(deepcopy(request)))
    args = source.parser.parse_args(["update", "--channel", name, "--yes"])
    update_cmd._cmd_update_impl(args, False)
    assert git(source.root, "rev-parse", "HEAD") == source.commits[1]
    assert (source.root / "content.txt").read_text() == "published"
    assert git(source.root, "tag", "--list") == ""
    assert completed[0]["expected_sha"] == source.commits[1]
    assert saved(source)["channel"] == name


@pytest.mark.parametrize("outcome", ["success", "failed", "concurrent", "transient", "already-current"])
def test_retirement_adopts_destination_only_after_success(source, monkeypatch, outcome):
    name = "preview-retiring"
    set_install_channel(name, source.root)
    original = deepcopy(saved(source))
    install_reader(monkeypatch, reader_result(source, name, "stable"))
    if outcome == "already-current":
        git(source.root, "checkout", "--detach", source.commits[1])
    requests = []

    def completion(request):
        requests.append(deepcopy(request))
        assert saved(source) == original
        assert git(source.root, "rev-parse", "HEAD") == source.commits[1]
        if outcome == "concurrent":
            set_install_channel("my-new-choice", source.root)
        return {"exit_code": 1 if outcome == "failed" else 0, "receipt": None,
                "windows_resume": None}

    monkeypatch.setattr(update_cmd, "run_completion", completion)
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    flags = ["--channel", name] if outcome == "transient" else []
    args = source.parser.parse_args(["update", "--yes", *flags])
    if outcome == "failed":
        with pytest.raises(SystemExit):
            update_cmd._cmd_update_impl(args, False)
    else:
        update_cmd._cmd_update_impl(args, False)
    expected = {"success": "stable", "already-current": "stable", "failed": name,
                "transient": name, "concurrent": "my-new-choice"}[outcome]
    assert saved(source)["channel"] == expected
    assert len(requests) == 1


@pytest.mark.parametrize("channel", ["preview-not-registered", "stable", "canary", "main"])
def test_missing_channel_cannot_fall_back_to_main(source, monkeypatch, channel):
    from hermes_cli import source_check

    set_install_channel(channel, source.root)
    def missing(name, repository):
        raise ValueError("Channel does not exist")
    monkeypatch.setattr(source_releases, "_resolve_channel", missing)
    status = source_check.check_for_updates(install_root=source.root, home=source.home, force=True)
    assert status["error"] == "release-unavailable"
    assert "targetSha" not in status
    before = git(source.root, "rev-parse", "HEAD")
    with pytest.raises(SystemExit):
        update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--yes"]), False)
    assert git(source.root, "rev-parse", "HEAD") == before


def test_passive_check_reports_retirement_without_adopting_it(source, monkeypatch):
    from hermes_cli import source_check, banner

    name = "preview-retiring"
    set_install_channel(name, source.root)
    before = saved(source)
    install_reader(monkeypatch, reader_result(source, name, "stable"))
    status = source_check.check_for_updates(install_root=source.root, home=source.home, force=True)
    assert status["channel"] == name
    assert status["targetSha"] == source.commits[1]
    assert status["retirement"] == {"destination": "stable", "sourceOnly": True}
    assert status["updateAvailable"] is True
    assert saved(source) == before
    assert name in banner.format_banner_version_label()


def test_fork_channel_cannot_select_official_source(source, monkeypatch):
    url = "https://github.com/Fixture/fork.git"
    git(source.root, "config", "remote.origin.url", url)
    git(source.root, "config", f"url.{source.origin}.insteadOf", url)
    install_reader(monkeypatch, reader_result(source, "preview"))
    with pytest.raises(ValueError, match="repository"):
        source_releases.resolve_source_target("preview", ["git"], source.root)
    install_reader(monkeypatch, reader_result(source, "preview", repository="Fixture/fork"))
    target = source_releases.resolve_source_target("preview", ["git"], source.root)
    assert target.repository == "Fixture/fork"
    assert target.commit == source.commits[1]


@pytest.fixture
def channel_archive(source, monkeypatch):
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    archive = source.home / "archive"
    archive.mkdir()
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(archive)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(source_releases, "_PUBLIC_BASE", base)
    try:
        yield archive, base
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def publish_channel_build(channel_archive, name, build_id, commit, *, sequence=1, stable=False):
    from hashlib import sha256
    from hermes_cli.release_channels import canonical_json

    archive, base = channel_archive
    prefix = f"releases/channel-builds/{build_id}/"
    identity = {"token": "a" * 16, "displayName": "Preview fixture", "appNamePascal": "Fixture",
                "artifactNamePascal": "Fixture", "appId": "ai.fixture.preview",
                "msixAppIdWithOrg": "Fixture.Preview", "cliName": "fixture-preview",
                "windowsExecutableName": "Fixture"}
    version = f"1.2.{sequence}" if stable else f"0.0.{sequence}"
    request = {"schema": 1, "channel": name, "buildId": build_id, "sequence": sequence,
               "repository": "NousResearch/hermes-agent", "commit": commit,
               "sourceVersion": f"1.2.{sequence}", "version": version, "windowsVersion": version + ".0",
               "identity": identity, "bundleEnv": {}, "publicBase": base}
    if stable:
        request["releaseTag"] = "v" + version
    manifest = {"schema": 1, "receiverProtocol": 1, "request": request, "packages": [{
        "platform": "darwin", "arch": "arm64", "variant": "bundled",
        "artifact": {"key": prefix + "fixture.zip", "size": 1, "sha256": "d" * 64},
        "identity": identity["appId"], "version": version, "teamId": "ABCDEFGHIJ",
        "feed": {"key": prefix + "darwin/stable.yml", "channel": "stable"}}]}
    body = canonical_json(manifest)
    channel = {"schema": 1, "name": name, "repository": request["repository"],
               "policy": "stable-release" if stable else "preview", "state": "active", "identity": identity,
               "revision": sequence, "nextSequence": sequence + 1, "head": {"buildId": build_id,
               "sequence": sequence, "manifestKey": prefix + "build.json", "sha256": sha256(body).hexdigest()}}
    for key, content in [(prefix + "build.json", body),
                         (f"releases/channels/{name}.json", canonical_json(channel))]:
        path = archive / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return channel


def test_real_http_reader_resolves_tagless_build_through_cli(source, monkeypatch, channel_archive):
    archive, _ = channel_archive
    name = "http-preview-351"
    channel = publish_channel_build(channel_archive, name, "b" * 32, source.commits[1])
    from hermes_cli import source_check
    status = source_check.check_for_updates(install_root=source.root, home=source.home,
                                            channel=name, force=True)
    assert status.get("targetSha") == source.commits[1], status
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda req: completed.append(req))
    update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--channel", name]), False)
    assert git(source.root, "rev-parse", "HEAD") == source.commits[1]
    assert completed[0]["expected_sha"] == source.commits[1]
    # A mutated immutable manifest is refused, never substituted by main.
    path = archive / channel["head"]["manifestKey"]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        source_releases.resolve_source_target(name, ["git"], source.root)


@pytest.fixture
def retired_channel_archive(source, channel_archive):
    from hermes_cli.release_channels import canonical_json

    archive, _ = channel_archive
    name = "offline-preview"
    preview = publish_channel_build(channel_archive, name, "a" * 32, source.commits[0])
    qualified = publish_channel_build(channel_archive, "stable", "b" * 32, source.commits[1], stable=True)
    retired = dict(preview, state="retired", destination="stable", minimumVersion="1.0.0",
                   destinationHead=qualified["head"], receiverProtocol=1,
                   receiver={"kind": "discontinued"}, lastHead=preview["head"])
    (archive / f"releases/channels/{name}.json").write_bytes(canonical_json(retired))
    publish_channel_build(channel_archive, "stable", "c" * 32, source.commits[2], sequence=2, stable=True)
    set_install_channel(name, source.root)
    return SimpleNamespace(archive=archive, name=name, retired=retired,
                           qualified=qualified)


@pytest.mark.parametrize("fault", [None, "binding", "manifest"])
def test_offline_retirement_uses_qualified_build_before_current_stable(
        source, monkeypatch, retired_channel_archive, fault):
    from hermes_cli import source_check
    from hermes_cli.release_channels import canonical_json

    fixture = retired_channel_archive
    original = deepcopy(saved(source))
    completed = []
    def completion(request):
        completed.append(deepcopy(request))
        assert git(source.root, "rev-parse", "HEAD") == request["expected_sha"]
        return {"exit_code": 0, "receipt": None, "windows_resume": None}
    monkeypatch.setattr(update_cmd, "run_completion", completion)
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    if fault == "binding":
        fixture.retired["destinationHead"]["sequence"] += 1
        (fixture.archive / f"releases/channels/{fixture.name}.json").write_bytes(canonical_json(fixture.retired))
    elif fault == "manifest":
        path = fixture.archive / fixture.qualified["head"]["manifestKey"]
        path.write_bytes(path.read_bytes() + b" ")
    status = source_check.check_for_updates(install_root=source.root, home=source.home, force=True)
    args = source.parser.parse_args(["update", "--yes"])
    if fault:
        assert status["error"] == "release-unavailable"
        with pytest.raises(SystemExit):
            update_cmd._cmd_update_impl(args, False)
        assert git(source.root, "rev-parse", "HEAD") == source.commits[0]
        assert saved(source) == original and not completed
        return
    assert status.get("targetSha") == source.commits[1], status
    assert status["buildId"] == fixture.qualified["head"]["buildId"]
    assert saved(source) == original
    update_cmd._cmd_update_impl(args, False)
    assert git(source.root, "rev-parse", "HEAD") == source.commits[1]
    assert saved(source)["channel"] == "stable"
    status = source_check.check_for_updates(install_root=source.root, home=source.home, force=True)
    assert status["targetSha"] == source.commits[2], status
    update_cmd._cmd_update_impl(args, False)
    assert git(source.root, "rev-parse", "HEAD") == source.commits[2]
    assert [request["expected_sha"] for request in completed] == source.commits[1:]
    assert "channel_retirement" not in completed[1]


@pytest.mark.parametrize("transport", ["git", "shallow", "zip"])
def test_retirement_refuses_to_downgrade_newer_source(
        source, monkeypatch, retired_channel_archive, transport):
    git(source.root, "checkout", "--detach", source.commits[2])
    if transport == "shallow":
        import shutil
        shutil.rmtree(source.root)
        git(source.root.parent, "clone", "--depth=1", source.origin.as_uri(), str(source.root))
    if transport == "zip":
        (source.root / "pyproject.toml").write_text('[project]\nversion = "1.2.2"\n')
        monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (True, ["git"], False))
    original = deepcopy(saved(source))
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda request: completed.append(request))
    with pytest.raises(ValueError, match="newer|downgrade"):
        source_releases.resolve_source_target(retired_channel_archive.name,
                                              None if transport == "zip" else ["git"], source.root)
    with pytest.raises(SystemExit):
        update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--yes"]), False)
    assert git(source.root, "rev-parse", "HEAD") == source.commits[2]
    assert saved(source) == original and not completed


@pytest.mark.parametrize("dirty", [False, True])
def test_tagless_zip_apply_uses_pinned_source_archive(source, monkeypatch, dirty):
    import urllib.request
    from hermes_cli import update_cmd_zip

    name = "zip-preview"
    set_install_channel(name, source.root)
    install_reader(monkeypatch, reader_result(source, name, "stable"))
    archive = source.home / "source.zip"
    git(source.origin, "archive", "--format=zip", "--prefix=hermes-agent-fixture/",
        "--output=" + str(archive), source.commits[1])
    urls = []
    def download(url, filename):
        import shutil
        urls.append(url)
        shutil.copyfile(archive, filename)
    monkeypatch.setattr(urllib.request, "urlretrieve", download)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (True, ["git"], False))
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda req: completed.append(deepcopy(req)))
    if dirty:
        (source.root / "notes.txt").write_text("do not remove")
        with pytest.raises(SystemExit):
            update_cmd._cmd_update_impl(source.parser.parse_args(["update"]), False)
        assert not urls and not completed
        assert (source.root / "content.txt").read_text() == "installed"
        assert (source.root / "notes.txt").read_text() == "do not remove"
    else:
        update_cmd._cmd_update_impl(source.parser.parse_args(["update"]), False)
        assert urls == [f"https://github.com/NousResearch/hermes-agent/archive/{source.commits[1]}.zip"]
        assert (source.root / "content.txt").read_text() == "published"
        assert completed[0]["expected_sha"] == source.commits[1]
        assert completed[0]["channel_retirement"]["destination"] == "stable"
    assert saved(source)["channel"] == name


@pytest.mark.parametrize("outcome", ["success", "failure", "uncorrelated", "missing"])
def test_retirement_waits_for_correlated_completion_process(source, monkeypatch, outcome):
    from hermes_cli import update_receipt

    name = "process-retiring"
    set_install_channel(name, source.root)
    original = deepcopy(saved(source))
    # This child is the completion transport fixture, not a simulated PM install.
    # The existing completion-process suite exercises fresh PM/selected Python.
    script = source.root / "hermes_cli/update_completion.py"
    script.parent.mkdir()
    script.write_text(
        "import json, pathlib, sys\n"
        "request = json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "pathlib.Path(request['home'], 'child-request.json').write_text(json.dumps(request))\n"
        f"outcome = {outcome!r}\n"
        "code = 17 if outcome == 'failure' else 0\n"
        "receipt = dict(request['receipt'], finished_at='fixture', outcome='failed' if code else 'success')\n"
        "result = dict(schema=1, exit_code=code, receipt=receipt, windows_resume=None, update_id=receipt['update_id'])\n"
        "if outcome == 'uncorrelated': result['update_id'] = 'another-update'\n"
        "if outcome != 'missing': pathlib.Path(sys.argv[2]).write_text(json.dumps(result))\n"
        "raise SystemExit(code)\n"
    )
    update_receipt.begin_update_receipt()
    request = {"schema": 1, "source": str(source.root), "home": str(source.home),
               "expected_sha": source.commits[1], "windows_resume": None,
               "receipt": deepcopy(update_receipt._current.get().data),
               "channel_retirement": {"original": original, "destination": "stable"}}
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    if outcome == "success":
        update_cmd._complete_source_update(request)
    else:
        with pytest.raises(SystemExit):
            update_cmd._complete_source_update(request)
    assert saved(source)["channel"] == ("stable" if outcome == "success" else name)
    child_request = json.loads((source.home / "child-request.json").read_text())
    assert child_request["channel_retirement"]["original"] == original
    assert child_request["expected_sha"] == source.commits[1]


def test_source_branch_record_has_no_bundle_and_explicit_branch_is_separate(source, monkeypatch):
    from hermes_cli import source_check

    branch_record = record("main")
    branch_record.update(policy="source-branch", identity=None,
                         delivery={"kind": "source-branch", "branch": "main"})
    install_reader(monkeypatch, SimpleNamespace(requested=branch_record,
                    terminal=branch_record, manifest=None))
    target = source_releases.resolve_source_target("main", ["git"], source.root)
    assert target.branch == "main" and target.commit is None
    status = source_check.check_for_updates(install_root=source.root, home=source.home, force=True)
    assert status["targetSha"] == source.commits[2], status
    assert "buildId" not in status
    # Explicit --branch remains usable without a channel record or release host.
    def forbidden(*args):
        pytest.fail("explicit source branch contacted channel authority")
    monkeypatch.setattr(source_releases, "_resolve_channel", forbidden)
    set_install_channel("unavailable-preview", source.root)
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda request: completed.append(request))
    update_cmd._cmd_update_impl(source.parser.parse_args(["update", "--branch", "main"]), False)
    assert git(source.root, "rev-parse", "HEAD") == source.commits[2]
    assert "channel_retirement" not in completed[0]
    assert saved(source)["channel"] == "unavailable-preview"


def test_changed_checkout_cannot_repin_selected_channel_during_apply(source, monkeypatch):
    set_install_channel("preview", source.root)
    install_reader(monkeypatch, reader_result(source, "preview", "stable"))
    pull = update_cmd._pull_updates
    def moved(*args, **kwargs):
        before = pull(*args, **kwargs)
        git(source.root, "checkout", "--detach", source.commits[2])
        return before
    monkeypatch.setattr(update_cmd, "_pull_updates", moved)
    completed = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda req: completed.append(req))
    with pytest.raises(SystemExit):
        update_cmd._cmd_update_impl(source.parser.parse_args(["update"]), False)
    assert not completed
    assert saved(source)["channel"] == "preview"


@pytest.mark.parametrize("fault", ["floor", "protocol", "build", "chain", "missing"])
def test_source_retirement_rejects_invalid_archive_constraints(source, retired_channel_archive, fault):
    from hermes_cli.release_channels import canonical_json
    fixture = retired_channel_archive
    if fault == "floor":
        fixture.retired["minimumVersion"] = "9.0.0"
    elif fault == "protocol":
        fixture.retired["receiverProtocol"] = 2
    elif fault == "build":
        fixture.retired["destinationHead"]["buildId"] = "f" * 32
    elif fault == "chain":
        fixture.retired["destination"] = fixture.name
    else:
        del fixture.retired["destinationHead"]
    (fixture.archive / f"releases/channels/{fixture.name}.json").write_bytes(canonical_json(fixture.retired))
    with pytest.raises(ValueError):
        source_releases.resolve_source_target(fixture.name, ["git"], source.root)
