"""Release-channel checks and updates against actual repositories and HTTP feeds."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from types import SimpleNamespace
import subprocess
import urllib.request
from urllib.parse import urlsplit

import pytest

from hermes_cli import main, update_cmd
from hermes_cli.update_channel import set_install_channel


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def releases(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Release Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    git(origin, "config", "tag.gpgsign", "false")
    commits = []
    for label in ("old", "stable", "canary", "unpublished"):
        (origin / "content.txt").write_text(label, encoding="utf-8")
        git(origin, "add", ".")
        git(origin, "commit", "-m", label)
        commits.append(git(origin, "rev-parse", "HEAD"))
    tags = {"stable": "v1.2.3", "canary": "v1.2.4-canary.20260911125822"}
    git(origin, "tag", "-a", tags["stable"], commits[1], "-m", "stable")
    git(origin, "tag", "-a", tags["canary"], commits[2], "-m", "canary")
    git(origin, "tag", "v99.0.0", commits[3])
    git(origin, "tag", "v99.0.1-canary.20260912125822", commits[3])
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", str(origin), str(checkout))
    git(checkout, "config", "user.name", "Release Fixture")
    git(checkout, "config", "user.email", "fixture@example.invalid")
    git(checkout, "config", "commit.gpgsign", "false")
    git(checkout, "checkout", "--detach", commits[0])
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    responses = {}
    requests = []
    for channel, tag in tags.items():
        responses[f"/releases/{channel}/index.html"] = (
            f'<meta name="hermes-build" content="{tag}">'
        )
        responses[f"/repos/NousResearch/hermes-agent/releases/tags/{tag}"] = {
            "tag_name": tag, "draft": False, "prerelease": channel == "canary",
        }
        responses[f"/repos/NousResearch/hermes-agent/commits/{tag}"] = {
            "sha": commits[1 if channel == "stable" else 2],
        }
    responses["/releases/stable/release-candidates.json"] = {
        "tag": tags["stable"], "commit": commits[1],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            data = responses.get(self.path)
            self.send_response(404 if data is None else 200)
            self.end_headers()
            if data is not None:
                self.wfile.write((data if isinstance(data, str) else json.dumps(data)).encode())

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    open_url = urllib.request.urlopen

    def local_urlopen(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        parsed = urlsplit(url)
        return open_url(f"http://127.0.0.1:{server.server_port}{parsed.path}"
                        + (f"?{parsed.query}" if parsed.query else ""), *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", local_urlopen)
    yield SimpleNamespace(root=checkout, origin=origin, commits=commits,
                          tags=tags, responses=responses, requests=requests)
    server.shutdown()
    server.server_close()
    thread.join()


def test_stable_resolution_uses_promoted_pointer_not_highest_tag(releases):
    assert update_cmd._resolve_latest_release_tag(["git"], releases.root) == (
        releases.tags["stable"], releases.commits[1],
    )
    assert releases.requests


@pytest.mark.parametrize("channel", ["stable", "canary"])
@pytest.mark.parametrize("start", ["old", "ahead", "local"])
def test_source_check_and_apply_land_on_selected_release(releases, monkeypatch, capsys, channel, start):
    if start != "old":
        git(releases.root, "checkout", "-b", "my-work", releases.commits[3])
    if start == "local":
        (releases.root / "my-work.txt").write_text("committed local work\n")
        git(releases.root, "add", ".")
        git(releases.root, "commit", "-m", "local work")
        (releases.root / "notes.txt").write_text("uncommitted notes\n")
    branch_sha = git(releases.root, "rev-parse", "HEAD")
    set_install_channel(channel, releases.root)
    before = git(releases.root, "rev-parse", "HEAD")
    update_cmd._cmd_update_check()
    assert releases.tags[channel] in capsys.readouterr().out
    assert git(releases.root, "rev-parse", "HEAD") == before

    # Exercise the real selection/fetch/checkout path, not dependency installation
    # or live service management. No host OS is simulated.
    opts = update_cmd._UpdateOptions(
        active_lazy_features=[], pre_update_version=None, gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False,
    )
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: [])
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda: (False, ["git"], False))
    applied = []
    monkeypatch.setattr(update_cmd, "_apply_pulled_update", lambda *a, **k: applied.append(git(releases.root, "rev-parse", "HEAD")))
    args = SimpleNamespace(branch=None, channel=None, force_venv=True)
    update_cmd._cmd_update_impl(args, False)
    expected = releases.commits[1 if channel == "stable" else 2]
    assert applied == [expected]
    assert git(releases.root, "rev-parse", "HEAD") == expected
    if start != "old":
        assert git(releases.root, "rev-parse", "my-work") == branch_sha
    if start == "local":
        assert (releases.root / "notes.txt").read_text() == "uncommitted notes\n"
    update_cmd._cmd_update_check()
    assert "Up to date with" in capsys.readouterr().out


def test_source_check_honors_transient_channel_without_rewriting_record(releases, capsys):
    set_install_channel("stable", releases.root)
    args = SimpleNamespace(branch=None, channel="canary", check=True)
    main.cmd_update(args)
    assert releases.tags["canary"] in capsys.readouterr().out
    update_cmd._cmd_update_check()
    assert releases.tags["stable"] in capsys.readouterr().out


def test_fork_origin_uses_its_own_published_release_not_the_official_pointer(releases):
    from hermes_cli.source_releases import resolve_source_release

    url = "https://github.com/Fixture/hermes-agent.git"
    git(releases.root, "config", "remote.origin.url", url)
    git(releases.root, "config", f"url.{releases.origin}.insteadOf", url)
    tag = releases.tags["stable"]
    git(releases.origin, "tag", "-f", tag, releases.commits[3])
    releases.responses["/repos/Fixture/hermes-agent/releases/latest"] = {
        "tag_name": tag, "draft": False, "prerelease": False,
    }
    releases.responses[f"/repos/Fixture/hermes-agent/commits/{tag}"] = {"sha": releases.commits[3]}
    assert resolve_source_release("stable", ["git"], releases.root) == (tag, releases.commits[3])
    assert not any(path.startswith("/releases/") for path in releases.requests)


def test_zip_fallback_keeps_selected_repository_and_commit(releases, monkeypatch):
    from hermes_cli import update_cmd_zip

    seen = []
    monkeypatch.setattr(update_cmd_zip, "_abort_zip_update_if_dirty_tree", lambda: None)
    class DownloadBoundary(Exception):
        pass
    def download(branch, url):
        seen.append(url)
        raise DownloadBoundary
    monkeypatch.setattr(update_cmd_zip, "_download_and_swap_zip", download)
    with pytest.raises(DownloadBoundary):
        update_cmd_zip._update_via_zip(
            SimpleNamespace(branch=None), target_sha=releases.commits[2],
            target_repository="Fixture/hermes-agent")
    assert seen == [f"https://github.com/Fixture/hermes-agent/archive/{releases.commits[2]}.zip"]


def test_selected_draft_never_falls_back_to_other_tags(releases):
    releases.responses[f"/repos/NousResearch/hermes-agent/releases/tags/{releases.tags['stable']}"]["draft"] = True
    assert update_cmd._resolve_latest_release_tag(["git"], releases.root) == (None, None)
    assert not any("/tags?" in path for path in releases.requests)


def test_main_check_still_uses_branch_without_release_requests(releases, capsys):
    set_install_channel("main", releases.root)
    update_cmd._cmd_update_check()
    assert "behind origin/main" in capsys.readouterr().out
    assert releases.requests == []


@pytest.mark.parametrize("channel", ["stable", "canary"])
def test_origin_tag_cannot_substitute_a_fork_commit(releases, channel):
    from hermes_cli.source_releases import resolve_source_release

    git(releases.origin, "tag", "-f", releases.tags[channel], releases.commits[3])
    assert resolve_source_release(channel, ["git"], releases.root) == (None, None)
    # Without git, the same selection remains pinned to the official commit.
    assert resolve_source_release(channel) == (
        releases.tags[channel], releases.commits[1 if channel == "stable" else 2],
    )


@pytest.mark.parametrize("channel", ["stable", "canary"])
def test_missing_pointers_fall_back_only_to_published_releases(releases, channel):
    from hermes_cli.source_releases import resolve_source_release

    releases.responses.pop("/releases/stable/release-candidates.json")
    releases.responses.pop(f"/releases/{channel}/index.html")
    published = releases.responses[f"/repos/NousResearch/hermes-agent/releases/tags/{releases.tags[channel]}"]
    releases.responses["/repos/NousResearch/hermes-agent/releases/latest"] = published
    releases.responses["/repos/NousResearch/hermes-agent/releases?per_page=100&page=1"] = [
        {"tag_name": "v99.0.1-canary.20260912125822", "draft": True, "prerelease": True},
        published,
    ]
    assert resolve_source_release(channel, ["git"], releases.root) == (
        releases.tags[channel], releases.commits[1 if channel == "stable" else 2],
    )
    assert not any("/tags?" in path for path in releases.requests)
    assert f"/repos/NousResearch/hermes-agent/releases/tags/{releases.tags[channel]}" not in releases.requests
