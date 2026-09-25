"""Exercise the passive checker with real linked worktrees and loopback HTTP."""
import json
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

# The fixture below serves channel records over loopback; the reader's real HTTP path is the subject.
pytestmark = pytest.mark.real_release_channels

MAIN_CHANNEL = "/releases/channels/main.json"


def source_channel(name, repository, branch="main"):
    return {"schema": 1, "name": name, "repository": repository, "policy": "source-branch",
            "state": "active", "revision": 1, "nextSequence": 1, "identity": None, "head": None,
            "delivery": {"kind": "source-branch", "branch": branch}}


class Installation(tuple):
    """The 8-tuple every test unpacks, plus the GitHub probe's request/response hooks."""
    authorizations: list
    response_headers: dict


@pytest.fixture
def installation(tmp_path, monkeypatch):
    from hermes_cli import source_releases

    root = tmp_path / "checkout"
    root.mkdir()
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    # Git probes may use real local remotes, never the external network.
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")

    def git(*args, cwd=root):
        return subprocess.check_output([
            "git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", *args,
        ], cwd=cwd, text=True).strip()

    git("init", "-b", "main")
    git("commit", "--allow-empty", "-m", "base")
    base = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-m", "local")
    head = git("rev-parse", "HEAD")
    git("remote", "add", "origin", "https://github.com/fixture/fork.git")
    linked = tmp_path / "linked"
    git("worktree", "add", "-b", "feature/gui", str(linked))
    responses = {MAIN_CHANNEL: (200, lambda: source_channel(
        "main", source_releases.source_repository(["git"], root)))}
    requests = []
    # Authorization header of every api.github.com call, in request order (None when anonymous).
    authorizations = []
    # Optional extra response headers per path (rate-limit headers for the failure-copy tests).
    response_headers = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            authorizations.append(self.headers.get("Authorization"))
            entry = responses.get(self.path, (404, {}))
            # A callable entry decides per request (it sees the handler, hence the headers).
            code, body = entry(self) if callable(entry) else entry
            if callable(body):
                body = body()
            self.send_response(code)
            for name, value in response_headers.get(self.path, {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write((body if isinstance(body, str) else json.dumps(body)).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(source_releases, "_PUBLIC_BASE", f"http://127.0.0.1:{server.server_port}")
    original = urllib.request.urlopen

    def local(request, *args, **kwargs):
        url = urlsplit(request.full_url)
        assert url.hostname in {"api.github.com", "hermes-assets.nousresearch.com"}
        rewritten = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{url.path}" + (f"?{url.query}" if url.query else ""),
            headers=dict(request.header_items()))
        return original(rewritten, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", local)
    # The credential ladder must not read this machine's gh login or env.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    from hermes_cli import github_api
    monkeypatch.setattr(github_api, "_gh_cli_token", lambda: None)
    fixture = Installation((root, linked, home, base, head, responses, requests, git))
    fixture.authorizations = authorizations
    fixture.response_headers = response_headers
    yield fixture
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("tip_status", [200, 503])
def test_in_place_check_targets_main_without_moving_checkout(installation, tip_status):
    """The in-place update source, not an unpublished local branch, owns the check."""
    from hermes_cli.source_check import check_for_updates

    root, linked, home, base, head, responses, requests, git = installation
    config = home / "config.yaml"
    branch_file = home / "desktop-update.json"
    branch_file.write_text("{}")
    local_url = "/repos/fixture/fork/commits/feature%2Fgui"
    main_url = "/repos/fixture/fork/commits/main"
    responses[local_url] = (422, {"message": "No commit found for SHA"})
    responses[main_url] = (tip_status, base if tip_status == 200 else {})
    before = {str(p.relative_to(root)): p.read_bytes()
              for p in (root / ".git").rglob("*") if p.is_file()}
    check = lambda: check_for_updates(install_root=linked, home=home,
                                     branch_config_path=branch_file)
    # Seed the failure cache, then opt in without forcing a refresh.
    assert check()["error"] == "fetch-failed"
    config.write_text("updates: {parked_branch_strategy: update_in_place}\n")
    saved = config.read_bytes()
    requests.clear()
    status = check()
    assert status["branch"] == "main", status
    assert local_url not in requests and main_url in requests
    if tip_status == 200:
        assert "error" not in status, status
        assert status["behind"] == 0 and not status["updateAvailable"]
    else:
        assert status["error"] == "fetch-failed"
        assert "HTTP 503" in status["message"]
    assert status["currentBranch"] == "feature/gui"
    assert status["currentSha"] == head
    assert config.read_bytes() == saved
    assert branch_file.read_text() == "{}"
    assert {str(p.relative_to(root)): p.read_bytes()
            for p in (root / ".git").rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("selection", ["explicit", "configured", "default", "custom-channel"])
def test_in_place_check_preserves_caller_branch_and_channel(installation, selection):
    from hermes_cli.source_check import check_for_updates

    root, linked, home, base, head, responses, requests, git = installation
    config = home / "config.yaml"
    config.write_text("updates: {parked_branch_strategy: update_in_place}\n"
                      if selection != "default" else "{}")
    branch_file = home / "desktop-update.json"
    branch_file.write_text(json.dumps({"branch": "desktop-choice"})
                           if selection in {"explicit", "configured"} else "{}")
    expected = {"explicit": "explicit-choice", "configured": "desktop-choice"}.get(selection, "feature/gui")
    channel = "branch-" + uuid4().hex[:12] if selection == "custom-channel" else "main"
    responses[f"/releases/channels/{channel}.json"] = (200, source_channel(channel, "fixture/fork"))
    path = f"/repos/fixture/fork/commits/{expected.replace('/', '%2F')}"
    responses[path] = (200, head)
    saved = branch_file.read_bytes()
    status = check_for_updates(install_root=linked, home=home, channel=channel,
                              branch="explicit-choice" if selection == "explicit" else None,
                              branch_config_path=branch_file)
    assert status["branch"] == expected and status["targetSha"] == head
    assert status["behind"] == 0
    assert branch_file.read_bytes() == saved
    assert status["currentBranch"] == git("branch", "--show-current", cwd=linked) == "feature/gui"


def test_target_worktree_owns_admission_and_fork_comparison(installation, monkeypatch):
    from hermes_cli.source_check import check_for_updates

    root, linked, home, base, head, responses, requests, git = installation
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "external"}))
    (linked / "install-stamp.json").write_text(json.dumps({"updateMechanism": "self"}))
    cache = home / "shared-cache.json"
    target = "a" * 40
    responses["/repos/fixture/fork/commits/feature%2Fgui"] = (200, target)
    responses[f"/repos/fixture/fork/compare/{head}...{target}"] = (200, {"ahead_by": 3, "commits": []})
    commands = []
    original = subprocess.run

    def record(args, **kwargs):
        commands.append(args)
        return original(args, **kwargs)

    before = {str(p.relative_to(root)): p.read_bytes() for p in (root / ".git").rglob("*") if p.is_file()}
    monkeypatch.setattr(subprocess, "run", record)
    status = check_for_updates(install_root=linked, home=home, cache_path=cache)
    assert status["supported"] is True
    assert status["branch"] == "feature/gui"
    assert status["behind"] == 3
    assert status["hermesRoot"] == str(linked)
    assert check_for_updates(install_root=root, home=home, cache_path=cache)["supported"] is False
    assert requests == [MAIN_CHANNEL, "/repos/fixture/fork/commits/feature%2Fgui",
                        f"/repos/fixture/fork/compare/{head}...{target}"]
    assert all(not any(arg in {"fetch", "checkout", "reset", "update-ref", "stash"} for arg in cmd) for cmd in commands)
    assert git("rev-parse", "HEAD", cwd=linked) == head
    assert {str(p.relative_to(root)): p.read_bytes() for p in (root / ".git").rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("tip_kind,compare,expected", [
    ("head", None, 0), ("base", None, 0),
    ("unknown", {"ahead_by": 61}, 61), ("unknown", {"ahead_by": 0}, 0),
    ("unknown", None, -1), ("unknown", {"ahead_by": True}, -1),
])
def test_counts_are_honest_without_fetch(installation, tip_kind, compare, expected):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    target = {"head": head, "base": base, "unknown": "a" * 40}[tip_kind]
    responses["/repos/fixture/fork/commits/main"] = (200, "\ufeff" + target)
    if compare is not None:
        responses[f"/repos/fixture/fork/compare/{head}...{target}"] = (200, "\ufeff" + json.dumps(compare))
    status = check_for_updates(install_root=root, home=home)
    assert status["behind"] == expected
    assert status["updateAvailable"] is (expected != 0)
    if tip_kind != "unknown":
        assert requests == [MAIN_CHANNEL, "/repos/fixture/fork/commits/main"]


def test_cache_force_expiry_and_passive_opt_out(installation, monkeypatch):
    from hermes_cli import source_check
    root, linked, home, base, head, responses, requests, git = installation
    clock = [1000000.0]
    monkeypatch.setattr(source_check.time, "time", lambda: clock[0])
    url = "/repos/fixture/fork/commits/main"
    responses[url] = (200, head)
    check = lambda **kw: source_check.check_for_updates(install_root=root, home=home, **kw)
    assert check()["behind"] == 0
    cache = next((home / "source-checks").glob("*.json"))
    raw = cache.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    cache.write_bytes(b"\xef\xbb\xbf" + raw)
    responses[url] = (503, {})
    (root / "dirty.txt").write_text("carried work")
    assert check()["dirty"] is True
    assert requests.count(url) == requests.count(MAIN_CHANNEL) == 1
    assert check(force=True)["error"] == "fetch-failed"
    clock[0] += 3599
    assert check()["error"] == "fetch-failed"
    assert requests.count(url) == requests.count(MAIN_CHANNEL) == 2
    clock[0] += 2
    responses[url] = (200, head)
    assert check()["behind"] == 0
    assert requests.count(url) == requests.count(MAIN_CHANNEL) == 3
    clock[0] += 86401
    assert check()["behind"] == 0
    assert requests.count(url) == requests.count(MAIN_CHANNEL) == 4
    (home / "config.yaml").write_text("updates: {check: false}")
    assert check(passive=True)["behind"] is None
    assert check()["behind"] == 0
    assert requests.count(url) == requests.count(MAIN_CHANNEL) == 4
    git("commit", "--allow-empty", "-m", "moved")
    responses[url] = (200, git("rev-parse", "HEAD"))
    assert check()["behind"] == 0
    assert requests.count(url) == requests.count(MAIN_CHANNEL) == 5


def test_explicit_and_current_branch_heal_only_after_confirmed_absence(installation, monkeypatch):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    git("remote", "set-url", "origin", str(root))
    # Use the same real linked worktree with a local origin. No GitHub fallback is involved.
    assert check_for_updates(install_root=linked, home=home)["branch"] == "feature/gui"
    assert check_for_updates(install_root=linked, home=home, branch="main")["branch"] == "main"
    assert check_for_updates(install_root=linked, home=home, branch="deleted")["branch"] == "main"
    git("remote", "set-url", "origin", str(home / "unreachable"))
    failed = check_for_updates(install_root=linked, home=home, branch="deleted", force=True)
    assert failed["branch"] == "deleted"
    assert failed["error"] == "fetch-failed"
    assert git("branch", "--show-current", cwd=linked) == "feature/gui"
    assert requests == [MAIN_CHANNEL]


@pytest.mark.parametrize("selection", ["explicit", "configured", "current", "detached"])
def test_dynamic_source_channel_preserves_branch_precedence(installation, selection):
    from hermes_cli.source_check import check_for_updates

    root, linked, home, base, head, responses, requests, git = installation
    name = "branch-" + uuid4().hex[:12]
    channel_path = f"/releases/channels/{name}.json"
    responses[channel_path] = (200, source_channel(name, "fixture/fork", "channel-default"))
    branch_file = home / "desktop-update.json"
    if selection in {"explicit", "configured"}:
        branch_file.write_text(json.dumps({"branch": "desktop-choice"}))
    if selection == "detached":
        git("checkout", "--detach", cwd=linked)
    expected = {"explicit": "explicit-choice", "configured": "desktop-choice",
                "current": "feature/gui", "detached": "channel-default"}[selection]
    branch_path = f"/repos/fixture/fork/commits/{expected.replace('/', '%2F')}"
    responses[branch_path] = (200, head)

    status = check_for_updates(install_root=linked, home=home, channel=name,
                              branch="explicit-choice" if selection == "explicit" else None,
                              branch_config_path=branch_file)
    assert status.get("branch") == expected, status
    assert status.get("targetSha") == head, status
    assert status["behind"] == 0
    assert requests == ([] if selection == "explicit" else [channel_path]) + [branch_path]


# An unpublished ``main`` record is not a channel failure: main IS the source
# branch, so the check keeps following a branch via git, with the usual branch
# precedence (test_unpublished_main_record_follows_the_branch below).
@pytest.mark.parametrize("name,failure", [
    ("stable", "missing"), ("canary", "missing"),
    (None, "missing"), (None, "malformed"), (None, "foreign"), (None, "unpublished"),
])
def test_channel_failure_never_probes_or_heals_a_branch(installation, name, failure):
    from hermes_cli.source_check import check_for_updates

    root, linked, home, base, head, responses, requests, git = installation
    name = name or "preview-" + uuid4().hex[:12]
    channel_path = f"/releases/channels/{name}.json"
    body = source_channel(name, "fixture/fork")
    code = 200
    if failure == "missing":
        code = 404
    elif failure == "malformed":
        body = "not JSON"
    elif failure == "foreign":
        body["repository"] = "NousResearch/hermes-agent"
    else:
        body["policy"] = "preview"
        del body["delivery"]
        body["identity"] = {
            "token": "a" * 16, "displayName": "Fixture", "appNamePascal": "Fixture",
            "artifactNamePascal": "Fixture", "appId": "ai.fixture.preview",
            "msixAppIdWithOrg": "Fixture.Preview", "cliName": "fixture-preview",
            "windowsExecutableName": "Fixture",
        }
    responses[channel_path] = (code, body)
    responses["/repos/fixture/fork/commits/main"] = (200, head)
    branch_file = home / "desktop-update.json"
    branch_file.write_text(json.dumps({"branch": "deleted"}))
    status = check_for_updates(install_root=linked, home=home, channel=name,
                              branch_config_path=branch_file)
    assert status.get("error") == "release-unavailable", status
    assert "targetSha" not in status
    assert status["behind"] is None
    assert requests == [channel_path]
    assert json.loads(branch_file.read_text())["branch"] == "deleted"


def test_unpublished_main_record_follows_the_branch(installation):
    """A 404 for main.json keeps a checkout updating via git: the configured
    branch is probed exactly as for a published source-branch channel, and the
    Desktop branch setting is left alone."""
    from hermes_cli.source_check import check_for_updates

    root, linked, home, base, head, responses, requests, git = installation
    responses[MAIN_CHANNEL] = (404, source_channel("main", "fixture/fork"))
    branch_path = "/repos/fixture/fork/commits/desktop-choice"
    responses[branch_path] = (200, head)
    branch_file = home / "desktop-update.json"
    branch_file.write_text(json.dumps({"branch": "desktop-choice"}))
    status = check_for_updates(install_root=linked, home=home, channel="main",
                               branch_config_path=branch_file)
    assert "error" not in status, status
    assert status.get("branch") == "desktop-choice", status
    assert status.get("targetSha") == head, status
    assert status["behind"] == 0
    assert requests == [MAIN_CHANNEL, branch_path]
    assert json.loads(branch_file.read_text())["branch"] == "desktop-choice"


def test_running_revision_is_not_applied_to_an_explicit_target(installation, monkeypatch):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    monkeypatch.setenv("HERMES_REVISION", "e" * 40)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(root))
    responses["/repos/fixture/fork/commits/feature%2Fgui"] = (200, head)
    assert check_for_updates(install_root=linked, home=home)["currentSha"] == head
    # Default invocation retains the Nix revision probe even without a Git checkout.
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: home)
    responses[MAIN_CHANNEL] = (200, source_channel("main", "NousResearch/hermes-agent"))
    responses["/repos/NousResearch/hermes-agent/commits/main"] = (200, "e" * 40)
    assert check_for_updates(home=home)["behind"] == 0


def test_deleted_desktop_branch_is_persisted_only_after_definitive_probe(installation):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    branch_file = home / "desktop-update.json"
    branch_file.write_text(json.dumps({"branch": "deleted", "other": "café"}, ensure_ascii=False),
                           encoding="utf-8-sig")
    git("remote", "set-url", "origin", str(home / "unreachable"))
    status = check_for_updates(install_root=linked, home=home, branch_config_path=branch_file)
    assert status["branch"] == "deleted"
    assert json.loads(branch_file.read_bytes())["branch"] == "deleted"
    git("remote", "set-url", "origin", str(root))
    status = check_for_updates(install_root=linked, home=home, branch_config_path=branch_file)
    assert status["branch"] == "main"
    assert json.loads(branch_file.read_text(encoding="utf-8")) == {"branch": "main", "other": "café"}
    assert not branch_file.read_bytes().startswith(b"\xef\xbb\xbf")


def test_inherited_git_target_cannot_redirect_an_explicit_install(installation, monkeypatch):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    responses["/repos/fixture/fork/commits/feature%2Fgui"] = (200, head)
    monkeypatch.setenv("GIT_DIR", str(root / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(root))
    status = check_for_updates(install_root=linked, home=home)
    assert status["currentBranch"] == "feature/gui"
    assert status["behind"] == 0


@pytest.mark.parametrize("mechanism", ["external", "electron-updater", "app-installer", "microsoft-store", "self", None])
def test_source_admission_is_stamp_owned_not_path_owned(installation, mechanism):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    if mechanism:
        (linked / "install-stamp.json").write_text(json.dumps({"updateMechanism": mechanism, "distribution": "nix" if mechanism == "external" else "source"}))
    responses["/repos/fixture/fork/commits/feature%2Fgui"] = (200, head)
    status = check_for_updates(install_root=linked, home=home)
    assert status["supported"] is (mechanism in ("self", None))
    assert requests == ([MAIN_CHANNEL, "/repos/fixture/fork/commits/feature%2Fgui"]
                        if status["supported"] else [])
    empty = home / "no-source"
    empty.mkdir()
    (empty / "install-stamp.json").write_text(json.dumps({"updateMechanism": mechanism, "distribution": "nix" if mechanism == "external" else "source"}))
    assert check_for_updates(install_root=empty, home=home)["reason"] == "not-a-git-checkout"


def test_embedded_revision_keeps_https_ref_advertisement_recovery(installation, monkeypatch):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    monkeypatch.setenv("HERMES_REVISION", head)
    monkeypatch.setattr("hermes_cli.config.get_project_root", lambda: home)
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda root: "nix")
    responses[MAIN_CHANNEL] = (200, source_channel("main", "NousResearch/hermes-agent"))
    original = subprocess.run
    probes = []

    def advertise(args, **kwargs):
        if "ls-remote" in args:
            probes.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, head + "\trefs/heads/main\n", "")
        return original(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", advertise)
    assert check_for_updates(home=home)["behind"] == 0
    assert len(probes) == 1
    assert "https://github.com/NousResearch/hermes-agent.git" in probes[0][0]
    assert probes[0][1]["stdin"] == subprocess.DEVNULL
    assert probes[0][1]["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_malformed_optional_changelog_and_cache_do_not_hide_the_update(installation):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    cache = home / "cache.json"
    responses["/repos/fixture/fork/commits/main"] = (200, "a" * 40)
    responses[f"/repos/fixture/fork/compare/{head}...{'a' * 40}"] = (200, {
        "ahead_by": 2, "commits": [{"sha": "b" * 40, "commit": 42}],
    })
    status = check_for_updates(install_root=root, home=home, cache_path=cache)
    assert status["behind"] == 2
    assert status["updateAvailable"] is True
    data = json.loads(cache.read_text())
    data["status"] = None
    cache.write_text(json.dumps(data))
    assert check_for_updates(install_root=root, home=home, cache_path=cache)["behind"] == 2
    assert requests == [MAIN_CHANNEL, "/repos/fixture/fork/commits/main",
                        f"/repos/fixture/fork/compare/{head}...{'a' * 40}"] * 2


@pytest.mark.parametrize("repository,heals", [("NousResearch/hermes-agent", True), ("fixture/fork", False)])
def test_official_ssh_healing_uses_public_https_without_retargeting_forks(installation, monkeypatch, repository, heals):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    git("remote", "set-url", "origin", f"git@github.com:{repository}.git")
    git("config", f"url.{root.as_uri()}.insteadOf", "https://github.com/NousResearch/hermes-agent.git")
    monkeypatch.setenv("GIT_SSH_COMMAND", "false")
    branch_file = home / "desktop-update.json"
    branch_file.write_text(json.dumps({"branch": "deleted"}))
    responses[f"/repos/{repository}/commits/main"] = (200, head)
    status = check_for_updates(install_root=linked, home=home, branch_config_path=branch_file)
    assert status["branch"] == ("main" if heals else "deleted")
    assert json.loads(branch_file.read_text())["branch"] == status["branch"]
    if heals:
        assert status["behind"] == 0
    else:
        assert status["error"] == "fetch-failed"


def github_authorizations(installation, path):
    """Authorization header of each api.github.com call to ``path``, in order."""
    return [auth for seen, auth in zip(installation[6], installation.authorizations) if seen == path]


def test_github_calls_carry_the_configured_token_and_retry_anonymously_on_401(installation, monkeypatch):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    url = "/repos/fixture/fork/commits/main"
    responses[url] = (200, head)
    monkeypatch.setenv("GITHUB_TOKEN", "  ghp_fixture  ")
    assert check_for_updates(install_root=root, home=home)["behind"] == 0
    assert github_authorizations(installation, url) == ["Bearer ghp_fixture"]

    # A rejected token is not a failed check: the same request is retried without it.
    responses[url] = lambda handler: (401, {}) if handler.headers.get("Authorization") else (200, head)
    assert check_for_updates(install_root=root, home=home, force=True)["behind"] == 0
    assert github_authorizations(installation, url)[-2:] == ["Bearer ghp_fixture", None]


def test_branch_tip_failure_names_the_cause(installation):
    from hermes_cli.source_check import check_for_updates
    root, linked, home, base, head, responses, requests, git = installation
    url = "/repos/fixture/fork/commits/main"
    responses[url] = (403, {})
    installation.response_headers[url] = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "4102444800"}
    status = check_for_updates(install_root=root, home=home)
    assert status["error"] == "fetch-failed"
    assert "rate limit" in status["message"] and "GITHUB_TOKEN" in status["message"]
    assert github_authorizations(installation, url) == [None]

    installation.response_headers.pop(url)
    responses[url] = (503, {})
    status = check_for_updates(install_root=root, home=home, force=True)
    assert status["error"] == "fetch-failed"
    assert "HTTP 503" in status["message"]
