"""Behavioral publication gates, using disposable signed Darwin bundles."""
import datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.macos_only


def updater():
    assert importlib.util.find_spec("hermes_cli.desktop_macos_update"), "missing bounded macOS updater"
    return importlib.import_module("hermes_cli.desktop_macos_update")


def run(*args, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, **kwargs)


def sign(app):
    run("/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app))


@pytest.fixture
def bundles(tmp_path, native_program):
    root = tmp_path / "home/hermes-agent"
    desktop = root / "apps/desktop"
    desktop.mkdir(parents=True)
    package = {"version": "1.2.3", "build": {"appId": "test.hermes.updater." + uuid.uuid4().hex}}
    (desktop / "package.json").write_text(json.dumps(package))
    (root / "hermes_cli").mkdir()
    (root / "hermes_cli/__init__.py").write_text('__version__ = "4.5.6"\n')
    run("git", "init", "-q", str(root))
    run("git", "-C", str(root), "add", ".")
    run("git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture source")
    sha = run("git", "-C", str(root), "rev-parse", "HEAD").stdout.decode().strip()
    candidate = desktop / "release/mac-arm64/Hermes.app"
    if os.uname().machine != "arm64":
        candidate = desktop / "release/mac/Hermes.app"
    resources = candidate / "Contents/Resources"
    resources.mkdir(parents=True)
    executable = candidate / "Contents/MacOS/Hermes"
    executable.parent.mkdir()
    shutil.copyfile(native_program, executable)
    executable.chmod(0o755)
    info = {"CFBundleIdentifier": package["build"]["appId"], "CFBundleExecutable": "Hermes",
            "CFBundleShortVersionString": package["version"], "CFBundlePackageType": "APPL"}
    (candidate / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
    stamp = {"schemaVersion": 1, "commit": sha, "source": "local", "dirty": False,
             "builtAt": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (resources / "install-stamp.json").write_text(json.dumps(stamp))
    (resources / "app.asar").write_bytes(b"fixture archive")
    (resources / "app.asar.unpacked/dist").mkdir(parents=True)
    (resources / "app.asar.unpacked/dist/index.html").write_text("fixture renderer")
    sign(candidate)
    target = tmp_path / "Applications/Hermes.app"
    target.parent.mkdir()
    shutil.copytree(candidate, target, symlinks=True)
    return root, candidate, target


@pytest.fixture(scope="session")
def native_program(tmp_path_factory):
    program = tmp_path_factory.mktemp("native-program") / "fixture"
    run("/usr/bin/clang", "-x", "c", "-", "-o", str(program),
        input=b'''#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <libgen.h>
#include <mach-o/dyld.h>
int main(int n,char**v){
 if(n>1){sleep(atoi(v[1]));return 0;}
 char exe[4096],script[4096],helper[4096]; unsigned size=sizeof(exe);
 if(_NSGetExecutablePath(exe,&size))return 2;
 char *dir=dirname(exe);
 snprintf(script,sizeof(script),"%s/../Resources/fixture-launch.sh",dir);
 if(access(script,R_OK))return 0;
 snprintf(helper,sizeof(helper),"%s/../Frameworks/Hermes Helper (Renderer).app/Contents/MacOS/Hermes Helper (Renderer)",dir);
 if(access(helper,X_OK))snprintf(helper,sizeof(helper),"%s/../Frameworks/Hermes Helper (GPU).app/Contents/MacOS/Hermes Helper (GPU)",dir);
 if(fork()==0){execl(helper,helper,"120",NULL);_exit(2);}
 if(fork()==0){execl("/bin/bash","/bin/bash",script,NULL);_exit(2);}
 sleep(120);return 0;
}''')
    return program


def test_validate_and_hash_signed_candidate_including_unpacked_assets(bundles):
    root, candidate, _ = bundles
    update = updater()
    expected = update.expectation(root)
    valid = update.validate(candidate, expected)
    (candidate / "Contents/Resources/app.asar.unpacked/dist/index.html").write_text("changed")
    sign(candidate)
    assert update.validate(candidate, expected).tree != valid.tree


@pytest.mark.parametrize("damage", ["commit", "dirty", "schema", "timestamp", "future", "fallback",
                                     "identity", "executable", "version", "signature", "architecture"])
def test_invalid_candidate_refused_before_publication(bundles, damage, monkeypatch):
    root, candidate, target = bundles
    update = updater()
    expected = update.expectation(root)
    original = update.fingerprint(target)
    stamp_path = candidate / "Contents/Resources/install-stamp.json"
    stamp = json.loads(stamp_path.read_text())
    changes = {"commit": {"commit": "a" * 40}, "dirty": {"dirty": True},
               "schema": {"schemaVersion": 2}, "timestamp": {"builtAt": "invalid"},
               "future": {"builtAt": "2999-01-01T00:00:00Z"}, "fallback": {"source": "fallback"}}
    if damage in changes:
        stamp.update(changes[damage])
        stamp_path.write_text(json.dumps(stamp))
        sign(candidate)
    elif damage in ("identity", "executable", "version"):
        info_path = candidate / "Contents/Info.plist"
        info = plistlib.loads(info_path.read_bytes())
        info[{"identity": "CFBundleIdentifier", "executable": "CFBundleExecutable",
              "version": "CFBundleShortVersionString"}[damage]] = "wrong"
        info_path.write_bytes(plistlib.dumps(info))
        sign(candidate)
    elif damage == "signature":
        (candidate / "Contents/MacOS/Hermes").write_bytes(b"bad")
    else:
        expected = {**expected, "arch": "incompatible"}
    with pytest.raises((ValueError, RuntimeError, OSError)):
        update.validate(candidate, expected)
    assert update.fingerprint(target) == original


def test_select_refuses_ambiguous_or_missing_candidate(bundles):
    root, candidate, _ = bundles
    update = updater()
    expected = update.expectation(root)
    assert update.select_candidate(root, expected) == candidate
    duplicate = candidate.parents[1] / "mac-universal/Hermes.app"
    shutil.copytree(candidate, duplicate)
    with pytest.raises(ValueError, match="ambiguous"):
        update.select_candidate(root, expected)
    shutil.rmtree(duplicate)
    shutil.rmtree(candidate)
    with pytest.raises(ValueError):
        update.select_candidate(root, expected)


@pytest.mark.parametrize("layout", ["inside-release", "source-ancestor", "symlink", "not-app"])
def test_preflight_rejects_unsafe_publication_layout(bundles, layout):
    root, candidate, target = bundles
    update = updater()
    if layout == "inside-release":
        target = candidate
    elif layout == "source-ancestor":
        target = root.parent
    elif layout == "symlink":
        link = target.parent / "Link.app"
        link.symlink_to(target)
        target = link
    else:
        target = target.parent
    with pytest.raises((ValueError, RuntimeError)):
        update.preflight(root, target)


def test_native_exchange_retains_both_identities_and_refuses_cross_volume(tmp_path, monkeypatch):
    update = updater()
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "identity").write_text("old")
    (right / "identity").write_text("new")
    update.exchange(left, right)
    assert (left / "identity").read_text() == "new"
    assert (right / "identity").read_text() == "old"
    update.exchange(left, right)
    assert (left / "identity").read_text() == "old"
    real_stat = Path.stat

    def different_volume(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path == right:
            fields = list(result)
            fields[2] += 1
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(Path, "stat", different_volume)
    with pytest.raises(ValueError, match="volume"):
        update.exchange(left, right)
    assert (left / "identity").read_text() == "old"


def shell_handoff(root, target):
    """Execute production shell against a stub updater; deny production writes/notifications."""
    import sys
    bin_dir = root / "venv/bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("python", "python3"):
        (bin_dir / name).symlink_to(sys.executable)
    hermes = bin_dir / "hermes"
    hermes.write_text('#!/bin/bash\ncase "$*" in *--help*) echo --keep-stash;; *) echo fixture-update;; esac\n')
    hermes.chmod(0o755)
    script = Path(__file__).resolve().parents[2] / "scripts/desktop-update/posix.sh"
    env = {"PATH": os.environ["PATH"], "HOME": str(root.parent),
           "TMPDIR": str(root.parent), "HERMES_UPDATE_SHIM_GRACE_SECONDS": "0"}
    policy = '(version 1)(allow default)(deny file-write* (subpath "/Applications") (subpath "/Users/mini/.hermes"))(deny process-exec (literal "/usr/bin/osascript"))'
    return subprocess.run(["/usr/bin/sandbox-exec", "-p", policy, "/bin/bash", str(script),
                           "--daemonized", "--install-root", str(root), "--desktop-pid", "0",
                           "--no-ui", "--relaunch-target", str(target)],
                          env=env, capture_output=True, text=True, timeout=150)


def test_handoff_preserves_historical_rollback(bundles):
    root, _, target = bundles
    historical = target.with_name(target.name + ".old")
    historical.mkdir()
    (historical / "recovery").write_text("historical recovery")
    shell_handoff(root, target)
    assert (historical / "recovery").read_text() == "historical recovery"


def test_handoff_launch_acceptance_without_readiness_is_not_success(bundles):
    root, _, target = bundles
    completed = shell_handoff(root, target)
    result = json.loads((root.parent / ".hermes-update-result.json").read_text())
    assert result["ok"] is False
    assert result["exit_code"] != 0
    assert completed.returncode != 0


@pytest.mark.parametrize("reader", ["main", "helper", "renderer", "mapped-only"])
def test_native_observation_finds_exact_bundle_readers(bundles, reader):
    import sys
    root, _, target = bundles
    update = updater()
    assert hasattr(update, "observe"), "reader observation gate is missing"
    def observed():
        import time
        # Live unrelated processes can exit during a scan. Unknown is safe; retry
        # a fresh scan, never turn it into absence or accept an incorrect known state.
        for _ in range(3):
            result = update.observe([target])
            if result.state != "unknown":
                return result
            time.sleep(0.2)
        pytest.fail(result.reason)

    assert observed().state == "none"
    executable = target / "Contents/MacOS/Hermes"
    if reader in ("helper", "renderer"):
        helper = target / f"Contents/Frameworks/{reader}.app/Contents/MacOS/{reader}"
        helper.parent.mkdir(parents=True)
        shutil.copyfile(executable, helper)
        helper.chmod(0o755)
        executable = helper
    if reader == "mapped-only":
        process = subprocess.Popen([sys.executable, "-c",
            "import mmap,os,sys; f=os.open(sys.argv[1],os.O_RDONLY); "
            "m=mmap.mmap(f,0,access=mmap.ACCESS_READ,trackfd=False); os.close(f); "
            "print('ready',flush=True); sys.stdin.read()",
            str(target / "Contents/Resources/app.asar")], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert process.stdout.readline() == b"ready\n"
    else:
        process = subprocess.Popen([str(executable), "30"])
    try:
        found = observed()
        assert found.state == "present", found
        assert process.pid in found.readers[target]
    finally:
        process.terminate()
        process.wait(timeout=10)
        if process.stdin:
            process.stdin.close()
            process.stdout.close()
    assert observed().state == "none"


def test_native_scanner_explicit_errors_are_unknown(bundles, monkeypatch):
    _, _, target = bundles
    update = updater()
    assert hasattr(update, "observe"), "reader observation gate is missing"

    def denied():
        raise PermissionError("fixture denied inspection")

    monkeypatch.setattr(update, "process_table", denied)
    assert update.observe([target]).state == "unknown"


@pytest.mark.parametrize("change", ["rotate", "truncate", "rewrite"])
def test_log_cursor_rejects_ambiguous_history(tmp_path, change):
    update = updater()
    assert hasattr(update, "LogCursor"), "fresh-log evidence gate is missing"
    log = tmp_path / "desktop.log"
    log.write_text("old readiness evidence\n")
    cursor = update.LogCursor(log)
    with log.open("a") as stream:
        stream.write("fresh startup\n")
    assert cursor.read() == "fresh startup\n"
    if change == "rotate":
        log.rename(tmp_path / "desktop.log.old")
        log.write_text("a new log with enough bytes to exceed the old offset\n")
    else:
        log.write_text("" if change == "truncate" else "changed prefix with a longer suffix than before\n")
    with pytest.raises(RuntimeError):
        cursor.read()


def test_fresh_remote_event_requires_canonical_process_and_renderer(bundles, monkeypatch):
    update = updater()
    assert hasattr(update, "adopt"), "adoption gate is missing"
    root, _, target = bundles
    expected = update.expectation(root)
    log = root.parent / "logs/desktop.log"
    log.parent.mkdir()
    log.write_text("Remote Hermes backend is ready\n")
    launched = []

    def start(app):
        launched.append(subprocess.Popen([str(app / "Contents/MacOS/Hermes"), "30"]))

    monkeypatch.setattr(update, "launch_app", start)
    try:
        with pytest.raises(RuntimeError):
            update.adopt(target, expected, log, startup=0.5, stability=0.2)
    finally:
        for process in launched:
            process.terminate()
            process.wait(timeout=10)


def test_marker_cleanup_is_owner_scoped(tmp_path):
    update = updater()
    assert hasattr(update, "clear_marker"), "owner-scoped marker handling is missing"
    marker = tmp_path / ".hermes-update-in-progress"
    marker.write_text("123\n1000\n")
    update.clear_marker(marker, 456)
    assert marker.read_text() == "123\n1000\n"
    update.clear_marker(marker, 123)
    assert not marker.exists()


def install_launch_fixture(app, log, mode="local", version="4.5.6", helper_kind="Renderer"):
    """Native shell/renderer + real HTTP child, using the existing log events only."""
    import shlex
    import sys
    resources = app / "Contents/Resources"
    name = f"Hermes Helper ({helper_kind})"
    helper = app / f"Contents/Frameworks/{name}.app/Contents/MacOS/{name}"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(app / "Contents/MacOS/Hermes", helper)
    helper.chmod(0o755)
    backend = resources / "fixture-backend.py"
    backend.write_text('''import datetime,http.server,json,pathlib,sys,threading,urllib.request,time
log=pathlib.Path(sys.argv[1]);mode=sys.argv[2];version=sys.argv[3]
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  data=json.dumps({"ok":True,"version":version}).encode()
  self.send_response(200);self.end_headers();self.wfile.write(data)
 def log_message(self,*args):pass
server=http.server.HTTPServer(("127.0.0.1",0),Handler)
threading.Thread(target=server.serve_forever,daemon=True).start()
def write(message):
 with log.open("a") as f:f.write("["+datetime.datetime.now(datetime.timezone.utc).isoformat()+"] [hermes] "+message+"\\n")
port=server.server_address[1]
urllib.request.urlopen("http://127.0.0.1:"+str(port)+"/api/health").read()
if mode=="local":
 write("HERMES_BACKEND_READY port="+str(port))
 write("[boot] Hermes backend is ready. Finalizing desktop startup")
elif mode=="remote":write("[boot] Remote Hermes backend is ready")
elif mode=="unrelated":
 write("HERMES_BACKEND_READY port=1")
 write("[boot] Hermes backend is ready. Finalizing desktop startup")
elif mode=="die":
 write("[boot] Remote Hermes backend is ready");time.sleep(1);sys.exit(0)
while True:time.sleep(1)
''')
    (resources / "fixture-launch.sh").write_text("exec " + " ".join(map(shlex.quote,
        [sys.executable, str(backend), str(log), mode, version])) + "\n")
    sign(app)


def stop_fixture(app):
    import psutil
    # Only this disposable bundle's native processes and their current children.
    owned = {}
    for process in psutil.process_iter(["exe"]):
        if process.info["exe"] and Path(process.info["exe"]).is_relative_to(app):
            owned[process.pid] = process
            owned.update({child.pid: child for child in process.children(recursive=True)})
    for process in reversed(list(owned.values())):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(list(owned.values()), timeout=5)
    for process in alive:
        if process.status() != psutil.STATUS_ZOMBIE:
            process.kill()
    psutil.wait_procs(alive, timeout=5)


@pytest.mark.parametrize("mode", ["local", "remote"])
@pytest.mark.live_system_guard_bypass  # LaunchServices reparents our disposable apps to PID 1.
def test_native_launch_and_mode_appropriate_adoption(bundles, mode):
    update = updater()
    assert hasattr(update, "adopt"), "adoption gate is missing"
    root, _, target = bundles
    log = root.parent / "logs/desktop.log"
    log.parent.mkdir()
    log.write_text("historical log\n")
    install_launch_fixture(target, log, mode)
    try:
        # Real /usr/bin/open and native observation. Production default 90s/10s.
        adopted = update.adopt(target, update.expectation(root), log)
        assert adopted.pid > 0
        assert adopted.executable == target / "Contents/MacOS/Hermes"
    finally:
        stop_fixture(target)


@pytest.mark.parametrize("failure", ["adoption", "copy", "final-reader", "late-reader", "result-write",
                                     "after-swap-exception", "recovery-timeout"])
def test_publication_failures_preserve_recovery_and_never_report_success(bundles, monkeypatch, failure):
    update = updater()
    assert hasattr(update, "complete"), "controlled publication and recovery are missing"
    root, candidate, target = bundles
    (target / "Contents/Resources/app.asar").write_bytes(b"old distinct archive")
    sign(target)
    old = update.fingerprint(target)
    marker = root.parent / ".hermes-update-in-progress"
    marker.write_text(f"{os.getpid()}\n1000\n")
    attempts = []
    # Deterministic state-machine unit cases; separate native tests exercise the
    # real observer (including main/helper/renderer and closed-fd mappings).
    monkeypatch.setattr(update, "observe", lambda paths: update.Observation(
        "none", {p: set() for p in paths}, {}))

    def adoption(app, *args, **kwargs):
        attempts.append(update.fingerprint(app))
        if len(attempts) == 1:
            raise RuntimeError("fixture candidate adoption failed")
        return update.Process(99999, (1, 0), 1, app / "Contents/MacOS/Hermes")

    monkeypatch.setattr(update, "adopt", adoption)
    if failure == "after-swap-exception":
        real_exchange = update.exchange
        swapped = 0

        def interrupted(left, right):
            nonlocal swapped
            real_exchange(left, right)
            if left == target:
                swapped += 1
                if swapped == 1:
                    raise RuntimeError("interrupted immediately after exchange")

        monkeypatch.setattr(update, "exchange", interrupted)
        monkeypatch.setattr(update, "adopt", lambda *args, **kwargs: None)
    if failure == "recovery-timeout":
        real_validate = update.validate

        def timeout(app, expected):
            if attempts and Path(app).parent.name.startswith(".hermes-exchange-"):
                raise subprocess.TimeoutExpired("codesign", 30)
            return real_validate(app, expected)

        monkeypatch.setattr(update, "validate", timeout)
    if failure == "copy":
        real_command = update.command

        def torn_copy(*args, **kwargs):
            result = real_command(*args, **kwargs)
            if args[0] == "/usr/bin/ditto":
                (Path(args[2]) / "Contents/Resources/app.asar").write_bytes(b"torn")
            return result

        monkeypatch.setattr(update, "command", torn_copy)
    if failure in ("final-reader", "late-reader"):
        real_observe = update.observe
        calls = 0

        def new_reader(paths):
            nonlocal calls
            calls += 1
            # First scan before staging; second immediately before exchange.
            if calls >= (2 if failure == "final-reader" else 3):
                return update.Observation("unknown", {p: set() for p in paths}, {}, "fixture scan uncertainty")
            return real_observe(paths)

        monkeypatch.setattr(update, "observe", new_reader)
    if failure == "result-write":
        (root.parent / ".hermes-update-result.json").mkdir()
    outcome = update.complete(root, target, "main", os.getpid(), 0, "update completed")
    assert outcome.code != 0
    if failure not in ("late-reader", "recovery-timeout"):
        assert update.fingerprint(target) == old
    else:
        assert "manual" in outcome.message.lower()
    retained = list(target.parent.glob(".hermes-exchange-*/Hermes.app"))
    if failure in ("adoption", "late-reader", "result-write"):
        assert len(retained) == 1
    if failure != "result-write":
        result = json.loads((root.parent / ".hermes-update-result.json").read_text())
        assert result["ok"] is False
        assert result["exit_code"] == outcome.code
    if failure == "adoption":
        assert attempts[1] == old
        assert "restored" in outcome.message.lower()


@pytest.mark.live_system_guard_bypass
def test_gpu_child_cannot_satisfy_renderer_gate(bundles):
    update = updater()
    root, _, target = bundles
    log = root.parent / "logs/desktop.log"
    log.parent.mkdir()
    log.write_text("historical log\n")
    install_launch_fixture(target, log, "remote", helper_kind="GPU")
    try:
        with pytest.raises(RuntimeError):
            update.adopt(target, update.expectation(root), log, startup=4, stability=1)
    finally:
        stop_fixture(target)


@pytest.mark.parametrize("line", ["[renderer] Error: Remote Hermes backend is ready: false",
                                  "[boot] Remote Hermes backend is ready: false",
                                  "Remote Hermes backend is ready"])
def test_readiness_rejects_diagnostic_lookalikes(line):
    update = updater()
    text = "[2026-09-10T10:00:00Z] [hermes] " + line + "\n"
    assert update.readiness(text, {}, {}) is None


@pytest.mark.live_system_guard_bypass
def test_two_native_updates_retain_unique_rollbacks(bundles):
    update = updater()
    root, candidate, target = bundles
    log = root.parent / "logs/desktop.log"
    log.parent.mkdir()
    log.write_text("history\n")
    install_launch_fixture(candidate, log, "remote")
    old = update.fingerprint(target)
    try:
        first = update.complete(root, target, "main", os.getpid(), 0, "updated")
        assert first.code == 0, first.message + "\n" + log.read_text()
        first_hash = update.fingerprint(target)
        stop_fixture(target)
        # A subsequent source revision, never the patch's frozen base.
        (root / "revision").write_text("second update")
        run("git", "-C", str(root), "add", "revision")
        run("git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "next source")
        stamp_path = candidate / "Contents/Resources/install-stamp.json"
        stamp = json.loads(stamp_path.read_text())
        stamp["commit"] = update.expectation(root)["sha"]
        stamp_path.write_text(json.dumps(stamp))
        (candidate / "Contents/Resources/app.asar").write_bytes(b"second candidate")
        sign(candidate)
        second = update.complete(root, target, "main", os.getpid(), 0, "updated")
        assert second.code == 0, second.message + "\n" + log.read_text()
        retained = list(target.parent.glob(".hermes-exchange-*/Hermes.app"))
        assert len(retained) == 2
        assert {update.fingerprint(path) for path in retained} == {old, first_hash}
        result = json.loads((root.parent / ".hermes-update-result.json").read_text())
        assert result["ok"] is True and result["exit_code"] == 0
    finally:
        stop_fixture(target)


@pytest.mark.live_system_guard_bypass
def test_native_controlled_failure_restores_and_reopens_old_shell(bundles, monkeypatch):
    update = updater()
    root, candidate, target = bundles
    log = root.parent / "logs/desktop.log"
    log.parent.mkdir()
    log.write_text("history\n")
    install_launch_fixture(target, log, "remote")
    install_launch_fixture(candidate, log, "local")
    old = update.fingerprint(target)
    real_readiness = update.readiness
    failed = False

    def interrupt_after_ready(*args, **kwargs):
        nonlocal failed
        ready = real_readiness(*args, **kwargs)
        if ready and not failed:
            failed = True
            raise RuntimeError("controlled candidate failure after native readiness")
        return ready

    monkeypatch.setattr(update, "readiness", interrupt_after_ready)
    try:
        outcome = update.complete(root, target, "main", os.getpid(), 0, "updated")
        assert outcome.code == 6, outcome.message + "\n" + log.read_text()
        assert failed
        assert "restored and reopened" in outcome.message
        assert update.fingerprint(target) == old
        assert len(list(target.parent.glob(".hermes-exchange-*/Hermes.app"))) == 1
    finally:
        stop_fixture(target)


@pytest.mark.parametrize("when", ["before", "after"])
@pytest.mark.live_system_guard_bypass  # Dedicated throwaway repo; child only invokes this private helper.
def test_sigkill_preserves_canonical_and_retained_bundles(bundles, when):
    import sys
    update = updater()
    root, candidate, target = bundles
    (target / "Contents/Resources/app.asar").write_bytes(b"old shell")
    sign(target)
    old, new = update.fingerprint(target), update.fingerprint(candidate)
    script = '''import os,signal,sys
from pathlib import Path
from hermes_cli import desktop_macos_update as update
root,target,when=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
exchange=update.exchange
def abrupt(left,right):
 if left==target and when=="before":os.kill(os.getpid(),signal.SIGKILL)
 exchange(left,right)
 if left==target and when=="after":os.kill(os.getpid(),signal.SIGKILL)
update.exchange=abrupt
result=update.complete(root,target,"main",os.getpid(),0,"updated")
print(result)
'''
    result = subprocess.run([sys.executable, "-c", script, str(root), str(target), when],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == -9, result.stdout + result.stderr
    retained = list(target.parent.glob(".hermes-exchange-*/Hermes.app"))
    assert len(retained) == 1
    assert update.fingerprint(target) == (old if when == "before" else new)
    assert update.fingerprint(retained[0]) == (new if when == "before" else old)
    assert not (root.parent / ".hermes-update-result.json").exists()


def test_unsupported_exchange_refuses_before_underlying_update(bundles, monkeypatch):
    root, candidate, target = bundles
    update = updater()
    old = update.fingerprint(target)

    def unsupported(*args):
        raise OSError("filesystem does not support atomic exchange")

    monkeypatch.setattr(update, "exchange", unsupported)
    with pytest.raises(OSError):
        update.preflight(root, target)
    assert update.fingerprint(target) == old
    result = shell_handoff(root, candidate)
    assert result.returncode != 0
    assert "fixture-update" not in result.stdout
    assert not list(target.parent.glob(".hermes-exchange-*"))


def test_launch_rejection_preserves_old_shell(bundles, monkeypatch):
    update = updater()
    root, _, target = bundles
    original = update.fingerprint(target)

    def rejected(app):
        raise RuntimeError("LaunchServices refused request")

    monkeypatch.setattr(update, "launch_app", rejected)
    with pytest.raises(RuntimeError, match="LaunchServices"):
        update.adopt(target, update.expectation(root), root.parent / "desktop.log")
    assert update.fingerprint(target) == original


def test_unrelated_healthy_port_cannot_supply_local_readiness():
    import http.server
    import threading
    update = updater()
    class Healthy(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"version":"4.5.6"}')
        def log_message(self, *args):
            pass
    server = http.server.HTTPServer(("127.0.0.1", 0), Healthy)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        text = f"[{stamp}] [hermes] HERMES_BACKEND_READY port={server.server_port}\n"
        text += f"[{stamp}] [hermes] [boot] Hermes backend is ready. Finalizing desktop startup\n"
        assert update.readiness(text, {}, {"backend_version": "4.5.6"}) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.live_system_guard_bypass
def test_native_main_death_during_stability_is_not_adoption(bundles, monkeypatch):
    import signal
    update = updater()
    root, _, target = bundles
    log = root.parent / "logs/desktop.log"
    log.parent.mkdir()
    log.write_text("history\n")
    install_launch_fixture(target, log, "remote")
    real_readiness = update.readiness
    owned = {}
    samples = 0

    def die_after_stability_starts(*args, **kwargs):
        nonlocal samples
        ready = real_readiness(*args, **kwargs)
        if ready:
            samples += 1
            if samples == 2:
                mains = [p for p in owned.values() if p.executable == target / "Contents/MacOS/Hermes"]
                assert len(mains) == 1
                os.kill(mains[0].pid, signal.SIGTERM)
        return ready

    monkeypatch.setattr(update, "readiness", die_after_stability_starts)
    try:
        with pytest.raises(RuntimeError, match="Desktop died"):
            update.adopt(target, update.expectation(root), log, startup=6, stability=3, owned=owned)
        assert samples >= 2
    finally:
        update.stop_owned(owned)
        stop_fixture(target)


@pytest.mark.parametrize("damage", ["permission", "short", "zero", "repeated-guard", "pid-reuse"])
def test_mapped_scan_rejects_incomplete_or_unstable_records(monkeypatch, damage):
    import ctypes
    import errno
    import struct
    import time
    update = updater()
    process = update.Process(123, (1, 0), 1, Path("/fixture"))
    cursor_values = []
    class API:
        def proc_pidinfo(self, pid, flavor, address, data, length):
            cursor_values.append(address)
            if damage in ("permission", "pid-reuse"):
                ctypes.set_errno(errno.EACCES if damage == "permission" else errno.EINVAL)
                return 0
            if damage == "short":
                return length - 1
            struct.pack_into("=QQ", data, 80, 4096, 0)
            struct.pack_into("=I", data, 32, 31 if damage == "repeated-guard" else 0)
            return length
    monkeypatch.setattr(update, "process_record", lambda *args: update.Process(123, (2, 0), 1, Path("/fixture")))
    with pytest.raises(RuntimeError):
        update.mapped_files(API(), process, time.monotonic() + 2)
    if damage == "repeated-guard":
        assert cursor_values == [0, 4096]


def test_guard_hint_requeries_exact_address_without_skipping_mapping(monkeypatch):
    import ctypes
    import errno
    import struct
    import time
    update = updater()
    process = update.Process(123, (1, 0), 1, None)
    cursor_values = []
    class API:
        def proc_pidinfo(self, pid, flavor, address, data, length):
            cursor_values.append(address)
            if address == 8192:
                ctypes.set_errno(errno.EINVAL)
                return 0
            struct.pack_into("=QQ", data, 80, 4096, 4096 if address == 4096 else 0)
            struct.pack_into("=I", data, 32, 31)
            if address == 4096:
                struct.pack_into("=I", data, 96, 7)
                struct.pack_into("=Q", data, 104, 8)
                data[248:257] = b"/mapped\0\0"
            return length
    monkeypatch.setattr(update, "process_record", lambda *args: process)
    assert update.mapped_files(API(), process, time.monotonic() + 2) == {(7, 8): Path("/mapped")}
    assert cursor_values == [0, 4096, 8192]


def test_full_process_enumeration_buffer_is_unknown(bundles, monkeypatch):
    update = updater()
    class API:
        def proc_listpids(self, kind, uid, data, length):
            return length if data is not None else 4
    monkeypatch.setattr(update, "native", API)
    assert update.observe([bundles[2]]).state == "unknown"


@pytest.mark.parametrize("when", ["final-scan", "after-exchange"])
def test_native_reader_arriving_at_exchange_boundary_blocks_mutation(bundles, monkeypatch, when):
    update = updater()
    root, candidate, target = bundles
    (target / "Contents/Resources/app.asar").write_bytes(b"old distinct shell")
    sign(target)
    old, new = update.fingerprint(target), update.fingerprint(candidate)
    actual_gate = update.require_absent
    readers = []
    calls = 0

    def reader_arrives(paths):
        nonlocal calls
        calls += 1
        if calls == (2 if when == "final-scan" else 3):
            bundle = paths[0] if when == "final-scan" else paths[1]
            readers.append(subprocess.Popen([str(bundle / "Contents/MacOS/Hermes"), "30"]))
            # Wait for a real native identity, never replace observation with a mock.
            import time
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                process = update.process_record(update.native(), readers[0].pid)
                if process and process.executable == bundle / "Contents/MacOS/Hermes":
                    break
                time.sleep(0.02)
        return actual_gate(paths)

    monkeypatch.setattr(update, "require_absent", reader_arrives)
    try:
        outcome = update.complete(root, target, "main", os.getpid(), 0, "updated")
        assert outcome.code != 0 and readers, outcome.message
        assert "readers present" in outcome.message, outcome.message
        retained = list(target.parent.glob(".hermes-exchange-*/Hermes.app"))
        assert len(retained) == 1
        assert update.fingerprint(target) == (old if when == "final-scan" else new)
        assert update.fingerprint(retained[0]) == (new if when == "final-scan" else old)
        if when == "after-exchange":
            assert "Manual recovery required" in outcome.message
    finally:
        for process in readers:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.parametrize("reason,final_state,calls", [
    ("process changed during reader scan", "none", 2),
    ("process changed during reader scan", "present", 2),
    ("process changed during reader scan", "unknown", 3),
    ("incomplete or denied mapped-file scan", "unknown", 1),
    ("malformed mapped-file region", "unknown", 1),
])
def test_rescan_requires_a_complete_fresh_observation(monkeypatch, reason, final_state, calls):
    update = updater()
    observed = []
    def scan(paths):
        observed.append(paths)
        state = "unknown" if len(observed) == 1 else final_state
        return update.Observation(state, {Path("/fixture"): {123} if state == "present" else set()}, {}, reason)
    monkeypatch.setattr(update, "observe", scan)
    assert update.scan_readers([Path("/fixture")]).state == final_state
    assert len(observed) == calls
