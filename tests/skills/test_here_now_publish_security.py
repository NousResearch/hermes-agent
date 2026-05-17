import json
import os
import stat
import subprocess
from pathlib import Path


PUBLISH_SCRIPT = Path("optional-skills/productivity/here-now/scripts/publish.sh").resolve()


CURL_STUB = '''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log_file = Path(os.environ["CURL_LOG"])
url = next((arg for arg in args if arg.startswith(("http://", "https://", "upload://"))), "")


def option_value(name):
    if name in args:
        index = args.index(name)
        if index + 1 < len(args):
            return args[index + 1]
    return None


if url.startswith("upload://"):
    data_arg = option_value("--data-binary") or ""
    local_path = data_arg[1:] if data_arg.startswith("@") else data_arg
    body = Path(local_path).read_text() if local_path else ""
    with log_file.open("a") as fh:
        fh.write(json.dumps({"event": "upload", "url": url, "local_path": local_path, "body": body}) + "\\n")
    sys.stdout.write("200")
    raise SystemExit(0)

data = option_value("-d") or "{}"
with log_file.open("a") as fh:
    fh.write(json.dumps({"event": "request", "url": url, "body": data}) + "\\n")

if url.endswith("/finalize"):
    print("{}")
    raise SystemExit(0)

request = json.loads(data)
uploads = [{"path": item["path"], "url": "upload://" + item["path"], "headers": {}} for item in request.get("files", [])]
print(json.dumps({
    "slug": request.get("slug") or "demo",
    "siteUrl": "https://demo.here.now/",
    "upload": {
        "versionId": "ver_1",
        "finalizeUrl": "https://here.now/finalize",
        "uploads": uploads,
        "skipped": [],
    },
    "claimToken": "NEW_TOKEN",
    "claimUrl": "https://here.now/claim?token=NEW_TOKEN",
    "expiresAt": "2026-02-18T01:00:00.000Z",
}))
'''


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_fake_commands(bin_dir: Path) -> None:
    _write_executable(bin_dir / "file", "#!/usr/bin/env bash\necho text/plain\n")
    _write_executable(bin_dir / "curl", CURL_STUB)


def _env_with_fake_commands(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_commands(bin_dir)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["CURL_LOG"] = str(tmp_path / "curl.log")
    return env


def test_publish_excludes_herenow_state_from_directory_upload(tmp_path: Path) -> None:
    site_dir = tmp_path / "site"
    state_dir = site_dir / ".herenow"
    state_dir.mkdir(parents=True)
    (site_dir / "index.html").write_text("hello")
    (state_dir / "state.json").write_text(json.dumps({"publishes": {"demo": {"claimToken": "SECRET"}}}))
    (state_dir / "fork-meta.json").write_text("{}")

    env = _env_with_fake_commands(tmp_path)
    result = subprocess.run(
        [str(PUBLISH_SCRIPT), ".", "--slug", "demo", "--base-url", "http://127.0.0.1", "--allow-nonherenow-base-url"],
        cwd=site_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in Path(env["CURL_LOG"]).read_text().splitlines()]
    create_body = json.loads(next(event["body"] for event in events if event["url"].endswith("/api/v1/publish/demo")))
    manifest_paths = {item["path"] for item in create_body["files"]}
    uploaded_paths = {event["url"].removeprefix("upload://") for event in events if event["event"] == "upload"}

    assert manifest_paths == {"index.html"}
    assert uploaded_paths == {"index.html"}
    assert "SECRET" not in "\n".join(event.get("body", "") for event in events if event["event"] == "upload")


def test_publish_refuses_direct_state_file_upload(tmp_path: Path) -> None:
    state_dir = tmp_path / ".herenow"
    state_dir.mkdir()
    state_file = state_dir / "state.json"
    state_file.write_text("{}")

    result = subprocess.run(
        [str(PUBLISH_SCRIPT), str(state_file), "--base-url", "http://127.0.0.1", "--allow-nonherenow-base-url"],
        cwd=tmp_path,
        env=_env_with_fake_commands(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to publish internal here.now state file" in result.stderr
