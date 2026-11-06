"""Remote group sharing checks the opened source, not a fresh local copy."""
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager

import pytest

from gateway import media_fetch
from gateway.platforms import base
from tests.gateway.fixtures.hosted_room_owner import events, owner, run_turn
from tests.gateway.test_hosted_room_shared_files import sharing, staged_rows
from tools.group_file_remote import _source_policy
from tools.group_file_remote_read import read_verified


class RemoteFilesystem:
    """Execute the remote read protocol on a disposable real filesystem, without a gateway."""
    def __init__(self, root, before_open=lambda: None, supported=True):
        self._remote_home = str(root)
        self.before_open = before_open
        self.supported = supported

    def execute(self, command, **kwargs):
        self.before_open()
        if not self.supported:
            return {"returncode": 127, "output": "python3: command not found"}
        executable, *args = shlex.split(command)
        assert executable == "python3"
        result = subprocess.run([sys.executable, *args], capture_output=True, text=True, check=False)
        return {"returncode": result.returncode, "output": result.stdout + result.stderr}


@pytest.mark.platforms("posix")
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["ordinary", "hardlink", "swapped-symlink", "unsupported"])
async def test_remote_source_is_checked_before_publishing(tmp_path, monkeypatch, source):
    sandbox = tmp_path / "remote"
    sandbox.mkdir()
    file = sandbox / "report.txt"
    file.write_bytes(b"ordinary report")
    secret = sandbox / ".hermes" / "vault" / "vault.key"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"private marker")
    if source == "hardlink":
        env = sandbox / ".env"
        env.write_bytes(b"private env marker")
        file.unlink()
        os.link(env, file)

    def swap():
        if source == "swapped-symlink":
            file.unlink()
            file.symlink_to(secret)

    remote = RemoteFilesystem(sandbox, swap, supported=source != "unsupported")
    monkeypatch.setattr(media_fetch, "_active_remote_env", lambda: remote)
    validate = base.validate_media_delivery_path
    monkeypatch.setattr(base, "validate_media_delivery_path", lambda path, **kw:
                        None if str(path).startswith(str(sandbox)) else validate(path, **kw))
    async with owner(tmp_path, monkeypatch) as (authority, service, runner):
        results = []
        runner._handle_message = sharing([file], "Done.", shared=results)
        await run_turn(authority, service)
        if source == "ordinary":
            assert results[0]["shared"] is True
            message, = events(service, "message.member")
            assert message["payload"]["attachments"][0]["size"] == len(b"ordinary report")
        else:
            assert "error" in results[0], results
            assert staged_rows(service) == []
            if source == "unsupported":
                assert "verified remote" in results[0]["error"]


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("change", ["same-size", "restored-mtime", "new-hardlink", "private-directory"])
def test_remote_descriptor_checks_cover_mutations_and_the_opened_path(tmp_path, monkeypatch, change):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "report.txt"
    file.write_bytes(b"first version")
    before = file.stat()
    policy = _source_policy(str(tmp_path))
    original_open, original_fdopen = os.open, os.fdopen

    def open_then_move(path, flags, **kwargs):
        fd = original_open(path, flags, **kwargs)
        if path == file.name:
            private = tmp_path / ".hermes" / "vault"
            private.parent.mkdir()
            workspace.rename(private)
        return fd

    @contextmanager
    def changing_reader(*args, **kwargs):
        with original_fdopen(*args, **kwargs) as handle:
            yield handle
            if change == "new-hardlink":
                os.link(file, workspace / ".env")
            else:
                file.write_bytes(b"other version")
                if change == "restored-mtime":
                    os.utime(file, ns=(before.st_atime_ns, before.st_mtime_ns))

    if change == "private-directory":
        monkeypatch.setattr(os, "open", open_then_move)
    else:
        monkeypatch.setattr(os, "fdopen", changing_reader)
    with pytest.raises(ValueError, match="private|changed.*Try again"):
        read_verified(str(file), 32, policy)
