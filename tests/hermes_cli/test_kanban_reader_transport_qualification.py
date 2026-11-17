import getpass
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.kanban_generation import gc_generations


def _generated_home(tmp_path: Path):
    from tests.hermes_cli.test_kanban_corrections import _published

    board, generation = _published(tmp_path / "producer")
    home = tmp_path / "home"
    selected = home / "kanban" / "boards" / "default"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(board, target_is_directory=True)
    return home, board, generation


def _node_reader_script(path: Path) -> None:
    path.write_text(
        "const fs=require('fs'); const {spawn}=require('child_process'); "
        "const {DatabaseSync}=require('node:sqlite');\n"
        "const [python,proof,mode,releaseSignal]=process.argv.slice(2);\n"
        "const child=spawn(python,['-m','hermes_cli.main','kanban','boards','reader-session',"
        "'--board','default','--protocol-version','1','--consumer','hermes-office',"
        "'--operation-id','node-session','--json-lines'],{env:process.env});\n"
        "let pending='', ready=null; child.stdout.on('data',chunk=>{ pending+=chunk; "
        "for(;;){const at=pending.indexOf('\\n'); if(at<0) break; "
        "const line=pending.slice(0,at); pending=pending.slice(at+1); if(!line) continue; "
        "const msg=JSON.parse(line); if(msg.type==='ready'){ ready=msg; "
        "const db=new DatabaseSync(msg.database_path,{readOnly:true}); "
        "const row=db.prepare('SELECT COUNT(*) AS n FROM tasks').get(); "
        "if(mode==='kill'){child.kill('SIGKILL'); fs.writeFileSync(proof,JSON.stringify({generation_id:msg.generation_id,rows:Number(row.n),lease_id:msg.lease_id})); "
        "const timer=setInterval(()=>{if(fs.existsSync(releaseSignal)){clearInterval(timer); db.close();}},10);} "
        "else {db.close(); child.stdin.end(JSON.stringify({protocol_version:1,op:'release',lease_id:msg.lease_id})+'\\n'); "
        "fs.writeFileSync(proof,JSON.stringify({generation_id:msg.generation_id,rows:Number(row.n),lease_id:msg.lease_id}));} }}});\n"
        "child.stderr.pipe(process.stderr); child.on('exit',(code,signal)=>{ "
        "if(!ready|| (mode!=='kill' && code!==0) || (mode==='kill' && signal!=='SIGKILL')) process.exit(72); });\n"
    )


@pytest.mark.parametrize("mode", ["release", "kill"])
def test_public_reader_session_real_node_open_query_release_and_helper_loss(tmp_path, mode):
    home, board, generation = _generated_home(tmp_path)
    node = shutil.which("node")
    assert node, "managed Node runtime is required"
    script = tmp_path / "reader.cjs"
    proof = tmp_path / f"node-{mode}.json"
    release_signal = tmp_path / "release-node"
    _node_reader_script(script)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env.pop("HERMES_KANBAN_DB", None)
    command = [node, str(script), sys.executable, str(proof), mode, str(release_signal)]
    process = None
    if mode == "release":
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=30, check=False,
        )
        assert result.returncode == 0, result.stderr
    else:
        process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(300):
            if proof.is_file():
                break
            time.sleep(0.01)
        else:
            process.kill()
            pytest.fail("Node consumer did not publish helper-loss proof")
    receipt = json.loads(proof.read_text())
    assert receipt["generation_id"] == "g1" and receipt["rows"] == 0
    leases = list((generation / "leases").glob("*.json"))
    if mode == "release":
        assert leases == []
    else:
        assert process is not None
        assert len(leases) == 1
        gc_generations(board, grace_seconds=0)
        assert len(list((generation / "leases").glob("*.json"))) == 1
        release_signal.write_text("close")
        _stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        gc_generations(board, grace_seconds=0)
        assert list((generation / "leases").glob("*.json")) == []


@pytest.mark.live_system_guard_bypass
def test_public_with_reader_lease_over_disposable_local_openssh(tmp_path):
    home, _board, generation = _generated_home(tmp_path)
    sshd = shutil.which("sshd") or "/usr/sbin/sshd"
    ssh = shutil.which("ssh")
    keygen = shutil.which("ssh-keygen")
    assert Path(sshd).is_file() and ssh and keygen
    server = tmp_path / "sshd"
    server.mkdir(mode=0o700)
    host_key, client_key = server / "host_key", server / "client_key"
    subprocess.run([keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(host_key)], check=True)
    subprocess.run([keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(client_key)], check=True)
    authorized = server / "authorized_keys"
    authorized.write_bytes(client_key.with_suffix(".pub").read_bytes())
    authorized.chmod(0o600)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = server / "sshd_config"
    config.write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {host_key}\n"
        f"AuthorizedKeysFile {authorized}\nPidFile {server / 'sshd.pid'}\n"
        "StrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n"
        f"PubkeyAuthentication yes\nUsePAM no\nPermitRootLogin no\nAllowUsers {getpass.getuser()}\n"
    )
    daemon = subprocess.Popen(
        [sshd, "-D", "-e", "-f", str(config)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        for _ in range(100):
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.05)
        else:
            pytest.fail("disposable sshd did not listen")
        child = tmp_path / "ssh_reader.py"
        proof = tmp_path / "ssh-proof.json"
        child.write_text(
            "import json,os,sqlite3,struct,sys\n"
            "print('started',flush=True)\n"
            "fd=int(os.environ['HERMES_READER_BOOTSTRAP_FD']); h=os.read(fd,4); "
            "cap=json.loads(os.read(fd,struct.unpack('!I',h)[0])); "
            "conn=sqlite3.connect(f\"file:{cap['database_path']}?mode=ro\",uri=True); "
            "rows=conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]; conn.close(); "
            "open(sys.argv[1],'w').write(json.dumps({'generation_id':cap['generation_id'],'rows':rows})); "
            "sys.stdin.buffer.read()\n"
        )
        status = tmp_path / "ssh-status"
        remote = shlex.join([
            "env", f"HERMES_HOME={home}", sys.executable, "-m", "hermes_cli.main",
            "kanban", "boards", "with-reader-lease", "--board", "default",
            "--protocol-version", "1", "--consumer", "hermes-office",
            "--operation-id", "ssh-reader", "--json-status-dir", str(status),
            "--", sys.executable, str(child), str(proof),
        ])
        result = subprocess.run(
            [ssh, "-i", str(client_key), "-p", str(port), "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             f"{getpass.getuser()}@127.0.0.1", f"cd {shlex.quote(str(Path.cwd()))} && {remote}"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(proof.read_text()) == {"generation_id": "g1", "rows": 0}
        ready = json.loads((status / "ready.json").read_text())
        terminal = json.loads((status / "terminal.json").read_text())
        assert "start_token" not in ready
        assert ready["type"] == "ready" and terminal == {
            "protocol_version": 1, "operation_id": "ssh-reader",
            "type": "released", "lease_id": ready["lease_id"],
        }
        assert not list((generation / "leases").glob("*.json"))
        assert not list(status.glob(".*.tmp"))

        stopped_child = tmp_path / "ssh_stopped_reader.py"
        stopped_proof = tmp_path / "ssh-stopped-proof.json"
        stopped_child.write_text(
            "import json,os,signal,struct,sys\n"
            "print('started',flush=True)\n"
            "fd=int(os.environ['HERMES_READER_BOOTSTRAP_FD']); h=os.read(fd,4); "
            "cap=json.loads(os.read(fd,struct.unpack('!I',h)[0])); "
            "open(sys.argv[1],'w').write(json.dumps({'pid':os.getpid(),'lease_id':cap['lease_id']})); "
            "os.kill(os.getpid(),signal.SIGSTOP)\n"
        )
        stopped_status = tmp_path / "ssh-stopped-status"
        stopped_remote = shlex.join([
            "env", f"HERMES_HOME={home}", sys.executable, "-m", "hermes_cli.main",
            "kanban", "boards", "with-reader-lease", "--board", "default",
            "--protocol-version", "1", "--consumer", "hermes-office",
            "--operation-id", "ssh-disconnect", "--json-status-dir", str(stopped_status),
            "--", sys.executable, str(stopped_child), str(stopped_proof),
        ])
        transport = subprocess.Popen(
            [ssh, "-i", str(client_key), "-p", str(port), "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             f"{getpass.getuser()}@127.0.0.1",
             f"cd {shlex.quote(str(Path.cwd()))} && {stopped_remote}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(500):
            if stopped_proof.is_file() and (stopped_status / "ready.json").is_file():
                break
            time.sleep(0.01)
        else:
            transport.kill()
            pytest.fail("SSH disconnect consumer did not reach stopped read state")
        stopped_ready = json.loads((stopped_status / "ready.json").read_text())
        stopped = json.loads(stopped_proof.read_text())
        transport.kill()
        transport.wait(timeout=10)
        for pid in (int(stopped_ready["lease_keeper_pid"]),):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        gc_generations(generation.parents[1], grace_seconds=0)
        assert list((generation / "leases").glob("*.json"))
        os.kill(int(stopped["pid"]), 9)
        for _ in range(500):
            if not Path(f"/proc/{stopped['pid']}").exists():
                break
            time.sleep(0.01)
        gc_generations(generation.parents[1], grace_seconds=0)
        assert not list((generation / "leases").glob("*.json"))
        assert not (stopped_status / "terminal.json").exists()
    finally:
        daemon.terminate()
        daemon.wait(timeout=10)
