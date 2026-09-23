"""Trust is minted at native callbacks, never by serialized source flags."""
import os
from pathlib import Path
import subprocess
import sys


def _probe(tmp_path, mode):
    repo = Path(__file__).resolve().parents[2]
    home, state = tmp_path / 'home', tmp_path / 'state'
    home.mkdir(exist_ok=True)
    state.mkdir(exist_ok=True)
    env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'LANG', 'TZ') if key in os.environ}
    env.update(HOME=str(home), USERPROFILE=str(home), HERMES_HOME=str(state), PYTHONPATH=str(repo))
    result = subprocess.run([sys.executable, str(Path(__file__).parent / 'fixtures' /
                             'native_ingress_trust_peer.py'), mode], cwd=repo, env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)


def test_server_stamped_ordinary_callback_reaches_real_turn_runner(tmp_path):
    _probe(tmp_path, 'ordinary')


def test_event_fields_cannot_mint_callback_provenance(tmp_path):
    _probe(tmp_path, 'guards')


def test_multiplex_transport_policy_is_not_runtime_secret_policy(tmp_path):
    _probe(tmp_path, 'multiplex')


def test_restart_reauthorizes_current_connector_route_and_sender(tmp_path):
    _probe(tmp_path, 'capture')
    _probe(tmp_path, 'recover')
    _probe(tmp_path, 'recover-again')
