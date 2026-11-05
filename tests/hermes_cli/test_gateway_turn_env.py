"""Client half of the per-turn channel: the classic CLI sends this shell's declared per-turn values
(``terminal.client_env_turn_passthrough``) with every ``prompt.submit``, and only to a gateway that
advertises ``turn-env-v1`` — an older owner refuses the unknown field.
"""
import pytest


class _Connection:
    def __init__(self, capabilities):
        self.capabilities, self.methods = capabilities, []

    def call(self, method, **params):
        self.methods.append(method)
        return {'capabilities': self.capabilities} if method == 'runtime.describe' else {}


def _agent(capabilities):
    from hermes_cli.gateway_agent import GatewayAgent
    agent = GatewayAgent.__new__(GatewayAgent)   # only the connection matters for this seam
    agent.connection = _Connection(capabilities)
    return agent


@pytest.mark.parametrize('capabilities, sent', [
    (['turn-env-v1'], True), (['durable-admission-v1'], False), (None, False), ('turn-env-v1', False)])
def test_turn_env_is_sent_only_when_the_gateway_advertises_it(monkeypatch, capabilities, sent):
    monkeypatch.setattr('hermes_cli.gateway_chat.shell_turn_env', lambda: {'JOB_RUN_ID': 'r1'})
    agent = _agent(capabilities)
    assert agent._turn_env() == ({'JOB_RUN_ID': 'r1'} if sent else {})


def test_a_refused_describe_never_blocks_the_turn(monkeypatch):
    monkeypatch.setattr('hermes_cli.gateway_chat.shell_turn_env', lambda: {'JOB_RUN_ID': 'r1'})
    agent = _agent(['turn-env-v1'])

    def boom(method, **params):
        raise RuntimeError('no such method')

    agent.connection.call = boom
    assert agent._turn_env() == {}


def test_the_capability_lookup_happens_once_per_process(monkeypatch):
    monkeypatch.setattr('hermes_cli.gateway_chat.shell_turn_env', lambda: {'JOB_RUN_ID': 'r1'})
    agent = _agent(['turn-env-v1'])
    agent._turn_env(), agent._turn_env(), agent._turn_env()
    assert agent.connection.methods == ['runtime.describe']


def test_shell_turn_env_reads_only_the_declared_names(monkeypatch):
    from hermes_cli import gateway_chat
    monkeypatch.setattr('hermes_cli.config.get_config_path', lambda: '/tmp/config.yaml')
    monkeypatch.setattr('hermes_cli.config_effective.load_user_config_effective',
                        lambda path: {'terminal': {'client_env_turn_passthrough': ['JOB_RUN_ID']}})
    monkeypatch.setenv('JOB_RUN_ID', 'r1')
    monkeypatch.setenv('JOB_API_KEY', 'never-declared')
    assert gateway_chat.shell_turn_env() == {'JOB_RUN_ID': 'r1'}


def test_a_bypass_launch_reads_no_profile(monkeypatch):
    from hermes_cli import gateway_chat
    monkeypatch.setenv('HERMES_IGNORE_USER_CONFIG', '1')
    monkeypatch.setenv('JOB_RUN_ID', 'r1')

    def never(path):  # pragma: no cover - must not be reached
        raise AssertionError('a bypass launch reads no profile')

    monkeypatch.setattr('hermes_cli.config_effective.load_user_config_effective', never)
    assert gateway_chat.shell_turn_env() == {}
