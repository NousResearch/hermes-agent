"""A profile's exact listings can carry ``HERMES_`` control names across the client boundary.

The refusal of the ``HERMES_`` space is the default, not a tombstone: a launcher that starts both
the gateway and its client (a workflow runner handing its sessions a per-run pin) lists the name
exactly in ``terminal.client_env_passthrough`` and it travels like any toolchain name. What no
listing buys is a :func:`agent.session_client_env._core_owned` name — the behaviour values and the
snapshot bridge names, which core writes per session and must never accept from a client — or a
credential, which reaches tools through ``terminal.env_passthrough`` instead. The shared bash
snapshot never keeps a client environment name: it is one backend's per-session values, not shell
state (the leak class that pinned the ``HERMES_SESSION_*`` exclusions in the first place).
"""
import re

from agent.session_client_env import (BEHAVIOUR_ENV, _core_owned, _travels, accept_client_env,
                                      client_env_rules, client_env_scope, collect_client_env,
                                      tool_env, toolchain_environ)
from tools.environments.base_session_env import _SNAPSHOT_EXCLUDED_ENV_REGEX
from tools.environments.local import LocalEnvironment

LANE = 'HERMES_WF_RUN_DIR'
LANE_ID = 'HERMES_WF_RUN_ID'


def _config(names=()):
    return {'terminal': {'client_env_passthrough': list(names)}}


def _rules(names=(), **kw):
    return client_env_rules(_config(names), **kw)


def test_listing_is_the_whole_opt_in():
    rules = _rules([LANE, LANE_ID, 'HERMES_UI_SESSION_ID', 'HERMES_KERNEL_DIR'])
    assert _travels(LANE, rules['secret'], rules['names'])
    assert not _travels('HERMES_WF_OTHER', rules['secret'], rules['names'])     # listed means listed
    assert not _travels('HERMES_UNLISTED_CONTROL', rules['secret'], rules['names'])
    # A listing buys nothing on a name core owns: the bridge carries per-session identity, and the
    # behaviour values are read through session_env, both per command and per session already.
    for owned in ('HERMES_SESSION_ID', 'HERMES_UI_SESSION_ID', 'HERMES_KERNEL_DIR', 'HERMES_RPC_PORT_8',
                  'HERMES_CRON_SESSION', *BEHAVIOUR_ENV):
        assert _travels(owned, rules['secret'], rules['names']) is False
        assert _core_owned(owned)
    # The secret policy still refuses first, listing included.
    assert _travels('HERMES_WF_TOKEN', rules['secret'], rules['names']) is False


def test_the_pin_freezes_and_reaches_the_seat_and_the_listing_survives_a_retry():
    environ = {LANE: '/run/b/r1', LANE_ID: 'r1', 'HERMES_WF_OTHER': 'no', 'HERMES_UI_SESSION_ID': 'ui',
               'HERMES_TOKEN': 't', 'PATH': '/bin'}
    assert collect_client_env(environ, _config([LANE, LANE_ID, 'HERMES_WF_OTHER'])) == {
        LANE: '/run/b/r1', LANE_ID: 'r1', 'HERMES_WF_OTHER': 'no', 'PATH': '/bin'}
    frozen = accept_client_env({LANE: '/run/b/r1', LANE_ID: 'r1', 'HERMES_UI_SESSION_ID': 'ui'},
                               _config([LANE, LANE_ID]))
    assert frozen == {LANE: '/run/b/r1', LANE_ID: 'r1'}
    rules = _rules([LANE, LANE_ID])
    with client_env_scope(frozen, local=True, rules=rules):
        assert tool_env() == {LANE: '/run/b/r1', LANE_ID: 'r1'}
        assert tool_env()[LANE] == '/run/b/r1'
    assert tool_env() == {}                                            # the binding is the scope


def test_another_estate_run_and_a_stale_daemon_value_never_answer():
    sent = accept_client_env({LANE: '/run/a/r7'}, _config([LANE, LANE_ID]))
    daemon = {'PATH': '/usr/bin', LANE: '/run/b/r1', LANE_ID: 'r1', 'FOO_TOOLS': '/opt/foo'}
    with client_env_scope(sent, local=True, rules=_rules([LANE, LANE_ID])):
        env = toolchain_environ(daemon)
    assert env[LANE] == '/run/a/r7'                                    # this session's value answers
    assert LANE_ID not in env                                          # the daemon's pin for another
                                                                       # run is left behind, not served
    with client_env_scope({}, local=True, rules=_rules([LANE, LANE_ID])):
        assert LANE not in tool_env()                                  # sent none: stripped, never
    with client_env_scope(sent, local=False, rules=_rules([LANE, LANE_ID])):
        assert tool_env() == {}                                        # remote: host-naming values stay
    assert toolchain_environ(daemon) == daemon                         # unbound (Desktop) is untouched:
                                                                       # the daemon was started without
                                                                       # the pins, and the snapshot never
                                                                       # keeps one (below)


def test_the_snapshot_never_keeps_a_client_environment_name():
    # The regex contract the dump unsets by is unchanged by its new tuple source...
    assert re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX).match('declare -x HERMES_KERNEL_DIR="/tmp/k"')
    assert re.compile(_SNAPSHOT_EXCLUDED_ENV_REGEX).match(f'declare -x {LANE}="/run/b/r1"') is None
    # ...and the local backend excludes a client environment name once it has served one, monotonic
    # like the base class' profile-scoped exclusion: a name one session's value reached the dump
    # through stays out even while another session runs unbound.
    env = object.__new__(LocalEnvironment)
    env._profile_scoped_passthrough = False
    env._client_env_names_seen = set()
    assert LANE not in env._snapshot_excluded_passthrough_names()
    with client_env_scope({LANE: '/run/b/r1', LANE_ID: 'r1'}, local=True, rules=_rules([LANE, LANE_ID])):
        names = env._snapshot_excluded_passthrough_names()
    assert LANE in names and LANE_ID in names
    assert LANE in env._snapshot_excluded_passthrough_names()          # kept after the scope ends
