import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def test_feishu_plain_group_session_key_is_lane_scoped_across_users():
    alice = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_w3kits",
        chat_name="W3Kits Dev",
        chat_type="group",
        user_id="ou_alice",
        user_name="Alice",
    )
    bob = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_w3kits",
        chat_name="W3Kits Dev",
        chat_type="group",
        user_id="ou_bob",
        user_name="Bob",
    )

    assert build_session_key(alice) == "agent:main:feishu:group:oc_w3kits"
    assert build_session_key(alice) == build_session_key(bob)


def test_feishu_thread_session_key_still_separates_thread():
    root = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_w3kits",
        chat_type="group",
        user_id="ou_alice",
    )
    thread = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_w3kits",
        chat_type="group",
        thread_id="omt_design",
        user_id="ou_alice",
    )

    assert build_session_key(root) == "agent:main:feishu:group:oc_w3kits"
    assert build_session_key(thread) == "agent:main:feishu:group:oc_w3kits:omt_design"


def test_feishu_runner_uses_lane_session_even_when_config_is_per_user():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(group_sessions_per_user=True)
    runner.session_store = None
    source = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_w3kits",
        chat_type="group",
        user_id="ou_alice",
    )

    assert runner._session_key_for_source(source) == "agent:main:feishu:group:oc_w3kits"
