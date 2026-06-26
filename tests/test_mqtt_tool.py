from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FakeMessage:
    def __init__(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        self.topic = topic
        self.payload = payload
        self.qos = qos
        self.retain = retain


class FakePublishResult:
    rc = 0

    def wait_for_publish(self, timeout: float | None = None) -> bool:
        self.timeout = timeout
        return True

    def is_published(self) -> bool:
        return True


class FakeClient:
    queued_messages: list[FakeMessage] = []
    published: list[tuple[str, str, int, bool]] = []
    subscribed: list[tuple[str, int]] = []
    connected: list[tuple[str, int, int]] = []
    username_pw: list[tuple[str, str | None]] = []
    tls_enabled = False
    disconnected = 0

    def __init__(self, client_id: str = "", protocol=None) -> None:
        self.client_id = client_id
        self.on_message = None

    @classmethod
    def reset(cls) -> None:
        cls.queued_messages = []
        cls.published = []
        cls.subscribed = []
        cls.connected = []
        cls.username_pw = []
        cls.tls_enabled = False
        cls.disconnected = 0

    def username_pw_set(self, username: str, password: str | None = None) -> None:
        type(self).username_pw.append((username, password))

    def tls_set(self) -> None:
        type(self).tls_enabled = True

    def connect(self, host: str, port: int, keepalive: int = 60) -> None:
        type(self).connected.append((host, port, keepalive))

    def disconnect(self) -> None:
        type(self).disconnected += 1

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> FakePublishResult:
        type(self).published.append((topic, payload, qos, retain))
        return FakePublishResult()

    def subscribe(self, topic_filter: str, qos: int = 0):
        type(self).subscribed.append((topic_filter, qos))
        return (0, 1)

    def loop_start(self) -> None:
        for msg in list(type(self).queued_messages):
            if self.on_message:
                self.on_message(self, None, msg)

    def loop_stop(self) -> None:
        pass


class MQTTToolTests(unittest.TestCase):
    def setUp(self) -> None:
        for key in [
            "MQTT_HOST",
            "MQTT_PORT",
            "MQTT_USERNAME",
            "MQTT_PASSWORD",
            "MQTT_CLIENT_ID",
            "MQTT_TLS",
        ]:
            os.environ.pop(key, None)
        FakeClient.reset()

    def test_availability_requires_mqtt_host(self) -> None:
        from tools import mqtt_tool

        self.assertFalse(mqtt_tool._check_mqtt_available())
        os.environ["MQTT_HOST"] = "broker.local"
        self.assertTrue(mqtt_tool._check_mqtt_available())

    def test_publish_uses_env_config_and_validates_topic(self) -> None:
        from tools import mqtt_tool

        mqtt_tool._mqtt_client_factory = lambda: FakeClient
        os.environ.update({
            "MQTT_HOST": "broker.local",
            "MQTT_PORT": "1884",
            "MQTT_USERNAME": "iot-user",
            "MQTT_PASSWORD": "secret",
            "MQTT_TLS": "true",
        })

        result = json.loads(mqtt_tool._handle_mqtt_publish({
            "topic": "devices/lamp/cmd",
            "payload": "ON",
            "qos": 1,
            "retain": True,
        }))

        self.assertTrue(result["result"]["success"])
        self.assertEqual(FakeClient.connected, [("broker.local", 1884, 60)])
        self.assertEqual(FakeClient.username_pw, [("iot-user", "secret")])
        self.assertTrue(FakeClient.tls_enabled)
        self.assertEqual(FakeClient.published, [("devices/lamp/cmd", "ON", 1, True)])

        invalid = json.loads(mqtt_tool._handle_mqtt_publish({"topic": "devices/+/cmd", "payload": "ON"}))
        self.assertIn("error", invalid)
        self.assertIn("wildcards", invalid["error"])

    def test_subscribe_recent_collects_messages_without_requiring_broker_history(self) -> None:
        from tools import mqtt_tool

        mqtt_tool._mqtt_client_factory = lambda: FakeClient
        os.environ["MQTT_HOST"] = "broker.local"
        FakeClient.queued_messages = [
            FakeMessage("sensors/kitchen/temp", b"23.5", qos=1, retain=True),
            FakeMessage("sensors/kitchen/humidity", b"51", qos=0, retain=False),
        ]

        result = json.loads(mqtt_tool._handle_mqtt_subscribe_recent({
            "topic_filter": "sensors/kitchen/#",
            "timeout_seconds": 0.01,
            "max_messages": 5,
        }))

        self.assertEqual(result["result"]["count"], 2)
        self.assertEqual(FakeClient.subscribed, [("sensors/kitchen/#", 0)])
        self.assertEqual(result["result"]["messages"][0]["payload"], "23.5")
        self.assertTrue(result["result"]["messages"][0]["retain"])

    def test_device_command_publishes_and_optionally_waits_for_state(self) -> None:
        from tools import mqtt_tool

        mqtt_tool._mqtt_client_factory = lambda: FakeClient
        os.environ["MQTT_HOST"] = "broker.local"
        FakeClient.queued_messages = [FakeMessage("devices/lamp/state", b"ON")]

        result = json.loads(mqtt_tool._handle_mqtt_device_command({
            "command_topic": "devices/lamp/cmd",
            "payload": "ON",
            "state_topic_filter": "devices/lamp/state",
            "timeout_seconds": 0.01,
        }))

        self.assertTrue(result["result"]["success"])
        self.assertEqual(FakeClient.published, [("devices/lamp/cmd", "ON", 0, False)])
        self.assertEqual(result["result"]["state_messages"][0]["payload"], "ON")

    def test_mqtt_toolset_is_resolvable(self) -> None:
        from toolsets import resolve_toolset

        self.assertCountEqual(
            resolve_toolset("mqtt"),
            ["mqtt_publish", "mqtt_subscribe_recent", "mqtt_device_command"],
        )


if __name__ == "__main__":
    unittest.main()
