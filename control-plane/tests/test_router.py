import pytest

from orchard.config import Settings
from orchard.models import Employee, InboundMessage
from orchard.registry import Registry
from orchard.router import Router


class FakeSupervisor:
    def __init__(self):
        self.calls = []

    async def handle(self, emp, session, text):
        self.calls.append(emp.id)
        return f"ok:{emp.id}"


class FakeIngress:
    def __init__(self):
        self.posts = []

    async def typing(self, channel_id):
        pass

    async def post(self, channel_id, text, thread_id=None):
        self.posts.append(text)


def _settings(tmp_path):
    s = Settings()
    s.paths.root = tmp_path / "data"
    s.paths.runtime = tmp_path / "run"
    s.paths.registry_db = tmp_path / "data" / "reg.db"
    s.security.auto_provision = True
    return s


@pytest.mark.asyncio
async def test_auto_provision_once_then_route_same_profile(tmp_path):
    s = _settings(tmp_path)
    reg = Registry(s.paths.registry_db)
    sup, ing = FakeSupervisor(), FakeIngress()
    r = Router(s, reg, sup, ing)

    await r.on_message(InboundMessage(sender_id="mmuser1", channel_id="c", text="hi"))
    await r.on_message(InboundMessage(sender_id="mmuser1", channel_id="c", text="again"))

    assert len(reg.all()) == 1                       # provisioned exactly once
    assert sup.calls == ["mmuser1", "mmuser1"]        # both routed to the same profile
    assert (s.paths.home_for("mmuser1") / "config.yaml").exists()
    assert (s.paths.home_for("mmuser1") / ".env").exists()


@pytest.mark.asyncio
async def test_reject_when_not_provisioned_and_no_auto(tmp_path):
    s = _settings(tmp_path)
    s.security.auto_provision = False
    reg = Registry(s.paths.registry_db)
    sup, ing = FakeSupervisor(), FakeIngress()
    r = Router(s, reg, sup, ing)
    await r.on_message(InboundMessage(sender_id="stranger", channel_id="c", text="hi"))
    assert sup.calls == []                            # never handled
    assert reg.all() == []                            # never provisioned
    assert "workspace" in ing.posts[0].lower()        # got the rejection notice


def test_employee_id_sanitization(tmp_path):
    s = _settings(tmp_path)
    r = Router(s, Registry(s.paths.registry_db), None, None)
    for raw in ["Ab.Cd@x", "123start", "n8x3qk...weird!!", "----", "УникодЮзер"]:
        eid = r._employee_id_for(raw)
        assert Employee.valid_id(eid), f"{raw!r} -> {eid!r} is invalid"
