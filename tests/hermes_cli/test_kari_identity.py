"""A2A 身份层(hermes_cli/kari_identity.py)单测:Ed25519 签票/验票。

重点是**防伪造**:同队成员即便能自报 callerUid,没有对方私钥也签不出能过验的票。
"""

import pytest

from hermes_cli import kari_identity as ki


def test_sign_verify_ok(tmp_path):
    ki.set_key_path(str(tmp_path / "main.key"))
    main_pub = ki.public_key_b64()
    t = ki.make_ticket("mainUid", "subUid", "ctx1", "你好", ttl=60)
    ok, body, reason = ki.verify_ticket(t, main_pub, "subUid", "ctx1", "你好")
    assert ok, reason
    assert body["caller"] == "mainUid" and body["ctx"] == "ctx1"


def test_wrong_target_rejected(tmp_path):
    ki.set_key_path(str(tmp_path / "m.key"))
    pub = ki.public_key_b64()
    t = ki.make_ticket("m", "subA", "c")
    ok, _, reason = ki.verify_ticket(t, pub, "subB")  # 票是发给 subA 的,subB 不能用
    assert not ok and "target" in reason


def test_tampered_body_rejected(tmp_path):
    ki.set_key_path(str(tmp_path / "m.key"))
    pub = ki.public_key_b64()
    t = ki.make_ticket("m", "sub", "c")
    t["body"]["caller"] = "evil"  # 签名后篡改票体
    ok, _, reason = ki.verify_ticket(t, pub, "sub")
    assert not ok and "签名" in reason


def test_expired_rejected(tmp_path, monkeypatch):
    ki.set_key_path(str(tmp_path / "m.key"))
    pub = ki.public_key_b64()
    real = ki.time.time
    monkeypatch.setattr(ki.time, "time", lambda: 1000.0)  # 在"过去"签票
    t = ki.make_ticket("m", "sub", "c", ttl=60)  # exp=1060
    monkeypatch.setattr(ki.time, "time", real)  # 回到真实时间(远超 1060)
    ok, _, reason = ki.verify_ticket(t, pub, "sub")  # 签名有效但已过期
    assert not ok and "过期" in reason


def test_forgery_with_other_key_fails(tmp_path):
    """核心安全:攻击者有团队 token + 自己的密钥,自报 caller=mainUid,但用 main 的公钥验不过。"""
    ki.set_key_path(str(tmp_path / "main.key"))
    main_pub = ki.public_key_b64()
    ki.set_key_path(str(tmp_path / "evil.key"))  # 换一把不同的私钥
    forged = ki.make_ticket("mainUid", "sub", "c")  # evil 签,但票里自称 caller=mainUid
    ok, _, reason = ki.verify_ticket(forged, main_pub, "sub")  # 用 main 真实公钥验
    assert not ok and "签名" in reason


def test_pubkey_stable_and_persisted(tmp_path):
    p = str(tmp_path / "k.key")
    ki.set_key_path(p)
    pub1 = ki.public_key_b64()
    ki.set_key_path(p)  # 清缓存重读同一文件
    pub2 = ki.public_key_b64()
    assert pub1 == pub2 and len(pub1) > 20


def test_message_binding_rejected(tmp_path):
    """票绑定了请求内容(contextId+文本):换了内容再验,签名虽真也拒(防持票改内容重放)。"""
    ki.set_key_path(str(tmp_path / "m.key"))
    pub = ki.public_key_b64()
    t = ki.make_ticket("m", "sub", "c1", "原始问题")
    ok, _, reason = ki.verify_ticket(t, pub, "sub", "c1", "被改的问题")  # 文本被改
    assert not ok and "内容与票" in reason
    ok2, _, reason2 = ki.verify_ticket(t, pub, "sub", "c2", "原始问题")  # contextId 被改
    assert not ok2 and "内容与票" in reason2
