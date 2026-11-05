"""yuanbao_proto.py - Yuanbao WebSocket 协议编解码（手写 protobuf wire-format，不依赖 google.protobuf）

每个 WebSocket frame = 一条 ConnMsg protobuf（标准 protobuf；conn.proto 注释里的 magic+len 二进制格式只用于 quic/tcp）：
  ConnMsg { Head head=1 (cmd_type, cmd, seq_no, msg_id, module, ...); bytes data=2 }
  data = 业务 payload（InboundMessagePush / SendC2CMessageReq / ...，包 trpc.yuanbao.yuanbao_conn.yuanbao_openclaw_proxy.*）
"""

from __future__ import annotations

import threading
import time
from typing import Optional

# conn 层消息类型（ConnMsg.Head.cmd_type）
PB_MSG_TYPES = {
    n: f"trpc.yuanbao.conn_common.{n}"
    for n in ("ConnMsg", "AuthBindReq", "AuthBindRsp", "PingReq", "PingRsp", "KickoutMsg", "DirectedPush", "PushMsg")
}
# cmd_type: 上行请求 / 请求回包 / 下行推送 / 推送 ACK
CMD_TYPE = {"Request": 0, "Response": 1, "Push": 2, "PushAck": 3}
CMD = {"AuthBind": "auth-bind", "Ping": "ping", "Kickout": "kickout", "UpdateMeta": "update-meta"}
MODULE = {"ConnAccess": "conn_access"}

# biz 层服务/方法映射。TS client 使用短名 'yuanbao_openclaw_proxy'（非完整包路径）。
_BIZ_PKG = "yuanbao_openclaw_proxy"
BIZ_SERVICES = {
    n: f"{_BIZ_PKG}.{n}"
    for n in ("InboundMessagePush",) + tuple(
        f"{m}{k}" for m in ("SendC2CMessage", "SendGroupMessage", "QueryGroupInfo", "GetGroupMemberList",
                            "SendPrivateHeartbeat", "SendGroupHeartbeat") for k in ("Req", "Rsp")
    )
}

HERMES_INSTANCE_ID = 17  # openclaw instance_id（固定值）
WS_HEARTBEAT_RUNNING = 1
WS_HEARTBEAT_FINISH = 2

_seq_lock = threading.Lock()
_seq_counter = 0
_SEQ_MAX = 2 ** 32 - 1  # uint32 上限


def next_seq_no() -> int:
    """生成递增序列号（线程安全，溢出时归零）"""
    global _seq_counter
    with _seq_lock:
        val = _seq_counter
        _seq_counter = (_seq_counter + 1) & _SEQ_MAX
    return val


# ---- Protobuf wire-format 基础工具

WT_VARINT = 0
WT_64BIT = 1
WT_LEN = 2
WT_32BIT = 5
_FIXED_SIZE = {WT_64BIT: 8, WT_32BIT: 4}


def _encode_varint(value: int) -> bytes:
    """protobuf varint（负数按 64-bit two's complement）"""
    if value < 0:
        value &= 0xFFFFFFFFFFFFFFFF
    out = []
    while True:
        bits = value & 0x7F
        value >>= 7
        if not value:
            out.append(bits)
            return bytes(out)
        out.append(bits | 0x80)


def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    """从 data[pos:] 解码 varint，返回 (value, new_pos)"""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
        if shift >= 64:
            raise ValueError("varint too long")
    return result, pos


def _encode_field(field_number: int, wire_type: int, value: bytes) -> bytes:
    return _encode_varint((field_number << 3) | wire_type) + value


def _encode_message(b: bytes) -> bytes:
    """length-prefixed bytes / 嵌套 message value"""
    return _encode_varint(len(b)) + b


def _encode_string(s: str) -> bytes:
    return _encode_message(s.encode("utf-8"))


# 完整 field 编码快捷方式：string / varint / 嵌套 message
def _s(fn: int, s: str) -> bytes:
    return _encode_field(fn, WT_LEN, _encode_string(s))


def _v(fn: int, n: int) -> bytes:
    return _encode_field(fn, WT_VARINT, _encode_varint(n))


def _m(fn: int, b: bytes) -> bytes:
    return _encode_field(fn, WT_LEN, _encode_message(b))


def _parse_fields(data: bytes) -> list[tuple[int, int, bytes | int]]:
    """→ [(field_number, wire_type, raw_value)]；raw_value 为 int（VARINT）或 bytes（LEN / 64BIT / 32BIT）"""
    fields = []
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        wire_type = tag & 0x07
        if wire_type == WT_VARINT:
            val, pos = _decode_varint(data, pos)
        else:
            if wire_type == WT_LEN:
                length, pos = _decode_varint(data, pos)
            elif wire_type in _FIXED_SIZE:
                length = _FIXED_SIZE[wire_type]
            else:
                raise ValueError(f"unknown wire type {wire_type} at pos {pos - 1}")
            val = data[pos: pos + length]
            pos += length
        fields.append((tag >> 3, wire_type, val))
    return fields


def _fields_to_dict(fields: list) -> dict[int, list]:
    """→ {field_number: [(wire_type, value), ...]}（repeated 字段有多个）"""
    d: dict[int, list] = {}
    for fn, wt, val in fields:
        d.setdefault(fn, []).append((wt, val))
    return d


def _parse_dict(data: bytes) -> dict[int, list]:
    return _fields_to_dict(_parse_fields(data))


def _first(fdict: dict, fn: int, wt: int):
    """第一个字段值（仅当其 wire type 匹配），无则 None"""
    entries = fdict.get(fn)
    return entries[0][1] if entries and entries[0][0] == wt else None


def _get_string(fdict: dict, fn: int, default: str = "") -> str:
    val = _first(fdict, fn, WT_LEN)
    return val.decode("utf-8", errors="replace") if isinstance(val, (bytes, bytearray)) else default


def _get_varint(fdict: dict, fn: int, default: int = 0) -> int:
    val = _first(fdict, fn, WT_VARINT)
    return val if isinstance(val, int) else default


def _get_bytes(fdict: dict, fn: int, default: bytes = b"") -> bytes:
    val = _first(fdict, fn, WT_LEN)
    return bytes(val) if isinstance(val, (bytes, bytearray)) else default


def _get_repeated_bytes(fdict: dict, fn: int) -> list[bytes]:
    return [bytes(val) for wt, val in fdict.get(fn, []) if wt == WT_LEN]


def _parse_repeated(fdict: dict, fn: int) -> list[dict]:
    return [_parse_dict(b) for b in _get_repeated_bytes(fdict, fn)]


# 字段表编码：parts = [(field_number, kind, value)]；kind:
#   "S" string 总是编码  "s" string 非空才编码  "v" varint 非零才编码  "n" varint 非 None 才编码  "m" 嵌套 bytes 非空才编码
#   "b" repeated MsgBodyElement  "t" LogInfoExt{1 trace_id} 非空才编码
_PART_ENCODERS = {
    "S": _s, "s": _s, "v": _v, "n": _v, "m": _m,
    "b": lambda fn, body: b"".join(_m(fn, _encode_msg_body_element(el)) for el in body),
    "t": lambda fn, trace_id: _m(fn, _s(1, trace_id)),
}


def _encode_parts(parts: list) -> bytes:
    buf = b""
    for fn, kind, val in parts:
        if kind == "S" or (val is not None if kind == "n" else val):
            buf += _PART_ENCODERS[kind](fn, val)
    return buf


# 字段表驱动编解码：spec = [(field_number, key, kind)]，kind:
#   "s" string（编码时 str(v)）  "r" string（原值）  "i" varint（编码时 int(v)）
# 编码跳过 falsy 值；解码只保留 truthy 值。spec 顺序即 wire 顺序和 dict 插入顺序。
_STR_KINDS = ("s", "r")


def _encode_spec(obj: dict, spec: list) -> bytes:
    buf = b""
    for fn, key, kind in spec:
        v = obj.get(key, "" if kind in _STR_KINDS else 0)
        if v:
            buf += _s(fn, str(v) if kind == "s" else v) if kind in _STR_KINDS else _v(fn, int(v))
    return buf


def _decode_spec(fdict: dict, spec: list) -> dict:
    out: dict = {}
    for fn, key, kind in spec:
        v = _get_string(fdict, fn) if kind in _STR_KINDS else _get_varint(fdict, fn)
        if v:
            out[key] = v
    return out


# ---- ConnMsg 层编解码
#   message Head { uint32 cmd_type=1; string cmd=2; uint32 seq_no=3; string msg_id=4;
#                  string module=5; bool need_ack=6; ... int32 status=10; }
#   message ConnMsg { Head head=1; bytes data=2; }


def _encode_head(
    cmd_type: int, cmd: str, seq_no: int, msg_id: str, module: str, need_ack: bool = False, status: int = 0,
) -> bytes:
    return _encode_parts([
        (1, "v", cmd_type), (2, "s", cmd), (3, "v", seq_no), (4, "s", msg_id), (5, "s", module),
        (6, "v", 1 if need_ack else 0), (10, "v", status & 0xFFFFFFFFFFFFFFFF),
    ])


def _decode_head(data: bytes) -> dict:
    fd = _parse_dict(data)
    return {
        "cmd_type": _get_varint(fd, 1), "cmd": _get_string(fd, 2), "seq_no": _get_varint(fd, 3), "msg_id": _get_string(fd, 4),
        "module": _get_string(fd, 5), "need_ack": bool(_get_varint(fd, 6)), "status": _get_varint(fd, 10),
    }


def encode_conn_msg_full(
    cmd_type: int, cmd: str, seq_no: int, msg_id: str, module: str, data: bytes, need_ack: bool = False,
) -> bytes:
    """编码完整的 ConnMsg（含 cmd/msg_id/module 等 head 字段）"""
    buf = _m(1, _encode_head(cmd_type, cmd, seq_no, msg_id, module, need_ack))
    return buf + _m(2, data) if data else buf


def encode_conn_msg(msg_type: int, seq_no: int, data: bytes) -> bytes:
    """编码 ConnMsg（简化接口：仅 cmd_type + seq_no + payload）"""
    return encode_conn_msg_full(msg_type, "", seq_no, "", "", data)


def decode_conn_msg(data: bytes) -> dict:
    """解码 ConnMsg → {msg_type, seq_no, data, head}（head 为完整 Head dict）"""
    fdict = _parse_dict(data)
    head = _decode_head(_get_bytes(fdict, 1))
    return {"msg_type": head["cmd_type"], "seq_no": head["seq_no"], "data": _get_bytes(fdict, 2), "head": head}


def _conn_request(cmd_type: int, cmd: str, msg_id: str, module: str, data: bytes = b"") -> bytes:
    return encode_conn_msg_full(cmd_type, cmd, next_seq_no(), msg_id, module, data)


# ---- BizMsg 层：业务 body 包装成 ConnMsg（head.cmd = method, head.module = service）
# 与 conn-codec.ts buildBusinessConnMsg(cmd, module, bizData, msgId) 行为一致。


def encode_biz_msg(service: str, method: str, req_id: str, body: bytes) -> bytes:
    """将已编码的业务 protobuf 包装为可直接发送的 ConnMsg bytes"""
    return _conn_request(CMD_TYPE["Request"], method, req_id, service, body)


def decode_biz_msg(data: bytes) -> dict:
    """解码 ConnMsg → {service, method, req_id, body, is_response, head}"""
    result = decode_conn_msg(data)
    head = result["head"]
    return {
        "service": head["module"], "method": head["cmd"], "req_id": head["msg_id"], "body": result["data"],
        "is_response": head["cmd_type"] == CMD_TYPE["Response"], "head": head,
    }


def _biz_request(method: str, prefix: str, body: bytes, msg_id: str = "") -> bytes:
    """biz 请求 ConnMsg；req_id 为 msg_id，空则 '<prefix>_<seq>'（seq 在 conn seq_no 之前分配）"""
    return encode_biz_msg(_BIZ_PKG, method, msg_id or f"{prefix}_{next_seq_no()}", body)


# ---- 业务 protobuf 消息编解码（biz payload）

# MsgContent：1 text, 2 uuid, 3 image_format, 4 data, 5 desc, 6 ext, 7 sound,
#   8 image_info_array (repeated), 9 index, 10 url, 11 file_size, 12 file_name,
#   999 ext_map (map<string,string>: repeated entry{1 key, 2 value})
#   ext_map key 格式 wexin_forward_msg_[forward_msg_id]_[userid]，value 为
#   base64(ForwardMsgData protobuf)（不是 JSON），用 decode_forward_msg_data() 解析。
_MSG_CONTENT_SPEC = [
    (1, "text", "s"), (2, "uuid", "s"), (4, "data", "s"), (5, "desc", "s"),
    (6, "ext", "s"), (7, "sound", "s"), (10, "url", "s"), (12, "file_name", "s"),
    (3, "image_format", "i"), (9, "index", "i"), (11, "file_size", "i"),
]
_IMAGE_INFO_SPEC = [(1, "type", "i"), (2, "size", "i"), (3, "width", "i"), (4, "height", "i"), (5, "url", "r")]
_MAP_ENTRY_SPEC = [(1, "key", "s"), (2, "value", "s")]


def _encode_msg_content(content: dict) -> bytes:
    buf = _encode_spec(content, _MSG_CONTENT_SPEC)
    for img in content.get("image_info_array") or []:
        buf += _m(8, _encode_spec(img, _IMAGE_INFO_SPEC))
    ext_map = content.get("ext_map")
    if isinstance(ext_map, dict):
        for k, v in ext_map.items():
            buf += _m(999, _encode_spec({"key": str(k), "value": str(v)}, _MAP_ENTRY_SPEC))
    return buf


def _decode_msg_content(data: bytes) -> dict:
    fdict = _parse_dict(data)
    content = _decode_spec(fdict, _MSG_CONTENT_SPEC)
    imgs = [img for img in (_decode_spec(d, _IMAGE_INFO_SPEC) for d in _parse_repeated(fdict, 8)) if img]
    ext_map = {_get_string(e, 1): _get_string(e, 2) for e in _parse_repeated(fdict, 999) if _get_string(e, 1)}
    content.update({k: v for k, v in (("image_info_array", imgs), ("ext_map", ext_map)) if v})
    return content


# MsgBodyElement：1 msg_type (string, e.g. "TIMTextElem"), 2 msg_content (MsgContent)
def _encode_msg_body_element(element: dict) -> bytes:
    content = element.get("msg_content", {})
    return _encode_parts([(1, "s", element.get("msg_type", "")), (2, "m", _encode_msg_content(content) if content else b"")])


def _decode_msg_body_element(data: bytes) -> dict:
    fdict = _parse_dict(data)
    content_bytes = _get_bytes(fdict, 2)
    return {"msg_type": _get_string(fdict, 1), "msg_content": _decode_msg_content(content_bytes) if content_bytes else {}}


# ---- 入站消息解析


# InboundMessagePush 字段表 [(field_number, key, getter)]；getter 为 _get_string / _get_varint 或自定义 (fdict, fn) -> value
_INBOUND_PUSH_SPEC = [
    (1, "callback_command", _get_string), (2, "from_account", _get_string), (3, "to_account", _get_string),
    (4, "sender_nickname", _get_string), (5, "group_id", _get_string), (6, "group_code", _get_string),
    (7, "group_name", _get_string), (8, "msg_seq", _get_varint), (9, "msg_random", _get_varint),
    (10, "msg_time", _get_varint), (11, "msg_key", _get_string), (12, "msg_id", _get_string),
    (13, "msg_body", lambda fd, fn: [_decode_msg_body_element(b) for b in _get_repeated_bytes(fd, fn)]),
    (14, "cloud_custom_data", _get_string), (15, "event_time", _get_varint), (16, "bot_owner_id", _get_string),
    (17, "recall_msg_seq_list", lambda fd, fn: [  # repeated ImMsgSeq{1 msg_seq, 2 msg_id}
        {"msg_seq": _get_varint(d, 1), "msg_id": _get_string(d, 2)} for d in _parse_repeated(fd, fn)] or None),
    (18, "claw_msg_type", _get_varint), (19, "private_from_group_code", _get_string),
    (20, "trace_id", lambda fd, fn: _get_string(_parse_dict(_get_bytes(fd, fn)), 1) if _get_bytes(fd, fn) else ""),  # LogInfoExt
]


def decode_inbound_push(data: bytes) -> Optional[dict]:
    """解析 InboundMessagePush biz payload；空值已过滤（msg_body / msg_seq 始终保留），解析失败返回 None。"""
    try:
        fdict = _parse_dict(data)
        result = {key: get(fdict, fn) for fn, key, get in _INBOUND_PUSH_SPEC}
        return {k: v for k, v in result.items() if v or k in {"msg_body", "msg_seq"}}
    except Exception:
        return None


# ---- WeChat forwarded chat-history parsing (ForwardMsgData)
# ext_map["wexin_forward_msg_<id>_<userid>"] = base64(ForwardMsgData) — protobuf, NOT JSON.
# Verified against live captures:
#   ForwardMsgData { uint32 sub_type=1 (1 = WeChat chat-history forward); uint32 begin_time=2;
#                    uint32 end_time=3; string nick_name=4 (forwarder); repeated ForwardMsg msg=5 }
#   ForwardMsg     { string sender=1; uint32 time=2; string plainText=3; repeated MsgContent msgContent=4 }
#   MsgContent     { uint32 type=1 (1=TEXT, 2=MULTIMEDIA, 3=nested forward); string text=2;
#                    repeated Multimedia multimedia=3 }
#   Multimedia     { string type=1 (image/file/document/url/video); string url=2; string file_name=4;
#                    uint32 file_size=5; uint32 width=6; uint32 height=7;
#                    string media_id=15 (usable directly as a ybres RID); string res_type=24 }
_FORWARD_MULTIMEDIA_SPEC = [(1, "type", "s"), (2, "url", "s"), (4, "file_name", "s"), (5, "file_size", "i"), (15, "media_id", "s")]


def _decode_forward_msg_content(data: bytes) -> dict:
    """MsgContent → {type, text?, multimedia?}（shape 与 _format_multimedia 对齐）"""
    fdict = _parse_dict(data)
    content: dict = {"type": _get_varint(fdict, 1)}
    if _get_string(fdict, 2):
        content["text"] = _get_string(fdict, 2)
    if _get_repeated_bytes(fdict, 3):
        content["multimedia"] = [_decode_spec(d, _FORWARD_MULTIMEDIA_SPEC) for d in _parse_repeated(fdict, 3)]
    return content


def _decode_forward_msg(fd: dict) -> dict:
    return {"sender": _get_string(fd, 1), "time": _get_varint(fd, 2), "plainText": _get_string(fd, 3),
            "msgContent": [_decode_forward_msg_content(b) for b in _get_repeated_bytes(fd, 4)]}


def decode_forward_msg_data(data: bytes) -> Optional[dict]:
    """Parse ForwardMsgData bytes (base64-decoded ext_map value) into the {sub_type, nick_name, msg, ...}
    structure consumed by ForwardedRecordsParseMiddleware.build_forward_text; None on parse failure."""
    try:
        fd = _parse_dict(data)
        return {
            "sub_type": _get_varint(fd, 1), "begin_time": _get_varint(fd, 2), "end_time": _get_varint(fd, 3),
            "nick_name": _get_string(fd, 4), "msg": [_decode_forward_msg(d) for d in _parse_repeated(fd, 5)],
        }
    except Exception:
        return None


# ---- Outbound message encoding
def encode_send_c2c_message(
    to_account: str, msg_body: list, from_account: str, msg_id: str = "", msg_random: int = 0,
    msg_seq: Optional[int] = None, group_code: str = "", trace_id: str = "",
) -> bytes:
    """SendC2CMessageReq → 完整 ConnMsg bytes（可直接发送）。

    msg_body items are {"msg_type": str, "msg_content": dict}; msg_id doubles as req_id when set;
    group_code is filled for the "private chat originating from a group" case.
    """
    return _biz_request("send_c2c_message", "c2c", _encode_parts([
        (1, "s", msg_id), (2, "S", to_account), (3, "s", from_account), (4, "v", msg_random),
        (5, "b", msg_body), (6, "s", group_code), (7, "n", msg_seq), (8, "t", trace_id),
    ]), msg_id)


def encode_send_group_message(
    group_code: str, msg_body: list, from_account: str, msg_id: str = "", to_account: str = "", random: str = "",
    msg_seq: Optional[int] = None, ref_msg_id: str = "", trace_id: str = "",
) -> bytes:
    """SendGroupMessageReq → 完整 ConnMsg bytes。to_account usually empty; ref_msg_id = quoted message."""
    return _biz_request("send_group_message", "grp", _encode_parts([
        (1, "s", msg_id), (2, "S", group_code), (3, "s", from_account), (4, "s", to_account), (5, "s", random),
        (6, "b", msg_body), (7, "s", ref_msg_id), (8, "n", msg_seq), (9, "t", trace_id),
    ]), msg_id)


# ---- AuthBind / Ping / PushAck


def encode_auth_bind(
    biz_id: str, uid: str, source: str, token: str, msg_id: str, app_version: str = "", operation_system: str = "",
    bot_version: str = "", route_env: str = "",
) -> bytes:
    """auth-bind 请求 ConnMsg bytes。

    AuthBindReq: 1 biz_id, 2 auth_info (AuthInfo{1 uid, 2 source, 3 token}),
      3 device_info (DeviceInfo{1 app_version, 2 app_operation_system, 10 instance_id, 24 bot_version}),
      5 env_name
    """
    dev_buf = _encode_parts([
        (1, "s", app_version), (2, "s", operation_system), (10, "S", str(HERMES_INSTANCE_ID)), (24, "s", bot_version),
    ])
    req_buf = _encode_parts([
        (1, "S", biz_id), (2, "m", _s(1, uid) + _s(2, source) + _s(3, token)), (3, "m", dev_buf), (5, "s", route_env),
    ])
    return _conn_request(CMD_TYPE["Request"], CMD["AuthBind"], msg_id, MODULE["ConnAccess"], req_buf)


def encode_ping(msg_id: str) -> bytes:
    """ping 请求 ConnMsg bytes（PingReq 为空消息）"""
    return _conn_request(CMD_TYPE["Request"], CMD["Ping"], msg_id, MODULE["ConnAccess"])


def encode_push_ack(original_head: dict) -> bytes:
    """push ACK 回包（回显原 head 的 cmd / msg_id / module）"""
    return _conn_request(
        CMD_TYPE["PushAck"], original_head.get("cmd", ""), original_head.get("msg_id", ""), original_head.get("module", ""),
    )


# ---- Heartbeat / 群信息 / 群成员列表


def encode_send_private_heartbeat(from_account: str, to_account: str, heartbeat: int = WS_HEARTBEAT_RUNNING) -> bytes:
    """SendPrivateHeartbeatReq{1 from_account, 2 to_account, 3 heartbeat (RUNNING=1, FINISH=2)} → ConnMsg bytes"""
    return _biz_request("send_private_heartbeat", "hb_priv", _s(1, from_account) + _s(2, to_account) + _v(3, heartbeat))


def encode_send_group_heartbeat(
    from_account: str, group_code: str, heartbeat: int = WS_HEARTBEAT_RUNNING, send_time: int = 0,
) -> bytes:
    """SendGroupHeartbeatReq{1 from_account, 2 to_account (群场景留空), 3 group_code,
    4 send_time (ms; 0 → now), 5 heartbeat} → ConnMsg bytes"""
    ts = send_time or int(time.time() * 1000)
    buf = _s(1, from_account) + _s(2, "") + _s(3, group_code) + _v(4, ts) + _v(5, heartbeat)
    return _biz_request("send_group_heartbeat", "hb_grp", buf)


def encode_query_group_info(group_code: str) -> bytes:
    """QueryGroupInfoReq{1 group_code} → ConnMsg bytes"""
    return _biz_request("query_group_info", "qgi", _s(1, group_code))


def decode_query_group_info_rsp(data: bytes) -> Optional[dict]:
    """QueryGroupInfoRsp{1 code, 2 message, 3 GroupInfo{1 group_name, 2 group_owner_user_id,
    3 group_owner_nickname, 4 group_size}} → {code, message?, group_name, owner_id, owner_nickname,
    member_count}（对齐 TS member.ts queryGroupInfo）；解析失败返回 None。"""
    try:
        fdict = _parse_dict(data)
        result: dict = {"code": _get_varint(fdict, 1)}
        if _get_string(fdict, 2):
            result["message"] = _get_string(fdict, 2)
        # field 3 taken regardless of wire type; non-bytes payloads fall back to defaults
        gi_entries = fdict.get(3, [])
        gi_bytes = gi_entries[0][1] if gi_entries else b""
        gi = _parse_dict(gi_bytes) if gi_bytes and isinstance(gi_bytes, (bytes, bytearray)) else {}
        result.update(
            group_name=_get_string(gi, 1), owner_id=_get_string(gi, 2), owner_nickname=_get_string(gi, 3),
            member_count=_get_varint(gi, 4),
        )
        return result
    except Exception:
        return None


def encode_get_group_member_list(group_code: str, offset: int = 0, limit: int = 200) -> bytes:
    """GetGroupMemberListReq{1 group_code, 2 offset, 3 limit} → ConnMsg bytes"""
    return _biz_request("get_group_member_list", "gml", _s(1, group_code) + (_v(2, offset) if offset else b"") + _v(3, limit))


def decode_get_group_member_list_rsp(data: bytes) -> Optional[dict]:
    """GetGroupMemberListRsp{1 code, 2 message, 3 members (repeated MemberInfo), 4 next_offset, 5 is_complete}；
    MemberInfo{1 user_id, 2 nickname, 3 role (0=member,1=admin,2=owner), 4 join_time, 5 name_card (群昵称)}。
    member dict 过滤空值但保留 role；解析失败返回 None。"""
    try:
        fdict = _parse_dict(data)
        members = [
            {"user_id": _get_string(m, 1), "nickname": _get_string(m, 2), "role": _get_varint(m, 3),
             "join_time": _get_varint(m, 4), "name_card": _get_string(m, 5)}
            for m in _parse_repeated(fdict, 3)
        ]
        return {
            "code": _get_varint(fdict, 1), "message": _get_string(fdict, 2),
            "members": [{k: v for k, v in mem.items() if v or k == "role"} for mem in members],
            "next_offset": _get_varint(fdict, 4), "is_complete": bool(_get_varint(fdict, 5)),
        }
    except Exception:
        return None
