"""
yuanbao_proto.py - Yuanbao WebSocket 协议编解码（纯 Python 实现）

协议层级：
  WebSocket frame
    └── ConnMsg (protobuf: trpc.yuanbao.conn_common.ConnMsg)
          ├── head: Head  (cmd_type, cmd, seq_no, msg_id, module, ...)
          └── data: bytes  (业务 payload，标准 protobuf)
                └── InboundMessagePush / SendC2CMessageReq / SendGroupMessageReq / ...
                      (trpc.yuanbao.yuanbao_conn.yuanbao_openclaw_proxy.*)

注意：conn 层（ConnMsg）本身是标准 protobuf，不是自定义二进制格式。
     conn.proto 注释里的自定义格式（magic+head_len+body_len）仅用于 quic/tcp，
     WebSocket 直接传 ConnMsg protobuf bytes（无粘包问题，每个 ws frame = 一条消息）。

实现方式：手写 varint / protobuf wire-format 编解码，不依赖第三方 protobuf 库。
"""

from __future__ import annotations

import logging
import struct
import threading
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ============================================================
# Debug 开关
# ============================================================

DEBUG_MODE = False


def _dbg(label: str, data: bytes) -> None:
    if DEBUG_MODE:
        hex_str = " ".join(f"{b:02x}" for b in data[:64])
        ellipsis = "..." if len(data) > 64 else ""
        logger.debug("[yuanbao_proto] %s (%dB): %s", label, len(data), hex_str + ellipsis)


# ============================================================
# 常量
# ============================================================

# conn 层消息类型枚举（ConnMsg.Head.cmd_type）
PB_MSG_TYPES = {
    "ConnMsg": "trpc.yuanbao.conn_common.ConnMsg",
    "AuthBindReq": "trpc.yuanbao.conn_common.AuthBindReq",
    "AuthBindRsp": "trpc.yuanbao.conn_common.AuthBindRsp",
    "PingReq": "trpc.yuanbao.conn_common.PingReq",
    "PingRsp": "trpc.yuanbao.conn_common.PingRsp",
    "KickoutMsg": "trpc.yuanbao.conn_common.KickoutMsg",
    "DirectedPush": "trpc.yuanbao.conn_common.DirectedPush",
    "PushMsg": "trpc.yuanbao.conn_common.PushMsg",
}

# cmd_type 枚举
CMD_TYPE = {
    "Request": 0,   # 上行请求
    "Response": 1,  # 上行请求的回包
    "Push": 2,      # 下行推送
    "PushAck": 3,   # 下行推送的回包（ACK）
}

# 内置命令字
CMD = {
    "AuthBind": "auth-bind",
    "Ping": "ping",
    "Kickout": "kickout",
    "UpdateMeta": "update-meta",
}

# 内置模块名
MODULE = {
    "ConnAccess": "conn_access",
}

# biz 层服务/方法映射
# TS client uses the short name 'yuanbao_openclaw_proxy' (not the full package path)
_BIZ_PKG = "yuanbao_openclaw_proxy"
BIZ_SERVICES = {
    "InboundMessagePush": f"{_BIZ_PKG}.InboundMessagePush",
    "SendC2CMessageReq": f"{_BIZ_PKG}.SendC2CMessageReq",
    "SendC2CMessageRsp": f"{_BIZ_PKG}.SendC2CMessageRsp",
    "SendGroupMessageReq": f"{_BIZ_PKG}.SendGroupMessageReq",
    "SendGroupMessageRsp": f"{_BIZ_PKG}.SendGroupMessageRsp",
    "QueryGroupInfoReq": f"{_BIZ_PKG}.QueryGroupInfoReq",
    "QueryGroupInfoRsp": f"{_BIZ_PKG}.QueryGroupInfoRsp",
    "GetGroupMemberListReq": f"{_BIZ_PKG}.GetGroupMemberListReq",
    "GetGroupMemberListRsp": f"{_BIZ_PKG}.GetGroupMemberListRsp",
    "SendPrivateHeartbeatReq": f"{_BIZ_PKG}.SendPrivateHeartbeatReq",
    "SendPrivateHeartbeatRsp": f"{_BIZ_PKG}.SendPrivateHeartbeatRsp",
    "SendGroupHeartbeatReq": f"{_BIZ_PKG}.SendGroupHeartbeatReq",
    "SendGroupHeartbeatRsp": f"{_BIZ_PKG}.SendGroupHeartbeatRsp",
}

# openclaw instance_id（固定值 17）
HERMES_INSTANCE_ID = 17

# Reply Heartbeat 状态常量
WS_HEARTBEAT_RUNNING = 1
WS_HEARTBEAT_FINISH = 2

# ============================================================
# 序列号生成
# ============================================================

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


# ============================================================
# Protobuf wire-format 基础工具（手写，不依赖 google.protobuf）
# ============================================================

# wire types
WT_VARINT = 0
WT_64BIT = 1
WT_LEN = 2
WT_32BIT = 5


def _encode_varint(value: int) -> bytes:
    """将非负整数编码为 protobuf varint"""
    if value < 0:
        # 处理有符号负数（int32/int64 用 two's complement，64-bit）
        value = value & 0xFFFFFFFFFFFFFFFF
    out = []
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            break
    return bytes(out)


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
    """编码一个 protobuf field（tag + value）"""
    tag = (field_number << 3) | wire_type
    return _encode_varint(tag) + value


def _encode_string(s: str) -> bytes:
    """编码 protobuf string 字段的 value 部分（length-prefixed UTF-8）"""
    encoded = s.encode("utf-8")
    return _encode_varint(len(encoded)) + encoded


def _encode_bytes(b: bytes) -> bytes:
    """编码 protobuf bytes 字段的 value 部分（length-prefixed）"""
    return _encode_varint(len(b)) + b


def _encode_message(b: bytes) -> bytes:
    """编码嵌套 message（length-prefixed）"""
    return _encode_varint(len(b)) + b


def _parse_fields(data: bytes) -> list[tuple[int, int, bytes | int]]:
    """
    解析 protobuf message 的所有字段，返回 [(field_number, wire_type, raw_value), ...]
    raw_value:
      - WT_VARINT: int
      - WT_LEN: bytes
      - WT_64BIT: bytes (8 bytes)
      - WT_32BIT: bytes (4 bytes)
    """
    fields = []
    pos = 0
    n = len(data)
    while pos < n:
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == WT_VARINT:
            val, pos = _decode_varint(data, pos)
            fields.append((field_number, wire_type, val))
        elif wire_type == WT_LEN:
            length, pos = _decode_varint(data, pos)
            val = data[pos: pos + length]
            pos += length
            fields.append((field_number, wire_type, val))
        elif wire_type == WT_64BIT:
            val = data[pos: pos + 8]
            pos += 8
            fields.append((field_number, wire_type, val))
        elif wire_type == WT_32BIT:
            val = data[pos: pos + 4]
            pos += 4
            fields.append((field_number, wire_type, val))
        else:
            raise ValueError(f"unknown wire type {wire_type} at pos {pos - 1}")
    return fields


def _fields_to_dict(fields: list) -> dict[int, list]:
    """将 fields 列表转为 {field_number: [value, ...]} 字典（repeated 字段会有多个）"""
    d: dict[int, list] = {}
    for fn, wt, val in fields:
        d.setdefault(fn, []).append((wt, val))
    return d


def _get_string(fdict: dict, fn: int, default: str = "") -> str:
    """从 fields dict 取第一个 string 字段"""
    entries = fdict.get(fn)
    if not entries:
        return default
    wt, val = entries[0]
    if wt == WT_LEN and isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="replace")
    return default


def _get_varint(fdict: dict, fn: int, default: int = 0) -> int:
    """从 fields dict 取第一个 varint 字段"""
    entries = fdict.get(fn)
    if not entries:
        return default
    wt, val = entries[0]
    if wt == WT_VARINT and isinstance(val, int):
        return val
    return default


def _get_bytes(fdict: dict, fn: int, default: bytes = b"") -> bytes:
    """从 fields dict 取第一个 bytes/message 字段"""
    entries = fdict.get(fn)
    if not entries:
        return default
    wt, val = entries[0]
    if wt == WT_LEN and isinstance(val, (bytes, bytearray)):
        return bytes(val)
    return default


def _get_repeated_bytes(fdict: dict, fn: int) -> list[bytes]:
    """取所有 repeated bytes/message 字段"""
    entries = fdict.get(fn, [])
    return [bytes(val) for wt, val in entries if wt == WT_LEN]


# ============================================================
# ConnMsg 层编解码
# ============================================================
#
# ConnMsg protobuf schema (conn.json):
#   message Head {
#     uint32 cmd_type = 1;
#     string cmd      = 2;
#     uint32 seq_no   = 3;
#     string msg_id   = 4;
#     string module   = 5;
#     bool   need_ack = 6;
#     ...
#     int32  status   = 10;
#   }
#   message ConnMsg {
#     Head  head = 1;
#     bytes data = 2;
#   }


def _encode_head(
    cmd_type: int,
    cmd: str,
    seq_no: int,
    msg_id: str,
    module: str,
    need_ack: bool = False,
    status: int = 0,
) -> bytes:
    """编码 ConnMsg.Head"""
    buf = b""
    if cmd_type != 0:
        buf += _encode_field(1, WT_VARINT, _encode_varint(cmd_type))
    if cmd:
        buf += _encode_field(2, WT_LEN, _encode_string(cmd))
    if seq_no != 0:
        buf += _encode_field(3, WT_VARINT, _encode_varint(seq_no))
    if msg_id:
        buf += _encode_field(4, WT_LEN, _encode_string(msg_id))
    if module:
        buf += _encode_field(5, WT_LEN, _encode_string(module))
    if need_ack:
        buf += _encode_field(6, WT_VARINT, _encode_varint(1))
    if status != 0:
        buf += _encode_field(10, WT_VARINT, _encode_varint(status & 0xFFFFFFFFFFFFFFFF))
    return buf


def _decode_head(data: bytes) -> dict:
    """解码 ConnMsg.Head，返回 dict"""
    fdict = _fields_to_dict(_parse_fields(data))
    return {
        "cmd_type": _get_varint(fdict, 1, 0),
        "cmd": _get_string(fdict, 2, ""),
        "seq_no": _get_varint(fdict, 3, 0),
        "msg_id": _get_string(fdict, 4, ""),
        "module": _get_string(fdict, 5, ""),
        "need_ack": bool(_get_varint(fdict, 6, 0)),
        "status": _get_varint(fdict, 10, 0),
    }


def encode_conn_msg(msg_type: int, seq_no: int, data: bytes) -> bytes:
    """
    编码 ConnMsg（简化接口，对应任务要求的签名）。

    Args:
        msg_type: cmd_type（CMD_TYPE 枚举值）
        seq_no:   序列号
        data:     内层 payload bytes（业务 protobuf）

    Returns:
        ConnMsg 编码后的 bytes
    """
    head_bytes = _encode_head(
        cmd_type=msg_type,
        cmd="",
        seq_no=seq_no,
        msg_id="",
        module="",
    )
    buf = _encode_field(1, WT_LEN, _encode_message(head_bytes))
    if data:
        buf += _encode_field(2, WT_LEN, _encode_bytes(data))
    _dbg("encode_conn_msg", buf)
    return buf


def decode_conn_msg(data: bytes) -> dict:
    """
    解码 ConnMsg，返回 {msg_type, seq_no, data, head}。

    Returns:
        {
          "msg_type": int,      # cmd_type
          "seq_no":   int,
          "data":     bytes,    # 内层 payload
          "head":     dict,     # 完整 head 字段
        }
    """
    _dbg("decode_conn_msg", data)
    fdict = _fields_to_dict(_parse_fields(data))
    head_bytes = _get_bytes(fdict, 1)
    payload = _get_bytes(fdict, 2)
    head = _decode_head(head_bytes) if head_bytes else {
        "cmd_type": 0, "cmd": "", "seq_no": 0, "msg_id": "", "module": "",
        "need_ack": False, "status": 0,
    }
    return {
        "msg_type": head["cmd_type"],
        "seq_no": head["seq_no"],
        "data": payload,
        "head": head,
    }


def encode_conn_msg_full(
    cmd_type: int,
    cmd: str,
    seq_no: int,
    msg_id: str,
    module: str,
    data: bytes,
    need_ack: bool = False,
) -> bytes:
    """
    编码完整的 ConnMsg（含 cmd/msg_id/module 等 head 字段）。
    比 encode_conn_msg 提供更多 head 控制。
    """
    head_bytes = _encode_head(
        cmd_type=cmd_type,
        cmd=cmd,
        seq_no=seq_no,
        msg_id=msg_id,
        module=module,
        need_ack=need_ack,
    )
    buf = _encode_field(1, WT_LEN, _encode_message(head_bytes))
    if data:
        buf += _encode_field(2, WT_LEN, _encode_bytes(data))
    _dbg("encode_conn_msg_full", buf)
    return buf


# ============================================================
# BizMsg 层编解码（biz payload 本身也是 protobuf）
# ============================================================
#
# 任务要求的 encode_biz_msg / decode_biz_msg 是一个中间抽象层：
#   encode_biz_msg(service, method, req_id, body) -> conn_msg_bytes
#   即：将业务 body 包装成 ConnMsg，其中 head.cmd = method, head.module = service
#
# 这与 conn-codec.ts 中 buildBusinessConnMsg() 的行为一致：
#   buildBusinessConnMsg(cmd, module, bizData, msgId) -> ConnMsg bytes


def encode_biz_msg(service: str, method: str, req_id: str, body: bytes) -> bytes:
    """
    将业务 payload 包装为 ConnMsg bytes。

    Args:
        service: 模块名（head.module），如 "yuanbao_openclaw_proxy"
        method:  命令字（head.cmd），如 "send_c2c_message"
        req_id:  消息 ID（head.msg_id）
        body:    已编码的业务 protobuf bytes

    Returns:
        ConnMsg bytes（可直接发送到 WebSocket）
    """
    return encode_conn_msg_full(
        cmd_type=CMD_TYPE["Request"],
        cmd=method,
        seq_no=next_seq_no(),
        msg_id=req_id,
        module=service,
        data=body,
    )


def decode_biz_msg(data: bytes) -> dict:
    """
    解码 ConnMsg bytes，返回业务层信息。

    Returns:
        {
          "service":     str,    # head.module
          "method":      str,    # head.cmd
          "req_id":      str,    # head.msg_id
          "body":        bytes,  # 内层 biz payload
          "is_response": bool,   # cmd_type == 1 (Response)
          "head":        dict,   # 完整 head
        }
    """
    result = decode_conn_msg(data)
    head = result["head"]
    return {
        "service": head["module"],
        "method": head["cmd"],
        "req_id": head["msg_id"],
        "body": result["data"],
        "is_response": head["cmd_type"] == CMD_TYPE["Response"],
        "head": head,
    }


# ============================================================
# 业务 protobuf 消息编解码（biz payload）
# ============================================================

# ---------- MsgContent 编解码 ----------
#   field 1: text (string)
#   field 2: uuid (string)
#   field 3: image_format (uint32)
#   field 4: data (string)
#   field 5: desc (string)
#   field 6: ext (string)
#   field 7: sound (string)
#   field 8: image_info_array (repeated message)
#   field 9: index (uint32)
#   field 10: url (string)
#   field 11: file_size (uint32)
#   field 12: file_name (string)


def _encode_msg_content(content: dict) -> bytes:
    buf = b""
    for fn, key in [
        (1, "text"), (2, "uuid"), (4, "data"), (5, "desc"),
        (6, "ext"), (7, "sound"), (10, "url"), (12, "file_name"),
    ]:
        v = content.get(key, "")
        if v:
            buf += _encode_field(fn, WT_LEN, _encode_string(str(v)))
    for fn, key in [(3, "image_format"), (9, "index"), (11, "file_size")]:
        v = content.get(key, 0)
        if v:
            buf += _encode_field(fn, WT_VARINT, _encode_varint(int(v)))
    # image_info_array (repeated)
    for img in content.get("image_info_array") or []:
        img_buf = b""
        for ifn, ikey in [(1, "type"), (2, "size"), (3, "width"), (4, "height")]:
            iv = img.get(ikey, 0)
            if iv:
                img_buf += _encode_field(ifn, WT_VARINT, _encode_varint(int(iv)))
        url = img.get("url", "")
        if url:
            img_buf += _encode_field(5, WT_LEN, _encode_string(url))
        buf += _encode_field(8, WT_LEN, _encode_message(img_buf))
    return buf


def _decode_msg_content(data: bytes) -> dict:
    fdict = _fields_to_dict(_parse_fields(data))
    content: dict = {}
    for fn, key in [
        (1, "text"), (2, "uuid"), (4, "data"), (5, "desc"),
        (6, "ext"), (7, "sound"), (10, "url"), (12, "file_name"),
    ]:
        v = _get_string(fdict, fn)
        if v:
            content[key] = v
    for fn, key in [(3, "image_format"), (9, "index"), (11, "file_size")]:
        v = _get_varint(fdict, fn)
        if v:
            content[key] = v
    imgs = []
    for img_bytes in _get_repeated_bytes(fdict, 8):
        ifdict = _fields_to_dict(_parse_fields(img_bytes))
        img = {}
        for ifn, ikey in [(1, "type"), (2, "size"), (3, "width"), (4, "height")]:
            iv = _get_varint(ifdict, ifn)
            if iv:
                img[ikey] = iv
        url = _get_string(ifdict, 5)
        if url:
            img["url"] = url
        if img:
            imgs.append(img)
    if imgs:
        content["image_info_array"] = imgs
    return content


# ---------- MsgBodyElement 编解码 ----------
#   field 1: msg_type (string)  e.g. "TIMTextElem"
#   field 2: msg_content (message MsgContent)


def _encode_msg_body_element(element: dict) -> bytes:
    buf = b""
    msg_type = element.get("msg_type", "")
    if msg_type:
        buf += _encode_field(1, WT_LEN, _encode_string(msg_type))
    content = element.get("msg_content", {})
    if content:
        content_bytes = _encode_msg_content(content)
        buf += _encode_field(2, WT_LEN, _encode_message(content_bytes))
    return buf


def _decode_msg_body_element(data: bytes) -> dict:
    fdict = _fields_to_dict(_parse_fields(data))
    msg_type = _get_string(fdict, 1, "")
    content_bytes = _get_bytes(fdict, 2)
    content = _decode_msg_content(content_bytes) if content_bytes else {}
    return {"msg_type": msg_type, "msg_content": content}


# ---------- LogInfoExt ----------
#   field 1: trace_id (string)


def _encode_log_ext(trace_id: str) -> bytes:
    if not trace_id:
        return b""
    return _encode_field(1, WT_LEN, _encode_string(trace_id))


def _decode_im_msg_seq(data: bytes) -> dict:
    """Decode a single ImMsgSeq sub-message (field 17 of InboundMessagePush).

    ImMsgSeq proto fields:
      1: msg_seq (uint64)
      2: msg_id  (string)
    """
    fdict = _fields_to_dict(_parse_fields(data))
    return {
        "msg_seq": _get_varint(fdict, 1),
        "msg_id": _get_string(fdict, 2),
    }


def _decode_log_ext(data: bytes) -> dict:
    fdict = _fields_to_dict(_parse_fields(data))
    return {"trace_id": _get_string(fdict, 1)}


# ============================================================
# 入站消息解析
# ============================================================
#
# InboundMessagePush fields:
#   1: callback_command (string)
#   2: from_account (string)
#   3: to_account (string)
#   4: sender_nickname (string)
#   5: group_id (string)
#   6: group_code (string)
#   7: group_name (string)
#   8: msg_seq (uint32)
#   9: msg_random (uint32)
#   10: msg_time (uint32)
#   11: msg_key (string)
#   12: msg_id (string)
#   13: msg_body (repeated MsgBodyElement)
#   14: cloud_custom_data (string)
#   15: event_time (uint32)
#   16: bot_owner_id (string)
#   17: recall_msg_seq_list (repeated ImMsgSeq)
#   18: claw_msg_type (uint32/enum)
#   19: private_from_group_code (string)
#   20: log_ext (message LogInfoExt)


def decode_inbound_push(data: bytes) -> Optional[dict]:
    """
    解析入站消息推送的 biz payload（InboundMessagePush proto bytes）。

    Args:
        data: ConnMsg.data 字段的 bytes（即 biz payload）

    Returns:
        {
          "from_account":  str,
          "to_account":    str (可选),
          "group_code":    str (可选，群消息才有),
          "group_id":      str (可选),
          "group_name":    str (可选),
          "msg_key":       str,
          "msg_id":        str,
          "msg_seq":       int,
          "msg_random":    int,
          "msg_time":      int,
          "sender_nickname": str,
          "msg_body":      [{"msg_type": str, "msg_content": dict}, ...],
          "callback_command": str,
          "cloud_custom_data": str,
          "bot_owner_id":  str,
          "claw_msg_type": int,
          "private_from_group_code": str,
          "trace_id":      str,
          "recall_msg_seq_list": [{"msg_seq": int, "msg_id": str}, ...] 或 None,
        }
        或 None（解析失败）
    """
    try:
        _dbg("decode_inbound_push input", data)
        fdict = _fields_to_dict(_parse_fields(data))

        msg_body = []
        for el_bytes in _get_repeated_bytes(fdict, 13):
            msg_body.append(_decode_msg_body_element(el_bytes))

        log_ext_bytes = _get_bytes(fdict, 20)
        trace_id = _decode_log_ext(log_ext_bytes).get("trace_id", "") if log_ext_bytes else ""

        recall_seq_raw = _get_repeated_bytes(fdict, 17)
        recall_msg_seq_list = [_decode_im_msg_seq(b) for b in recall_seq_raw] or None

        result: dict = {
            "callback_command": _get_string(fdict, 1),
            "from_account": _get_string(fdict, 2),
            "to_account": _get_string(fdict, 3),
            "sender_nickname": _get_string(fdict, 4),
            "group_id": _get_string(fdict, 5),
            "group_code": _get_string(fdict, 6),
            "group_name": _get_string(fdict, 7),
            "msg_seq": _get_varint(fdict, 8),
            "msg_random": _get_varint(fdict, 9),
            "msg_time": _get_varint(fdict, 10),
            "msg_key": _get_string(fdict, 11),
            "msg_id": _get_string(fdict, 12),
            "msg_body": msg_body,
            "cloud_custom_data": _get_string(fdict, 14),
            "event_time": _get_varint(fdict, 15),
            "bot_owner_id": _get_string(fdict, 16),
            "recall_msg_seq_list": recall_msg_seq_list,
            "claw_msg_type": _get_varint(fdict, 18),
            "private_from_group_code": _get_string(fdict, 19),
            "trace_id": trace_id,
        }
        # 过滤空值（保持 API 整洁）
        return {k: v for k, v in result.items() if v or k in ("msg_body", "msg_seq")}
    except Exception as e:
        if DEBUG_MODE:
            logger.debug("[yuanbao_proto] decode_inbound_push failed: %s", e)
        return None


# ============================================================
# 出站消息编码
# ============================================================

def _encode_send_c2c_req(
    to_account: str,
    from_account: str,
    msg_body: list,
    msg_id: str = "",
    msg_random: int = 0,
    msg_seq: Optional[int] = None,
    group_code: str = "",
    trace_id: str = "",
) -> bytes:
    """
    编码 SendC2CMessageReq biz payload。

    SendC2CMessageReq fields:
      1: msg_id (string)
      2: to_account (string)
      3: from_account (string)
      4: msg_random (uint32)
      5: msg_body (repeated MsgBodyElement)
      6: group_code (string)
      7: msg_seq (uint64)
      8: log_ext (LogInfoExt)
    """
    buf = b""
    if msg_id:
        buf += _encode_field(1, WT_LEN, _encode_string(msg_id))
    buf += _encode_field(2, WT_LEN, _encode_string(to_account))
    if from_account:
        buf += _encode_field(3, WT_LEN, _encode_string(from_account))
    if msg_random:
        buf += _encode_field(4, WT_VARINT, _encode_varint(msg_random))
    for el in msg_body:
        el_bytes = _encode_msg_body_element(el)
        buf += _encode_field(5, WT_LEN, _encode_message(el_bytes))
    if group_code:
        buf += _encode_field(6, WT_LEN, _encode_string(group_code))
    if msg_seq is not None:
        buf += _encode_field(7, WT_VARINT, _encode_varint(msg_seq))
    if trace_id:
        log_bytes = _encode_log_ext(trace_id)
        buf += _encode_field(8, WT_LEN, _encode_message(log_bytes))
    return buf


def _encode_send_group_req(
    group_code: str,
    from_account: str,
    msg_body: list,
    msg_id: str = "",
    to_account: str = "",
    random: str = "",
    msg_seq: Optional[int] = None,
    ref_msg_id: str = "",
    trace_id: str = "",
) -> bytes:
    """
    编码 SendGroupMessageReq biz payload。

    SendGroupMessageReq fields:
      1: msg_id (string)
      2: group_code (string)
      3: from_account (string)
      4: to_account (string)
      5: random (string)
      6: msg_body (repeated MsgBodyElement)
      7: ref_msg_id (string)
      8: msg_seq (uint64)
      9: log_ext (LogInfoExt)
    """
    buf = b""
    if msg_id:
        buf += _encode_field(1, WT_LEN, _encode_string(msg_id))
    buf += _encode_field(2, WT_LEN, _encode_string(group_code))
    if from_account:
        buf += _encode_field(3, WT_LEN, _encode_string(from_account))
    if to_account:
        buf += _encode_field(4, WT_LEN, _encode_string(to_account))
    if random:
        buf += _encode_field(5, WT_LEN, _encode_string(random))
    for el in msg_body:
        el_bytes = _encode_msg_body_element(el)
        buf += _encode_field(6, WT_LEN, _encode_message(el_bytes))
    if ref_msg_id:
        buf += _encode_field(7, WT_LEN, _encode_string(ref_msg_id))
    if msg_seq is not None:
        buf += _encode_field(8, WT_VARINT, _encode_varint(msg_seq))
    if trace_id:
        log_bytes = _encode_log_ext(trace_id)
        buf += _encode_field(9, WT_LEN, _encode_message(log_bytes))
    return buf


def encode_send_c2c_message(
    to_account: str,
    msg_body: list,
    from_account: str,
    msg_id: str = "",
    msg_random: int = 0,
    msg_seq: Optional[int] = None,
    group_code: str = "",
    trace_id: str = "",
) -> bytes:
    """
    编码 C2C 发消息请求，返回完整 ConnMsg bytes（可直接发送到 WebSocket）。

    Args:
        to_account:   收件人账号
        msg_body:     消息体列表，每个元素: {"msg_type": str, "msg_content": dict}
                      例如: [{"msg_type": "TIMTextElem", "msg_content": {"text": "hello"}}]
        from_account: 发件人账号（机器人账号）
        msg_id:       消息唯一 ID（空时使用 req_id）
        msg_random:   随机数（防重）
        msg_seq:      消息序列号（可选）
        group_code:   来自群聊的私聊场景时填写
        trace_id:     链路追踪 ID

    Returns:
        ConnMsg bytes
    """
    biz_bytes = _encode_send_c2c_req(
        to_account=to_account,
        from_account=from_account,
        msg_body=msg_body,
        msg_id=msg_id,
        msg_random=msg_random,
        msg_seq=msg_seq,
        group_code=group_code,
        trace_id=trace_id,
    )
    _dbg("encode_send_c2c biz payload", biz_bytes)
    req_id = msg_id or f"c2c_{next_seq_no()}"
    return encode_conn_msg_full(
        cmd_type=CMD_TYPE["Request"],
        cmd="send_c2c_message",
        seq_no=next_seq_no(),
        msg_id=req_id,
        module=_BIZ_PKG,
        data=biz_bytes,
    )


def encode_send_group_message(
    group_code: str,
    msg_body: list,
    from_account: str,
    msg_id: str = "",
    to_account: str = "",
    random: str = "",
    msg_seq: Optional[int] = None,
    ref_msg_id: str = "",
    trace_id: str = "",
) -> bytes:
    """
    编码群消息发送请求，返回完整 ConnMsg bytes（可直接发送到 WebSocket）。

    Args:
        group_code:   群号
        msg_body:     消息体列表
        from_account: 发件人账号（机器人账号）
        msg_id:       消息唯一 ID
        to_account:   指定接收者（一般为空）
        random:       去重随机字符串
        msg_seq:      消息序列号
        ref_msg_id:   引用消息 ID
        trace_id:     链路追踪 ID

    Returns:
        ConnMsg bytes
    """
    biz_bytes = _encode_send_group_req(
        group_code=group_code,
        from_account=from_account,
        msg_body=msg_body,
        msg_id=msg_id,
        to_account=to_account,
        random=random,
        msg_seq=msg_seq,
        ref_msg_id=ref_msg_id,
        trace_id=trace_id,
    )
    _dbg("encode_send_group biz payload", biz_bytes)
    req_id = msg_id or f"grp_{next_seq_no()}"
    return encode_conn_msg_full(
        cmd_type=CMD_TYPE["Request"],
        cmd="send_group_message",
        seq_no=next_seq_no(),
        msg_id=req_id,
        module=_BIZ_PKG,
        data=biz_bytes,
    )


# ============================================================
# AuthBind / Ping 帮助函数
# ============================================================

def encode_auth_bind(
    biz_id: str,
    uid: str,
    source: str,
    token: str,
    msg_id: str,
    app_version: str = "",
    operation_system: str = "",
    bot_version: str = "",
    route_env: str = "",
) -> bytes:
    """
    构造 auth-bind 请求 ConnMsg bytes。

    AuthBindReq fields:
      1: biz_id (string)
      2: auth_info (message AuthInfo: uid=1, source=2, token=3)
      3: device_info (message DeviceInfo: app_version=1, app_operation_system=2, instance_id=10, bot_version=24)
      5: env_name (string)
    """
    # AuthInfo
    auth_buf = (
        _encode_field(1, WT_LEN, _encode_string(uid))
        + _encode_field(2, WT_LEN, _encode_string(source))
        + _encode_field(3, WT_LEN, _encode_string(token))
    )
    # DeviceInfo
    dev_buf = b""
    if app_version:
        dev_buf += _encode_field(1, WT_LEN, _encode_string(app_version))
    if operation_system:
        dev_buf += _encode_field(2, WT_LEN, _encode_string(operation_system))
    dev_buf += _encode_field(10, WT_LEN, _encode_string(str(HERMES_INSTANCE_ID)))
    if bot_version:
        dev_buf += _encode_field(24, WT_LEN, _encode_string(bot_version))

    req_buf = (
        _encode_field(1, WT_LEN, _encode_string(biz_id))
        + _encode_field(2, WT_LEN, _encode_message(auth_buf))
        + _encode_field(3, WT_LEN, _encode_message(dev_buf))
    )
    if route_env:
        req_buf += _encode_field(5, WT_LEN, _encode_string(route_env))

    return encode_conn_msg_full(
        cmd_type=CMD_TYPE["Request"],
        cmd=CMD["AuthBind"],
        seq_no=next_seq_no(),
        msg_id=msg_id,
        module=MODULE["ConnAccess"],
        data=req_buf,
    )


def encode_ping(msg_id: str) -> bytes:
    """构造 ping 请求 ConnMsg bytes（PingReq 为空消息）"""
    return encode_conn_msg_full(
        cmd_type=CMD_TYPE["Request"],
        cmd=CMD["Ping"],
        seq_no=next_seq_no(),
        msg_id=msg_id,
        module=MODULE["ConnAccess"],
        data=b"",
    )


def encode_push_ack(original_head: dict) -> bytes:
    """构造 push ACK 回包"""
    return encode_conn_msg_full(
        cmd_type=CMD_TYPE["PushAck"],
        cmd=original_head.get("cmd", ""),
        seq_no=next_seq_no(),
        msg_id=original_head.get("msg_id", ""),
        module=original_head.get("module", ""),
        data=b"",
    )


# ============================================================
# Heartbeat 编码
# ============================================================

def encode_send_private_heartbeat(
    from_account: str,
    to_account: str,
    heartbeat: int = WS_HEARTBEAT_RUNNING,
) -> bytes:
    """
    编码 SendPrivateHeartbeatReq，返回完整 ConnMsg bytes。

    SendPrivateHeartbeatReq fields:
      1: from_account (string)
      2: to_account   (string)
      3: heartbeat    (varint: RUNNING=1, FINISH=2)
    """
    buf = (
        _encode_field(1, WT_LEN, _encode_string(from_account))
        + _encode_field(2, WT_LEN, _encode_string(to_account))
        + _encode_field(3, WT_VARINT, _encode_varint(heartbeat))
    )
    req_id = f"hb_priv_{next_seq_no()}"
    return encode_biz_msg(
        service=_BIZ_PKG,
        method="send_private_heartbeat",
        req_id=req_id,
        body=buf,
    )


def encode_send_group_heartbeat(
    from_account: str,
    group_code: str,
    heartbeat: int = WS_HEARTBEAT_RUNNING,
    send_time: int = 0,
) -> bytes:
    """
    编码 SendGroupHeartbeatReq，返回完整 ConnMsg bytes。

    SendGroupHeartbeatReq fields:
      1: from_account (string)
      2: to_account   (string)  — 群场景留空
      3: group_code   (string)
      4: send_time    (int64, ms timestamp)
      5: heartbeat    (varint: RUNNING=1, FINISH=2)
    """
    import time as _time
    ts = send_time or int(_time.time() * 1000)
    buf = (
        _encode_field(1, WT_LEN, _encode_string(from_account))
        + _encode_field(2, WT_LEN, _encode_string(""))  # to_account empty for group
        + _encode_field(3, WT_LEN, _encode_string(group_code))
        + _encode_field(4, WT_VARINT, _encode_varint(ts))
        + _encode_field(5, WT_VARINT, _encode_varint(heartbeat))
    )
    req_id = f"hb_grp_{next_seq_no()}"
    return encode_biz_msg(
        service=_BIZ_PKG,
        method="send_group_heartbeat",
        req_id=req_id,
        body=buf,
    )


# ============================================================
# 群信息查询
# ============================================================

def encode_query_group_info(group_code: str) -> bytes:
    """
    编码 QueryGroupInfoReq，返回完整 ConnMsg bytes。

    QueryGroupInfoReq fields:
      1: group_code (string)
    """
    buf = _encode_field(1, WT_LEN, _encode_string(group_code))
    req_id = f"qgi_{next_seq_no()}"
    return encode_biz_msg(
        service=_BIZ_PKG,
        method="query_group_info",
        req_id=req_id,
        body=buf,
    )


def decode_query_group_info_rsp(data: bytes) -> Optional[dict]:
    """
    解码 QueryGroupInfoRsp biz payload。

    Proto 结构（对齐 TS biz-codec / member.ts queryGroupInfo）：

      message QueryGroupInfoRsp {
        int32  code       = 1;
        string message    = 2;
        GroupInfo group_info = 3;   // 嵌套 message
      }

      message GroupInfo {
        string group_name            = 1;
        string group_owner_user_id   = 2;
        string group_owner_nickname  = 3;
        uint32 group_size            = 4;
      }

    Returns:
        解码后的 dict，或 None（解析失败）
    """
    try:
        fdict = _fields_to_dict(_parse_fields(data))
        code = _get_varint(fdict, 1, 0)
        msg = _get_string(fdict, 2)

        result: dict = {"code": code}
        if msg:
            result["message"] = msg

        # field 3 = nested GroupInfo message
        gi_entries = fdict.get(3, [])
        gi_bytes = gi_entries[0][1] if gi_entries else b""
        if gi_bytes and isinstance(gi_bytes, (bytes, bytearray)):
            gi = _fields_to_dict(_parse_fields(gi_bytes))
            result["group_name"] = _get_string(gi, 1) or ""
            result["owner_id"] = _get_string(gi, 2) or ""
            result["owner_nickname"] = _get_string(gi, 3) or ""
            result["member_count"] = _get_varint(gi, 4, 0)
        else:
            result["group_name"] = ""
            result["owner_id"] = ""
            result["owner_nickname"] = ""
            result["member_count"] = 0

        return result
    except Exception:
        return None


# ============================================================
# 群成员列表查询
# ============================================================

def encode_get_group_member_list(
    group_code: str,
    offset: int = 0,
    limit: int = 200,
) -> bytes:
    """
    编码 GetGroupMemberListReq，返回完整 ConnMsg bytes。

    GetGroupMemberListReq fields:
      1: group_code (string)
      2: offset     (uint32)
      3: limit      (uint32)
    """
    buf = _encode_field(1, WT_LEN, _encode_string(group_code))
    if offset:
        buf += _encode_field(2, WT_VARINT, _encode_varint(offset))
    buf += _encode_field(3, WT_VARINT, _encode_varint(limit))
    req_id = f"gml_{next_seq_no()}"
    return encode_biz_msg(
        service=_BIZ_PKG,
        method="get_group_member_list",
        req_id=req_id,
        body=buf,
    )


def decode_get_group_member_list_rsp(data: bytes) -> Optional[dict]:
    """
    解码 GetGroupMemberListRsp biz payload。

    GetGroupMemberListRsp fields:
      1: code         (int32)
      2: message      (string)
      3: members      (repeated message MemberInfo)
      4: next_offset  (uint32)
      5: is_complete  (bool/varint)

    MemberInfo fields:
      1: user_id      (string)
      2: nickname     (string)
      3: role         (uint32)  — 0=member, 1=admin, 2=owner
      4: join_time    (uint32)
      5: name_card    (string)  — 群昵称

    Returns:
        {
          "code": int,
          "message": str,
          "members": [{"user_id": str, "nickname": str, "role": int, ...}, ...],
          "next_offset": int,
          "is_complete": bool,
        }
        或 None（解析失败）
    """
    try:
        fdict = _fields_to_dict(_parse_fields(data))
        code = _get_varint(fdict, 1, 0)

        members = []
        for member_bytes in _get_repeated_bytes(fdict, 3):
            mdict = _fields_to_dict(_parse_fields(member_bytes))
            member = {
                "user_id": _get_string(mdict, 1),
                "nickname": _get_string(mdict, 2),
                "role": _get_varint(mdict, 3),
                "join_time": _get_varint(mdict, 4),
                "name_card": _get_string(mdict, 5),
            }
            members.append({k: v for k, v in member.items() if v or k == "role"})

        return {
            "code": code,
            "message": _get_string(fdict, 2),
            "members": members,
            "next_offset": _get_varint(fdict, 4),
            "is_complete": bool(_get_varint(fdict, 5)),
        }
    except Exception:
        return None
