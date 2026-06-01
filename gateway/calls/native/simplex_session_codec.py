from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

_KEY_STR_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
_BASE_REVERSE_DIC: dict[str, dict[str, int]] = {}


def _get_base_value(alphabet: str, character: str) -> int:
    if alphabet not in _BASE_REVERSE_DIC:
        _BASE_REVERSE_DIC[alphabet] = {
            alphabet[i]: i for i in range(len(alphabet))
        }
    return _BASE_REVERSE_DIC[alphabet][character]


def _append_bits(
    *,
    data: list[str],
    data_val: int,
    data_position: int,
    bits_per_char: int,
    get_char_from_int,
    value: int,
    bit_count: int,
) -> tuple[int, int]:
    for _ in range(bit_count):
        data_val = (data_val << 1) | (value & 1)
        if data_position == bits_per_char - 1:
            data_position = 0
            data.append(get_char_from_int(data_val))
            data_val = 0
        else:
            data_position += 1
        value >>= 1
    return data_val, data_position


def _compress(uncompressed: str | None, bits_per_char: int, get_char_from_int) -> str:
    if uncompressed is None:
        return ""

    dictionary: dict[str, int] = {}
    dictionary_to_create: dict[str, bool] = {}
    context_w = ""
    enlarge_in = 2
    dict_size = 3
    num_bits = 2
    data: list[str] = []
    data_val = 0
    data_position = 0

    for context_c in uncompressed:
        if context_c not in dictionary:
            dictionary[context_c] = dict_size
            dict_size += 1
            dictionary_to_create[context_c] = True

        context_wc = context_w + context_c
        if context_wc in dictionary:
            context_w = context_wc
            continue

        if context_w in dictionary_to_create:
            if ord(context_w[0]) < 256:
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=0,
                    bit_count=num_bits,
                )
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=ord(context_w[0]),
                    bit_count=8,
                )
            else:
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=1,
                    bit_count=num_bits,
                )
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=ord(context_w[0]),
                    bit_count=16,
                )
            enlarge_in -= 1
            if enlarge_in == 0:
                enlarge_in = 1 << num_bits
                num_bits += 1
            del dictionary_to_create[context_w]
        else:
            data_val, data_position = _append_bits(
                data=data,
                data_val=data_val,
                data_position=data_position,
                bits_per_char=bits_per_char,
                get_char_from_int=get_char_from_int,
                value=dictionary[context_w],
                bit_count=num_bits,
            )

        enlarge_in -= 1
        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1
        dictionary[context_wc] = dict_size
        dict_size += 1
        context_w = context_c

    if context_w:
        if context_w in dictionary_to_create:
            if ord(context_w[0]) < 256:
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=0,
                    bit_count=num_bits,
                )
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=ord(context_w[0]),
                    bit_count=8,
                )
            else:
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=1,
                    bit_count=num_bits,
                )
                data_val, data_position = _append_bits(
                    data=data,
                    data_val=data_val,
                    data_position=data_position,
                    bits_per_char=bits_per_char,
                    get_char_from_int=get_char_from_int,
                    value=ord(context_w[0]),
                    bit_count=16,
                )
            enlarge_in -= 1
            if enlarge_in == 0:
                enlarge_in = 1 << num_bits
                num_bits += 1
            del dictionary_to_create[context_w]
        else:
            data_val, data_position = _append_bits(
                data=data,
                data_val=data_val,
                data_position=data_position,
                bits_per_char=bits_per_char,
                get_char_from_int=get_char_from_int,
                value=dictionary[context_w],
                bit_count=num_bits,
            )

        enlarge_in -= 1
        if enlarge_in == 0:
            num_bits += 1

    data_val, data_position = _append_bits(
        data=data,
        data_val=data_val,
        data_position=data_position,
        bits_per_char=bits_per_char,
        get_char_from_int=get_char_from_int,
        value=2,
        bit_count=num_bits,
    )

    while True:
        data_val <<= 1
        if data_position == bits_per_char - 1:
            data.append(get_char_from_int(data_val))
            break
        data_position += 1

    return "".join(data)


def _read_bits(data, reset_value: int, max_power: int) -> tuple[int, Any]:
    bits = 0
    power = 1
    while power != max_power:
        resb = data["val"] & data["position"]
        data["position"] >>= 1
        if data["position"] == 0:
            data["position"] = reset_value
            data["val"] = data["get_next_value"](data["index"])
            data["index"] += 1
        bits |= (1 if resb > 0 else 0) * power
        power <<= 1
    return bits, data


def _decompress(length: int, reset_value: int, get_next_value) -> str | None:
    dictionary: dict[int, str | int] = {0: 0, 1: 1, 2: 2}
    enlarge_in = 4
    dict_size = 4
    num_bits = 3
    result: list[str] = []
    data = {
        "val": get_next_value(0),
        "position": reset_value,
        "index": 1,
        "get_next_value": get_next_value,
    }

    bits, data = _read_bits(data, reset_value, 4)
    next_val = bits
    if next_val == 0:
        bits, data = _read_bits(data, reset_value, 256)
        c = chr(bits)
    elif next_val == 1:
        bits, data = _read_bits(data, reset_value, 65536)
        c = chr(bits)
    elif next_val == 2:
        return ""
    else:
        return None

    dictionary[3] = c
    w = c
    result.append(c)

    while True:
        if data["index"] > length:
            return ""

        bits, data = _read_bits(data, reset_value, 1 << num_bits)
        c_value = bits
        if c_value == 0:
            bits, data = _read_bits(data, reset_value, 256)
            dictionary[dict_size] = chr(bits)
            dict_size += 1
            c_value = dict_size - 1
            enlarge_in -= 1
        elif c_value == 1:
            bits, data = _read_bits(data, reset_value, 65536)
            dictionary[dict_size] = chr(bits)
            dict_size += 1
            c_value = dict_size - 1
            enlarge_in -= 1
        elif c_value == 2:
            return "".join(result)

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1

        if c_value in dictionary:
            entry = str(dictionary[c_value])
        elif c_value == dict_size:
            entry = w + w[0]
        else:
            return None

        result.append(entry)
        dictionary[dict_size] = w + entry[0]
        dict_size += 1
        enlarge_in -= 1
        w = entry

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1


def compress_to_base64(value: str | None) -> str:
    if value is None:
        return ""
    result = _compress(value, 6, lambda index: _KEY_STR_BASE64[index])
    padding = len(result) % 4
    if padding == 1:
        return result + "==="
    if padding == 2:
        return result + "=="
    if padding == 3:
        return result + "="
    return result


def decompress_from_base64(value: str | None) -> str | None:
    if value is None:
        return ""
    if value == "":
        return None
    try:
        return _decompress(
            len(value),
            32,
            lambda index: _get_base_value(_KEY_STR_BASE64, value[index]),
        )
    except (KeyError, IndexError):
        return None


def compress_json(value: Any) -> str:
    """Compress JSON with SimpleX-compatible LZ-String Base64 encoding."""
    return compress_to_base64(json.dumps(value, separators=(",", ":")))


def decompress_json(value: str) -> Any:
    """Decompress a SimpleX LZ-String Base64 JSON value."""
    decompressed = decompress_from_base64(value)
    if decompressed is None:
        raise ValueError("Failed to decompress SimpleX LZ-String data")
    try:
        return json.loads(decompressed)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to decode SimpleX LZ-String JSON data") from exc


def encode_webrtc_session(sdp: Mapping[str, Any], ice_candidates: list[Any]) -> dict[str, str]:
    return {
        "rtcSession": compress_json(dict(sdp)),
        "rtcIceCandidates": compress_json(list(ice_candidates)),
    }


def decode_webrtc_session(session: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    rtc_session = session.get("rtcSession")
    rtc_ice_candidates = session.get("rtcIceCandidates")
    if not isinstance(rtc_session, str) or not isinstance(rtc_ice_candidates, str):
        raise ValueError("SimpleX WebRTC session is missing compressed SDP or ICE")

    sdp = decompress_json(rtc_session)
    ice_candidates = decompress_json(rtc_ice_candidates)
    if not isinstance(sdp, dict) or not isinstance(ice_candidates, list):
        raise ValueError("SimpleX WebRTC session decoded to an invalid shape")
    return sdp, ice_candidates
