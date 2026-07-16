from __future__ import annotations


def run_case(case: dict):
    helper = _restart_codec_helper(case["kind"])
    if helper is None:
        from gateway.run import GatewayRunner

        if case["kind"] == "decode":
            result = GatewayRunner._decode_restart_failure_entry(case.get("value"))
        elif case["kind"] == "encode":
            result = GatewayRunner._encode_restart_failure_entry(case.get("entry") or {})
        else:
            raise AssertionError(f"unknown restart_codec case kind: {case['kind']!r}")
    else:
        if case["kind"] == "decode":
            result = helper(case.get("value"))
        elif case["kind"] == "encode":
            result = helper(case.get("entry") or {})
        else:
            raise AssertionError(f"unknown restart_codec case kind: {case['kind']!r}")
    return {"return": result, "messages": [], "db": []}


def _restart_codec_helper(kind: str):
    try:
        import gateway.fork_ext.restart_codec as restart_codec
    except ModuleNotFoundError:
        return None
    if kind == "decode":
        return restart_codec.decode_restart_failure_entry
    if kind == "encode":
        return restart_codec.encode_restart_failure_entry
    return None
